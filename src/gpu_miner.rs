//! GPU Miner (OpenCL) Module
//!
//! --- MODIFIED (Midstate Optimization) ---
//!
//! This version implements a SHA-256 "midstate" optimization.
//!
//! 1.  We split the OpenCL kernel into three parts:
//!     - `sha256_init_update`: The SHA-256 logic up to processing
//!       a partial block.
//!     - `calculate_midstate_kernel`: A new kernel run *once per job*
//!       on a *single thread*. It calculates the SHA-256 state
//!       (the "midstate") after hashing the constant `base_line`.
//!     - `find_hash_kernel`: The main mining kernel. It now
//!       takes the `midstate` as input, copies it, and only
//!       adds the unique `nonce_hex` before finalizing the hash.
//!
//! 2.  The Rust host `find_seed_gpu` is updated to:
//!     - Create a new `midstate_buf` on the GPU.
//!     - Call `calculate_midstate_kernel` once to populate it.
//!     - Pass this `midstate_buf` to `find_hash_kernel` in the
//!       main loop.
//!
//! This saves millions of redundant hash calculations per batch.
//!

use crate::TOTAL_HASHES;
use ocl::{
    builders::ProgramBuilder,
    enums::{DeviceInfo, DeviceInfoResult},
    // --- We still need OclPrm for the manual impl ---
    traits::OclPrm,
    Buffer,
    Context,
    Device,
    DeviceType,
    Kernel,
    MemFlags,
    Platform,
    Queue,
    Result as OclResult,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// --- ADDED: Define the SHA-256 Context Struct ---
// This MUST match the `sha256_ctx` struct in the OpenCL kernel.
// `OclPrm` allows this Rust struct to be sent to/from the GPU.
//
// --- MODIFIED: Added PartialEq, removed Default from derive ---
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Sha256Ctx {
    h: [u32; 8], // 32 bytes
    m: [u8; 64], // 64 bytes
    n: u64,      // 8 bytes
} // Total: 104 bytes

// --- ADDED: Manual Default implementation ---
// This is required by the OclPrm trait bound, and we can't
// derive it because of the [u8; 64] array.
impl Default for Sha256Ctx {
    fn default() -> Self {
        Sha256Ctx {
            h: [0; 8],
            m: [0; 64],
            n: 0,
        }
    }
}

// --- ADDED: Manual (unsafe) trait implementation ---
// SAFETY: This struct is `#[repr(C)]` and all its fields (u32, u8, u64)
// are valid OclPrm types. It also now implements all required bounds:
// Debug, Clone, Copy, Default, and PartialEq.
unsafe impl OclPrm for Sha256Ctx {}

// The OpenCL C kernel.
// --- MODIFIED: See new kernel file ---
const OPENCL_KERNEL_SRC: &str = include_str!("kernel.cl");

// How many hashes to run on the GPU in a single batch.
// 10 million is a good starting point. Tune this for your GPU.
const GLOBAL_WORK_SIZE: usize = 10_485_760;
const NO_RESULT: u32 = u32::MAX;

/// The GpuMiner struct holds the persistent OpenCL objects.
pub struct GpuMiner {
    context: Context,
    queue: Queue,
    // --- ADDED: We now need two kernels ---
    midstate_kernel: Kernel,
    find_hash_kernel: Kernel,
    // Human-friendly device name (e.g. GPU model)
    pub device_name: String,
}

impl GpuMiner {
    /// Creates a new GpuMiner, finds the device, and compiles the kernel.
    pub fn new() -> OclResult<Self> {
        let platform = Platform::default();
        let device = Device::list_all(platform)?
            .into_iter()
            .find(|d| {
                if let Ok(DeviceInfoResult::Type(device_type)) = d.info(DeviceInfo::Type) {
                    device_type == DeviceType::GPU
                } else {
                    false
                }
            })
            .ok_or("No GPU device found")?;

        // Capture the device name for UI display
        let device_name = device.name()?;
        crate::events::publish_event(&format!("[GPU] Using OpenCL device: {}", device_name));

        let context = Context::builder().devices(device).build()?;
        let queue = Queue::new(&context, device, None)?;

        let program = match ProgramBuilder::new()
            .devices(device)
            .src(OPENCL_KERNEL_SRC)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                // Log the error and a short snippet of the kernel source
                let src_preview: String = OPENCL_KERNEL_SRC
                    .lines()
                    .take(60)
                    .collect::<Vec<&str>>()
                    .join("\n");
                crate::events::publish_event(&format!("[GPU] OpenCL program build FAILED: {}", e));
                crate::events::publish_event("[GPU] Kernel source (first 60 lines):");
                for line in src_preview.lines() {
                    crate::events::publish_event(&format!("[GPU]   {}", line));
                }
                crate::events::publish_event(
                    "[GPU] Check your OpenCL drivers and kernel for syntax errors.",
                );
                return Err(e);
            }
        };

        // --- Kernel 1: The new midstate calculator ---
        // (This code is now valid since Sha256Ctx implements OclPrm)
        let midstate_kernel = Kernel::builder()
            .program(&program)
            .name("calculate_midstate_kernel")
            .queue(queue.clone())
            .arg_named("base_line", None::<&Buffer<u8>>)
            .arg_named("base_len", 0i32)
            .arg_named("midstate_out", None::<&Buffer<Sha256Ctx>>) // Output
            .build()?;

        // --- Kernel 2: The main hash finder ---
        // (This code is now valid since Sha256Ctx implements OclPrm)
        let find_hash_kernel = Kernel::builder()
            .program(&program)
            .name("find_hash_kernel")
            .queue(queue.clone())
            .arg_named("midstate", None::<&Buffer<Sha256Ctx>>) // Input
            .arg_named("start_nonce", 0u64)
            .arg_named("difficulty", 0u32)
            .arg_named("result_local_id", None::<&Buffer<u32>>)
            .arg_named("result_hash", None::<&Buffer<u8>>)
            .build()?;

        Ok(GpuMiner {
            context,
            queue,
            midstate_kernel,
            find_hash_kernel,
            device_name,
        })
    }

    /// --- MODIFIED: Updated signature and logic ---
    /// Finds a seed using the GPU.
    /// Returns `Ok(Some((seed, hash)))` if found.
    /// Returns `Ok(None)` if interrupted by `stop_signal`.
    /// Returns `Err` if an OpenCL error occurs.
    pub fn find_seed_gpu(
        &mut self,
        base_line: &str,
        difficulty: usize,
        stop_signal: Arc<AtomicBool>,
    ) -> OclResult<Option<(String, String)>> {
        let start_time = Instant::now();
        let mut start_nonce: u64 = rand::random();
        let base_len = base_line.len();

        // --- Create GPU Buffers ---
        let base_line_buf = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR)
            .len(base_len)
            .copy_host_slice(base_line.as_bytes())
            .build()?;

        // --- MODIFICATION: Create TWO midstate buffers ---
        // 1. A Read/Write buffer for the `midstate_kernel` to write into.
        let midstate_buf_rw: Buffer<Sha256Ctx> = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::READ_WRITE) // Written by kernel 1
            .len(1)
            .build()?;

        // 2. A Read-Only buffer for the `find_hash_kernel` to read from.
        //    This is required to use `__constant` memory.
        let midstate_buf_const: Buffer<Sha256Ctx> = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::READ_ONLY) // Read by kernel 2
            .len(1)
            .build()?;
        // --- End Modification ---

        let result_local_id_buf: Buffer<u32> = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::WRITE_ONLY | MemFlags::COPY_HOST_PTR)
            .len(1)
            .copy_host_slice(&[NO_RESULT])
            .build()?;

        let result_hash_buf: Buffer<u8> = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::WRITE_ONLY)
            .len(32)
            .build()?;

        // --- ADDED: Run Midstate Kernel *ONCE* ---
        crate::events::publish_event("[GPU] Calculating midstate for new job...");
        self.midstate_kernel.set_arg(0, &base_line_buf)?;
        self.midstate_kernel.set_arg(1, base_len as i32)?;
        // --- MODIFICATION: Write to the _rw buffer ---
        self.midstate_kernel.set_arg(2, &midstate_buf_rw)?;

        unsafe {
            self.midstate_kernel
                .cmd()
                .queue(&self.queue)
                .global_work_size(1) // Run ONCE on ONE thread
                .enq()?;
        }

        midstate_buf_rw
            .cmd()
            .copy(&midstate_buf_const, None, None)
            .queue(&self.queue)
            .enq()?;

        crate::events::publish_event(&format!(
            "[GPU] Starting hash search with batch size {GLOBAL_WORK_SIZE}..."
        ));

        // --- Main Mining Loop ---
        loop {
            // --- Check stop signal (same as before) ---
            if stop_signal.load(Ordering::Relaxed) {
                crate::events::publish_event("[GPU] Stop signal received. Aborting work.");
                return Ok(None); // Interrupted
            }

            // Reset the "found" buffer for this new batch
            result_local_id_buf
                .write(&[NO_RESULT; 1][..])
                .queue(&self.queue)
                .enq()?;

            // --- Set kernel arguments (MODIFIED) ---
            self.find_hash_kernel.set_arg(0, &midstate_buf_const)?;
            self.find_hash_kernel.set_arg(1, start_nonce)?;
            self.find_hash_kernel.set_arg(2, difficulty as u32)?;
            self.find_hash_kernel.set_arg(3, &result_local_id_buf)?;
            self.find_hash_kernel.set_arg(4, &result_hash_buf)?;

            // Launch the main kernel (same as before)
            unsafe {
                self.find_hash_kernel
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size(GLOBAL_WORK_SIZE)
                    .enq()?;
            }

            // Read back the result nonce
            let mut result_local_id = [NO_RESULT; 1];
            result_local_id_buf
                .read(&mut result_local_id[..])
                .queue(&self.queue)
                .enq()?;

            // Report hashes for this batch
            TOTAL_HASHES.fetch_add(GLOBAL_WORK_SIZE as u64, Ordering::Relaxed);

            // --- Check if we found a solution (same as before) ---
            if result_local_id[0] != NO_RESULT {
                // --- Atomic check before claiming victory (same as before) ---
                if stop_signal
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    // We were first! Read the hash and return.
                    let found_nonce = start_nonce + (result_local_id[0] as u64);

                    let seed_str = format!("{:x}", found_nonce);

                    let mut hash_bytes = [0u8; 32];
                    result_hash_buf
                        .read(&mut hash_bytes[..])
                        .queue(&self.queue)
                        .enq()?;

                    let hash_hex = hex::encode(hash_bytes);
                    let duration = start_time.elapsed();
                    let hashes_done = TOTAL_HASHES.load(Ordering::Relaxed); // Read total
                    let hash_rate = (hashes_done as f64) / duration.as_secs_f64();

                    crate::events::publish_event(&format!(
                        "+++ [GPU] Found valid seed! Nonce: {} Seed: {} Hash: {}",
                        found_nonce, seed_str, hash_hex
                    ));
                    crate::events::publish_event(&format!(
                        "GPU perf: {} hashes in {:?} ({:.2} MH/s)",
                        hashes_done,
                        duration,
                        hash_rate / 1_000_000.0
                    ));

                    // Return the (seed, hash_hex) tuple
                    return Ok(Some((seed_str, hash_hex)));
                } else {
                    // --- Race condition (same as before) ---
                    crate::events::publish_event(
                        "[GPU] Found hash, but was interrupted. Discarding.",
                    );
                    return Ok(None); // Interrupted
                }
            }

            // No solution found in this batch, prepare for the next
            start_nonce += GLOBAL_WORK_SIZE as u64;
        }
    }
}
