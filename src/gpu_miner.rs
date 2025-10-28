use crate::TOTAL_HASHES;
use ocl::{
    builders::ProgramBuilder,
    enums::{DeviceInfo, DeviceInfoResult},
    traits::OclPrm,
    Buffer, Context, Device, DeviceType, Kernel, MemFlags, Platform, Queue, Result as OclResult,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// This MUST match the `sha256_ctx` struct in the OpenCL kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Sha256Ctx {
    h: [u32; 8], // 32 bytes
    m: [u8; 64], // 64 bytes
    n: u64,      // 8 bytes
} // Total: 104 bytes

// Manual Default implementation
impl Default for Sha256Ctx {
    fn default() -> Self {
        Sha256Ctx {
            h: [0; 8],
            m: [0; 64],
            n: 0,
        }
    }
}

// SAFETY: This struct is `#[repr(C)]` and all its fields are valid OclPrm types.
unsafe impl OclPrm for Sha256Ctx {}

// The OpenCL C kernel.
const OPENCL_KERNEL_SRC: &str = include_str!("kernel.cl");

// --- Tuned Work Sizes ---
// Must be a multiple of 64 or 256 for optimal GPU occupancy.
const LOCAL_WORK_SIZE: usize = 256;
// 10.4M is a good starting point. (256 * 40960)
const GLOBAL_WORK_SIZE: usize = 10_485_760;
const NO_RESULT: u32 = u32::MAX;

/// The GpuMiner struct holds the persistent OpenCL objects.
pub struct GpuMiner {
    context: Context,
    queue: Queue,
    // --- Renamed kernel 1 ---
    calculate_setup_kernel: Kernel,
    find_hash_kernel: Kernel,
    pub device_name: String,

    // --- ADDED: Persistent buffers for the final block template ---
    final_template_buf: Buffer<u8>,
    n_mod_64_buf: Buffer<u32>,
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

        let device_name = device.name()?;
        crate::events::publish_event(&format!("[GPU] Using OpenCL device: {}", device_name));

        let context = Context::builder().devices(device).build()?;
        let queue = Queue::new(&context, device, None)?;

        let program = match ProgramBuilder::new()
            .devices(device)
            .src(OPENCL_KERNEL_SRC)
            // --- ADDED: Compiler flags for micro-optimization ---
            .cmplr_opt("-cl-mad-enable") // Enable Fused Multiply-Add (good for SHA-256)
            .cmplr_opt("-cl-no-signed-zeros") // Optimizes floating point (not relevant, but no harm)
            .build(&context)
        {
            Ok(p) => p,
            Err(e) => {
                // (Error logging unchanged)
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

        // --- Kernel 1: The new setup kernel (calculates 3 things) ---
        let calculate_setup_kernel = Kernel::builder()
            .program(&program)
            .name("calculate_setup_kernel") // <-- Renamed kernel
            .queue(queue.clone())
            .arg_named("base_line", None::<&Buffer<u8>>)
            .arg_named("base_len", 0i32)
            .arg_named("midstate_out", None::<&Buffer<Sha256Ctx>>) // Output 1
            .arg_named("final_block_template_out", None::<&Buffer<u8>>) // Output 2
            .arg_named("n_mod_64_nonce_start_out", None::<&Buffer<u32>>) // Output 3
            .build()?;

        // --- Kernel 2: The main hash finder (now takes 3 constants) ---
        let find_hash_kernel = Kernel::builder()
            .program(&program)
            .name("find_hash_kernel")
            .queue(queue.clone())
            .arg_named("midstate", None::<&Buffer<Sha256Ctx>>) // Input 1
            .arg_named("final_block_template", None::<&Buffer<u8>>) // Input 2
            .arg_named("n_mod_64_nonce_start", 0u32) // Input 3 (by value)
            .arg_named("start_nonce", 0u64)
            .arg_named("difficulty", 0u32)
            .arg_named("result_local_id", None::<&Buffer<u32>>)
            .arg_named("result_hash", None::<&Buffer<u8>>)
            .build()?;

        // --- ADDED: Create the persistent template buffers ---
        let final_template_buf: Buffer<u8> = Buffer::builder()
            .context(&context)
            .flags(MemFlags::READ_WRITE) // Written by kernel 1, read by kernel 2
            .len(64)
            .build()?;

        let n_mod_64_buf: Buffer<u32> = Buffer::builder()
            .context(&context)
            .flags(MemFlags::READ_WRITE) // Written by kernel 1, read by host
            .len(1)
            .build()?;

        Ok(GpuMiner {
            context,
            queue,
            calculate_setup_kernel,
            find_hash_kernel,
            device_name,
            final_template_buf, // Add to struct
            n_mod_64_buf,       // Add to struct
        })
    }

    /// Finds a seed using the GPU.
    pub fn find_seed_gpu(
        &mut self,
        base_line: &str,
        difficulty: usize,
        stop_signal: Arc<AtomicBool>,
    ) -> OclResult<Option<(String, String)>> {
        let start_time = Instant::now();
        let mut start_nonce: u64 = rand::random();
        let base_len = base_line.len();

        // --- Create GPU Buffers (Host -> Device) ---
        let base_line_buf = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR)
            .len(base_len)
            .copy_host_slice(base_line.as_bytes())
            .build()?;

        // Midstate RW buffer (written by setup kernel)
        let midstate_buf_rw: Buffer<Sha256Ctx> = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::READ_WRITE)
            .len(1)
            .build()?;

        // Midstate RO buffer (read by main kernel)
        let midstate_buf_const: Buffer<Sha256Ctx> = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::READ_ONLY)
            .len(1)
            .build()?;

        // --- Create GPU Buffers (Device -> Host) ---
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

        // --- Run Setup Kernel *ONCE* ---
        crate::events::publish_event("[GPU] Calculating midstate and final block template...");
        self.calculate_setup_kernel.set_arg(0, &base_line_buf)?;
        self.calculate_setup_kernel.set_arg(1, base_len as i32)?;
        self.calculate_setup_kernel.set_arg(2, &midstate_buf_rw)?;
        self.calculate_setup_kernel
            .set_arg(3, &self.final_template_buf)?;
        self.calculate_setup_kernel.set_arg(4, &self.n_mod_64_buf)?;

        unsafe {
            self.calculate_setup_kernel
                .cmd()
                .queue(&self.queue)
                .global_work_size(1) // Run ONCE on ONE thread
                .enq()?;
        }

        // Copy midstate from RW buffer to __constant RO buffer
        midstate_buf_rw
            .cmd()
            .copy(&midstate_buf_const, None, None)
            .queue(&self.queue)
            .enq()?;

        // --- Read back the nonce offset *ONCE* ---
        let mut n_mod_64_host = [0u32; 1];
        self.n_mod_64_buf
            .read(&mut n_mod_64_host[..])
            .queue(&self.queue)
            .enq()?;
        // We *must* block here to get this value before the main loop.
        // This is fine, it's a tiny read and only happens once.
        self.queue.finish()?;
        let n_mod_64_val = n_mod_64_host[0];

        crate::events::publish_event(&format!(
            "[GPU] Starting hash search with batch size {GLOBAL_WORK_SIZE} (local size {LOCAL_WORK_SIZE})..."
        ));

        // --- Main Mining Loop ---
        loop {
            if stop_signal.load(Ordering::Relaxed) {
                crate::events::publish_event("[GPU] Stop signal received. Aborting work.");
                return Ok(None);
            }

            // Reset the "found" buffer
            result_local_id_buf
                .write(&[NO_RESULT; 1][..])
                .queue(&self.queue)
                .enq()?;

            // --- Set kernel arguments (MODIFIED) ---
            self.find_hash_kernel.set_arg(0, &midstate_buf_const)?;
            self.find_hash_kernel.set_arg(1, &self.final_template_buf)?;
            self.find_hash_kernel.set_arg(2, n_mod_64_val)?; // Pass by value
            self.find_hash_kernel.set_arg(3, start_nonce)?;
            self.find_hash_kernel.set_arg(4, difficulty as u32)?;
            self.find_hash_kernel.set_arg(5, &result_local_id_buf)?;
            self.find_hash_kernel.set_arg(6, &result_hash_buf)?;

            // Launch the main kernel
            unsafe {
                self.find_hash_kernel
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size(GLOBAL_WORK_SIZE)
                    .local_work_size(LOCAL_WORK_SIZE) // Explicitly set
                    .enq()?;
            }

            // Read back the result ID
            let mut result_local_id = [NO_RESULT; 1];
            result_local_id_buf
                .read(&mut result_local_id[..])
                .queue(&self.queue)
                .enq()?;

            // Report hashes for this batch
            TOTAL_HASHES.fetch_add(GLOBAL_WORK_SIZE as u64, Ordering::Relaxed);

            // --- Check if we found a solution ---
            if result_local_id[0] != NO_RESULT {
                // (Victory/Race condition logic is unchanged)
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
                    let hashes_done = TOTAL_HASHES.load(Ordering::Relaxed);
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

                    return Ok(Some((seed_str, hash_hex)));
                } else {
                    crate::events::publish_event(
                        "[GPU] Found hash, but was interrupted. Discarding.",
                    );
                    return Ok(None);
                }
            }
            start_nonce += GLOBAL_WORK_SIZE as u64;
        }
    }
}
