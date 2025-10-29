use crate::TOTAL_HASHES;
use rust_cuda::{
    context::{CudaContext, CudaStream},
    device::CudaDevice,
    launch,
    memory::DeviceBuffer,
    module::CudaModule,
};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// Load the PTX file that build.rs compiled for us.
const KERNEL_PTX: &[u8] = include_bytes!(env!("NVCC_PTX_PATH"));
const LOCAL_WORK_SIZE: u32 = 256;
const GLOBAL_WORK_SIZE: u32 = 41_943_040; // 10_485_760
const GRID_SIZE: u32 = GLOBAL_WORK_SIZE / LOCAL_WORK_SIZE;
const NO_RESULT: u32 = u32::MAX;

// This MUST match the `sha256_ctx` struct in the CUDA kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Sha256Ctx {
    h: [u32; 8], // 32 bytes
    m: [u8; 64], // 64 bytes
    n: u64,      // 8 bytes
} // Total: 104 bytes

// (Default impl is unchanged)
impl Default for Sha256Ctx {
    fn default() -> Self {
        Sha256Ctx {
            h: [0; 8],
            m: [0; 64],
            n: 0,
        }
    }
}
// SAFETY: This struct is repr(C) and used for DeviceBuffer
unsafe impl rust_cuda::memory::Pod for Sha256Ctx {}

pub struct GpuMiner {
    // A CUDA context is implicitly managed by the CudaDevice
    _device: CudaDevice,
    _context: CudaContext,
    stream: CudaStream,
    module: CudaModule,
    pub device_name: String,

    // Persistent buffers
    final_template_buf: DeviceBuffer<u8>,
    n_mod_64_buf: DeviceBuffer<u32>,
    base_line_buf: DeviceBuffer<u8>,
    midstate_buf_rw: DeviceBuffer<Sha256Ctx>,
    midstate_buf_const: DeviceBuffer<Sha256Ctx>,
    result_local_id_buf: DeviceBuffer<u32>,
    result_hash_buf: DeviceBuffer<u8>,
}

impl GpuMiner {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        rust_cuda::init::init_cuda()?;
        let device = CudaDevice::get_device(0)?;
        let context = device.create_context()?;
        let device_name = device.get_name()?;
        let stream = context.create_stream()?;

        crate::events::publish_event(&format!("[GPU] Using CUDA device: {}", device_name));

        let module = context.load_module_from_slice(KERNEL_PTX)?;

        // --- Create *all* persistent buffers ---
        let final_template_buf = unsafe { DeviceBuffer::uninit(&context, 64)? };
        let n_mod_64_buf = unsafe { DeviceBuffer::uninit(&context, 1)? };
        let base_line_buf = unsafe { DeviceBuffer::uninit(&context, 512)? }; // Max 512 byte base_line
        let midstate_buf_rw = unsafe { DeviceBuffer::uninit(&context, 1)? };
        let midstate_buf_const = unsafe { DeviceBuffer::uninit(&context, 1)? };
        let result_local_id_buf = unsafe { DeviceBuffer::uninit(&context, 1)? };
        let result_hash_buf = unsafe { DeviceBuffer::uninit(&context, 32)? };

        Ok(GpuMiner {
            _device: device,
            _context: context,
            stream,
            module,
            device_name,
            final_template_buf,
            n_mod_64_buf,
            base_line_buf,
            midstate_buf_rw,
            midstate_buf_const,
            result_local_id_buf,
            result_hash_buf,
        })
    }

    pub fn find_seed_gpu(
        &mut self,
        base_line: &str,
        difficulty: usize,
        stop_signal: Arc<AtomicBool>,
    ) -> Result<Option<(String, String)>, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let mut start_nonce: u64 = rand::random();
        let base_len = base_line.len();

        if base_len > self.base_line_buf.len() {
            return Err("base_line is too long for the pre-allocated buffer".into());
        }

        // --- Upload base_line ---
        self.stream
            .memcpy_htod_async(&mut self.base_line_buf, base_line.as_bytes())?;

        // --- Run Setup Kernel *ONCE* ---
        crate::events::publish_event("[GPU] Calculating midstate and final block template...");
        let setup_kernel = self
            .module
            .get_function(&CString::new("calculate_setup_kernel")?)?;

        unsafe {
            launch!(
                // Kernel, Grid (1 block), Block (1 thread), 0 shared mem, stream
                &setup_kernel<<<1, 1, 0, &self.stream>>>(
                    self.base_line_buf.as_device_ptr(),
                    base_len as i32,
                    self.midstate_buf_rw.as_device_ptr(),
                    self.final_template_buf.as_device_ptr(),
                    self.n_mod_64_buf.as_device_ptr()
                )
            )?;
        }

        // Copy midstate from RW buffer to __constant__ RO buffer
        // In CUDA, we must use a named symbol for the __constant__ buffer.
        // We'll copy from device RW (midstate_buf_rw) to device Constant (symbol "c_midstate")
        let c_midstate_symbol = self
            .module
            .get_global_symbol(&CString::new("c_midstate")?)?;
        self.stream.memcpy_dtod_async(
            &mut c_midstate_symbol.as_device_ptr(),
            self.midstate_buf_rw.as_device_ptr(),
            1, // 1 copy of Sha256Ctx
        )?;

        // Copy final block template to its __constant__ symbol
        let c_template_symbol = self
            .module
            .get_global_symbol(&CString::new("c_final_block_template")?)?;
        self.stream.memcpy_dtod_async(
            &mut c_template_symbol.as_device_ptr(),
            self.final_template_buf.as_device_ptr(),
            64, // 64 bytes
        )?;

        // --- Read back the nonce offset *ONCE* ---
        let mut n_mod_64_host = [0u32; 1];
        self.stream
            .memcpy_dtoh_async(&mut n_mod_64_host, &self.n_mod_64_buf)?;

        // We *must* block here to get this value before the main loop.
        self.stream.synchronize()?;
        let n_mod_64_val = n_mod_64_host[0];

        crate::events::publish_event(&format!(
            "[GPU] Starting hash search with batch size {GLOBAL_WORK_SIZE} (local size {LOCAL_WORK_SIZE})..."
        ));

        let find_hash_kernel = self
            .module
            .get_function(&CString::new("find_hash_kernel")?)?;

        // --- Main Mining Loop ---
        loop {
            if stop_signal.load(Ordering::Relaxed) {
                crate::events::publish_event("[GPU] Stop signal received. Aborting work.");
                return Ok(None);
            }

            // Reset the "found" buffer
            self.stream
                .memcpy_htod_async(&mut self.result_local_id_buf, &[NO_RESULT])?;

            // --- Launch the main kernel ---
            unsafe {
                launch!(
                    &find_hash_kernel<<<GRID_SIZE, LOCAL_WORK_SIZE, 0, &self.stream>>>(
                        // c_midstate (Input 1) and c_final_block_template (Input 2)
                        // are global __constant__ variables, not kernel args.
                        n_mod_64_val, // Input 3 (by value)
                        start_nonce,
                        difficulty as u32,
                        self.result_local_id_buf.as_device_ptr(),
                        self.result_hash_buf.as_device_ptr()
                    )
                )?;
            }

            // Read back the result ID
            let mut result_local_id = [NO_RESULT; 1];
            self.stream
                .memcpy_dtoh_async(&mut result_local_id, &self.result_local_id_buf)?;

            // We must wait for the copy to finish before checking the value.
            self.stream.synchronize()?;

            // Report hashes for this batch
            TOTAL_HASHES.fetch_add(GLOBAL_WORK_SIZE as u64, Ordering::Relaxed);

            // --- Check if we found a solution ---
            if result_local_id[0] != NO_RESULT {
                if stop_signal
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    // We were first! Read the hash and return.
                    let found_nonce = start_note + (result_local_id[0] as u64);
                    let seed_str = format!("{:x}", found_nonce);

                    let mut hash_bytes = [0u8; 32];
                    // We already synced, but a DtoH copy is needed
                    self.stream
                        .memcpy_dtoh_async(&mut hash_bytes, &self.result_hash_buf)?;
                    self.stream.synchronize()?;

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
