//! GPU Miner (OpenCL) Module
//!
//! This file contains all the logic for finding a hash seed using the GPU.
//!
//! --- How It Works ---
//!
//! 1.  `GpuMiner::new()`: Called once at startup. It finds the first available
//!     OpenCL GPU, creates a "Context", and compiles the OpenCL C "Kernel"
//!     (the `OPENCL_KERNEL_SRC` string).
//!
//! 2.  `find_seed_gpu()`: Called in the main loop.
//!     a.  It takes the "base line" (e.g., "parent_hash my_name ") and the
//!         `difficulty`.
//!     b.  It creates buffers (memory) on the GPU to hold the input
//!         (`base_line`) and the output (the `found_seed` and `found_hash`).
//!     c.  It launches the OpenCL kernel in massive batches (e.g., 1 million
//!         "threads" at a time).
//!     d.  Each GPU "thread" gets a unique ID (`global_id`) and calculates a
//!         nonce (seed) to test: `nonce = start_nonce + global_id`.
//!     e.  The kernel hashes `base_line + nonce_as_hex_string` entirely on the
//!         GPU.
//!     f.  If a thread finds a valid hash, it *atomically* writes its `nonce`
//!         and the `hash` to the output buffers and stops all other threads.
//!     g.  The Rust code (host) checks the output buffer. If a `nonce` was
//     found, it's returned.
//!     h.  If no `nonce` was found in the batch, it increases `start_nonce`
//     and launches the *next* batch.
//!

use crate::TOTAL_HASHES;
use ocl::{
    builders::ProgramBuilder,
    enums::{DeviceInfo, DeviceInfoResult}, // --- ADDED ---
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
use std::sync::atomic::Ordering;
use std::time::Instant;

/// The OpenCL C kernel. This is a small program compiled at runtime
/// and sent to the GPU.
const OPENCL_KERNEL_SRC: &str = r#"
/*
 * OpenCL SHA-256 Kernel
 *
 * This code runs on the GPU. It includes a full SHA-256 implementation
 * and a helper to convert a ulong (u64) nonce into a hex string,
 * which is required by the game's hash format.
 */

// --- SHA-256 Implementation (for OpenCL) ---
// (This is a standard, compact SHA-256 implementation adapted for OpenCL)

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHR(x, n) ((x) >> (n))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

typedef struct {
    uchar M[64];
    uint H[8];
    ulong N;
} sha256_ctx;

constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// --- End of SHA-256 ---


// Helper to convert a ulong nonce to a 1-16 char hex string
// This is the tricky part, as sprintf is not available.
// Returns the length of the hex string.
inline int ulong_to_hex(ulong n, char* out) {
    // --- FIXED: Use 'const' for local array, not 'constant' ---
    const char hex_chars[] = "0123456789abcdef";
    char buf[16];
    int i = 0;

    if (n == 0) {
        out[0] = '0';
        return 1;
    }

    while (n > 0) {
        buf[i++] = hex_chars[n & 0xF];
        n >>= 4;
    }

    // Reverse the buffer into out
    for (int j = 0; j < i; j++) {
        out[j] = buf[i - 1 - j];
    }
    return i;
}


// SHA-256 transform function
void sha256_transform(sha256_ctx *ctx) {
    uint W[64];
    uint a, b, c, d, e, f, g, h;
    
    #pragma unroll
    for (int i = 0, j = 0; i < 16; i++, j += 4) {
        W[i] = (ctx->M[j] << 24) | (ctx->M[j + 1] << 16) | (ctx->M[j + 2] << 8) | ctx->M[j + 3];
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];
    }

    a = ctx->H[0]; b = ctx->H[1]; c = ctx->H[2]; d = ctx->H[3];
    e = ctx->H[4]; f = ctx->H[5]; g = ctx->H[6]; h = ctx->H[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint T1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
        uint T2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    ctx->H[0] += a; ctx->H[1] += b; ctx->H[2] += c; ctx->H[3] += d;
    ctx->H[4] += e; ctx->H[5] += f; ctx->H[6] += g; ctx->H[7] += h;
}

// Main SHA-256 function
void sha256(const uchar* data, int len, uchar* hash_out) {
    sha256_ctx ctx;
    
    // Init H
    ctx.H[0] = 0x6a09e667; ctx.H[1] = 0xbb67ae85; ctx.H[2] = 0x3c6ef372; ctx.H[3] = 0xa54ff53a;
    ctx.H[4] = 0x510e527f; ctx.H[5] = 0x9b05688c; ctx.H[6] = 0x1f83d9ab; ctx.H[7] = 0x5be0cd19;
    ctx.N = 0;

    int pos = 0;
    while(pos < len) {
        int copy_len = min(64 - (int)(ctx.N % 64), len - pos);
        
        // --- FIXED: Replaced memcpy with a loop ---
        for(int i=0; i < copy_len; i++) {
            ctx.M[(ctx.N % 64) + i] = data[pos + i];
        }

        ctx.N += copy_len;
        pos += copy_len;

        if ((ctx.N % 64) == 0) {
            sha256_transform(&ctx);
        }
    }

    // Padding
    int n_mod_64 = (int)(ctx.N % 64);
    ctx.M[n_mod_64++] = 0x80;
    
    if (n_mod_64 > 56) {
        // --- FIXED: Replaced memset with a loop ---
        for(int i=n_mod_64; i < 64; i++) {
            ctx.M[i] = 0;
        }
        sha256_transform(&ctx);
        n_mod_64 = 0;
    }
    
    // --- FIXED: Replaced memset with a loop ---
    for(int i=n_mod_64; i < 56; i++) {
        ctx.M[i] = 0;
    }
    
    // Append length
    ulong n_bits = ctx.N * 8;
    ctx.M[56] = (uchar)(n_bits >> 56);
    ctx.M[57] = (uchar)(n_bits >> 48);
    ctx.M[58] = (uchar)(n_bits >> 40);
    ctx.M[59] = (uchar)(n_bits >> 32);
    ctx.M[60] = (uchar)(n_bits >> 24);
    ctx.M[61] = (uchar)(n_bits >> 16);
    ctx.M[62] = (uchar)(n_bits >> 8);
    ctx.M[63] = (uchar)(n_bits);

    sha256_transform(&ctx);
    
    // Output hash
    for(int i = 0; i < 8; i++) {
        hash_out[i*4 + 0] = (uchar)(ctx.H[i] >> 24);
        hash_out[i*4 + 1] = (uchar)(ctx.H[i] >> 16);
        hash_out[i*4 + 2] = (uchar)(ctx.H[i] >> 8);
        hash_out[i*4 + 3] = (uchar)(ctx.H[i]);
    }
}


// Helper to check for N leading zero bits
// --- FIXED: Changed address space from 'global const' to 'const' (defaults to private) ---
inline int count_zero_bits_kernel(const uchar* hash, int difficulty) {
    int bits = 0;
    for (int i = 0; i < 32; i++) {
        uchar byte = hash[i];
        if (byte == 0) {
            bits += 8;
        } else {
            // Count leading zeros in the first non-zero byte
            uchar b = byte;
            while ((b & 0x80) == 0) {
                bits++;
                b <<= 1;
            }
            break;
        }
        if (bits >= difficulty) break;
    }
    return bits;
}


/*
 * == The Main Kernel ==
 *
 * This is launched in parallel by millions of "threads" on the GPU.
 */
__kernel void find_hash_kernel(
    // --- FIXED: Changed 'char' to 'uchar' to match Rust's u8 buffer ---
    __global const uchar* base_line, // "parent_hash name "
    int base_len,                   // Length of base_line
    ulong start_nonce,              // The starting nonce for this *batch*
    uint difficulty,                // Target difficulty
    // --- FIXED: Use ulong for nonce, but uint for atomic result ---
    __global uint* result_local_id, // Output: The winning local_id (if found)
    __global uchar* result_hash     // Output: The winning hash (if found)
) {
    // Get the unique ID for this thread
    ulong global_id = get_global_id(0);
    ulong nonce = start_nonce + global_id;

    // --- FIXED: Read from uint buffer, not ulong. Use simple dereference ---
    if (*result_local_id != (uint)(-1)) {
        return;
    }

    // 1. Construct the input string: "parent_hash name nonce_hex"
    uchar line[256]; // Buffer for the full line
    
    // --- FIXED: Replaced memcpy with a loop ---
    for(int i=0; i < base_len; i++) {
        line[i] = base_line[i];
    }

    // Convert the nonce (ulong) to a hex string
    char nonce_hex[17]; // 16 hex chars + null
    int nonce_len = ulong_to_hex(nonce, nonce_hex);

    // --- FIXED: Replaced memcpy with a loop ---
    for(int i=0; i < nonce_len; i++) {
        line[base_len + i] = nonce_hex[i];
    }
    
    int total_len = base_len + nonce_len;
    // line[total_len] = 0; // Not needed for sha256 function

    // 2. Hash the constructed line
    uchar hash_output[32];
    sha256(line, total_len, hash_output);

    // 3. Check if the hash meets the difficulty
    int bits = count_zero_bits_kernel(hash_output, difficulty);

    if (bits >= difficulty) {
        // We found a winner!
        // --- FIXED: Use uint for atomic operation. Store global_id, not full nonce ---
        if (atomic_cmpxchg(result_local_id, (uint)(-1), (uint)global_id) == (uint)(-1)) {
            // We were first! Write our hash to the output buffer.
            for(int i=0; i < 32; i++) {
                result_hash[i] = hash_output[i];
            }
        }
    }
}
"#;

// How many hashes to run on the GPU in a single batch.
// 10 million is a good starting point. Tune this for your GPU.
const GLOBAL_WORK_SIZE: usize = 10_485_760;
// --- FIXED: Use u32::MAX for the 32-bit result buffer ---
const NO_RESULT: u32 = u32::MAX;

/// The GpuMiner struct holds the persistent OpenCL objects.
pub struct GpuMiner {
    context: Context,
    queue: Queue,
    kernel: Kernel,
    // --- FIXED: Removed unused device field ---
    // device: Device,
}

impl GpuMiner {
    /// Creates a new GpuMiner, finds the device, and compiles the kernel.
    pub fn new() -> OclResult<Self> {
        // 1. Find a GPU platform and device
        let platform = Platform::default();
        let device = Device::list_all(platform)?
            .into_iter()
            // --- FIXED: Use `if let` to pattern match the Result ---
            .find(|d| {
                if let Ok(DeviceInfoResult::Type(device_type)) = d.info(DeviceInfo::Type) {
                    device_type == DeviceType::GPU
                } else {
                    false
                }
            })
            .ok_or("No GPU device found")?;

        println!("[GPU] Using OpenCL device: {}", device.name()?);

        // 2. Create OpenCL context and command queue
        // --- FIXED: Removed the ambiguous `.into()` call ---
        let context = Context::builder().devices(device).build()?;
        let queue = Queue::new(&context, device, None)?;

        // 3. Compile the OpenCL kernel
        let program = ProgramBuilder::new()
            .devices(device)
            .src(OPENCL_KERNEL_SRC)
            .build(&context)?;

        // --- FIXED: Prime the kernel builder with all 6 argument types ---
        let kernel = Kernel::builder()
            .program(&program)
            .name("find_hash_kernel")
            .queue(queue.clone())
            .arg_named("base_line", None::<&Buffer<u8>>) // Arg 0
            .arg_named("base_len", 0i32) // Arg 1
            .arg_named("start_nonce", 0u64) // Arg 2
            .arg_named("difficulty", 0u32) // Arg 3
            .arg_named("result_local_id", None::<&Buffer<u32>>) // Arg 4
            .arg_named("result_hash", None::<&Buffer<u8>>) // Arg 5
            .build()?;

        Ok(GpuMiner {
            context,
            queue,
            kernel,
            // --- FIXED: Removed unused device field ---
            // device,
        })
    }

    /// Finds a seed using the GPU.
    pub fn find_seed_gpu(
        &mut self,
        base_line: &str,
        difficulty: usize,
    ) -> OclResult<(String, String)> {
        let start_time = Instant::now();
        let mut start_nonce: u64 = 0;
        let base_len = base_line.len();

        // --- Create GPU Buffers ---

        // 1. Input: The base string ("parent name ")
        let base_line_buf = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR)
            .len(base_len)
            .copy_host_slice(base_line.as_bytes())
            .build()?;

        // 2. Output: The winning nonce.
        // --- FIXED: Changed to Buffer<u32> for the local_id ---
        let result_local_id_buf: Buffer<u32> = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::WRITE_ONLY | MemFlags::COPY_HOST_PTR)
            .len(1) // one u32
            .copy_host_slice(&[NO_RESULT])
            .build()?;

        // 3. Output: The winning hash (32 bytes)
        let result_hash_buf: Buffer<u8> = Buffer::builder()
            .context(&self.context)
            .flags(MemFlags::WRITE_ONLY)
            .len(32) // 32 u8s
            .build()?;

        println!("[GPU] Starting hash search with batch size {GLOBAL_WORK_SIZE}...");

        // --- Main Mining Loop ---
        loop {
            // Set kernel arguments
            self.kernel.set_arg(0, &base_line_buf)?;
            self.kernel.set_arg(1, base_len as i32)?;
            self.kernel.set_arg(2, start_nonce)?;
            self.kernel.set_arg(3, difficulty as u32)?;
            // --- FIXED: Pass the u32 buffer ---
            self.kernel.set_arg(4, &result_local_id_buf)?;
            self.kernel.set_arg(5, &result_hash_buf)?;

            // Launch the kernel
            unsafe {
                self.kernel
                    .cmd()
                    .queue(&self.queue)
                    .global_work_size(GLOBAL_WORK_SIZE)
                    .enq()?;
            }

            // Read back the result nonce
            // --- FIXED: Read into a u32 buffer ---
            let mut result_local_id = [NO_RESULT; 1];
            result_local_id_buf
                .read(&mut result_local_id[..])
                .queue(&self.queue)
                .enq()?;

            // Report hashes for this batch
            TOTAL_HASHES.fetch_add(GLOBAL_WORK_SIZE as u64, Ordering::Relaxed);

            // --- Check if we found a solution ---
            if result_local_id[0] != NO_RESULT {
                // --- FIXED: Reconstruct the full 64-bit nonce ---
                let found_nonce = start_nonce + (result_local_id[0] as u64);
                let seed_str = format!("{:x}", found_nonce);

                // Read the hash string back
                let mut hash_bytes = [0u8; 32];
                result_hash_buf
                    .read(&mut hash_bytes[..])
                    .queue(&self.queue)
                    .enq()?;

                let hash_hex = hex::encode(hash_bytes);
                let duration = start_time.elapsed();
                let hashes_done = start_nonce + GLOBAL_WORK_SIZE as u64;
                let hash_rate = (hashes_done as f64) / duration.as_secs_f64();

                println!("\n+++ [GPU] Found valid seed! +++");
                println!("    Nonce: {}", found_nonce);
                println!("    Seed:  {}", seed_str);
                println!("    Hash:  {} (Bits: ?)", hash_hex);
                println!(
                    "    Perf:  {} hashes in {:?} ({:.2} MH/s)",
                    hashes_done,
                    duration,
                    hash_rate / 1_000_000.0
                );

                // Return the (seed, hash_hex) tuple
                return Ok((seed_str, hash_hex));
            }

            // No solution found in this batch, prepare for the next
            start_nonce += GLOBAL_WORK_SIZE as u64;
        }
    }
}
