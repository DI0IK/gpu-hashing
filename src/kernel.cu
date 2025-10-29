#include <cuda.h>
#include <stdint.h>

// --- SHA-256 Context Struct ---
// (Must match Rust)
struct sha256_ctx
{
    uint32_t H[8];
    uint8_t M[64];
    uint64_t N;
};

// --- SHA-256 Constants ---
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// --- Byte-to-Hex Lookup Table ---
__constant__ uchar2 hex_lookup[256] = {
    {'0', '0'}, {'0', '1'}, {'0', '2'}, {'0', '3'}, {'0', '4'}, {'0', '5'}, {'0', '6'}, {'0', '7'}, {'0', '8'}, {'0', '9'}, {'0', 'a'}, {'0', 'b'}, {'0', 'c'}, {'0', 'd'}, {'0', 'e'}, {'0', 'f'}, {'1', '0'}, {'1', '1'}, {'1', '2'}, {'1', '3'}, {'1', '4'}, {'1', '5'}, {'1', '6'}, {'1', '7'}, {'1', '8'}, {'1', '9'}, {'1', 'a'}, {'1', 'b'}, {'1', 'c'}, {'1', 'd'}, {'1', 'e'}, {'1', 'f'}, {'2', '0'}, {'2', '1'}, {'2', '2'}, {'2', '3'}, {'2', '4'}, {'2', '5'}, {'2', '6'}, {'2', '7'}, {'2', '8'}, {'2', '9'}, {'2', 'a'}, {'2', 'b'}, {'2', 'c'}, {'2', 'd'}, {'2', 'e'}, {'2', 'f'}, {'3', '0'}, {'3', '1'}, {'3', '2'}, {'3', '3'}, {'3', '4'}, {'3', '5'}, {'3', '6'}, {'3', '7'}, {'3', '8'}, {'3', '9'}, {'3', 'a'}, {'3', 'b'}, {'3', 'c'}, {'3', 'd'}, {'3', 'e'}, {'3', 'f'}, {'4', '0'}, {'4', '1'}, {'4', '2'}, {'4', '3'}, {'4', '4'}, {'4', '5'}, {'4', '6'}, {'4', '7'}, {'4', '8'}, {'4', '9'}, {'4', 'a'}, {'4', 'b'}, {'4', 'c'}, {'4', 'd'}, {'4', 'e'}, {'4', 'f'}, {'5', '0'}, {'5', '1'}, {'5', '2'}, {'5', '3'}, {'5', '4'}, {'5', '5'}, {'5', '6'}, {'5', '7'}, {'5', '8'}, {'5', '9'}, {'5', 'a'}, {'5', 'b'}, {'5', 'c'}, {'5', 'd'}, {'5', 'e'}, {'5', 'f'}, {'6', '0'}, {'6', '1'}, {'6', '2'}, {'6', '3'}, {'6', '4'}, {'6', '5'}, {'6', '6'}, {'6', '7'}, {'6', '8'}, {'6', '9'}, {'6', 'a'}, {'6', 'b'}, {'6', 'c'}, {'6', 'd'}, {'6', 'e'}, {'6', 'f'}, {'7', '0'}, {'7', '1'}, {'7', '2'}, {'7', '3'}, {'7', '4'}, {'7', '5'}, {'7', '6'}, {'7', '7'}, {'7', '8'}, {'7', '9'}, {'7', 'a'}, {'7', 'b'}, {'7', 'c'}, {'7', 'd'}, {'7', 'e'}, {'7', 'f'}, {'8', '0'}, {'8', '1'}, {'8', '2'}, {'8', '3'}, {'8', '4'}, {'8', '5'}, {'8', '6'}, {'8', '7'}, {'8', '8'}, {'8', '9'}, {'8', 'a'}, {'8', 'b'}, {'8', 'c'}, {'8', 'd'}, {'8', 'e'}, {'8', 'f'}, {'9', '0'}, {'9', '1'}, {'9', '2'}, {'9', '3'}, {'9', '4'}, {'9', '5'}, {'9', '6'}, {'9', '7'}, {'9', '8'}, {'9', '9'}, {'9', 'a'}, {'9', 'b'}, {'9', 'c'}, {'9', 'd'}, {'9', 'e'}, {'9', 'f'}, {'a', '0'}, {'a', '1'}, {'a', '2'}, {'a', '3'}, {'a', '4'}, {'a', '5'}, {'a', '6'}, {'a', '7'}, {'a', '8'}, {'a', '9'}, {'a', 'a'}, {'a', 'b'}, {'a', 'c'}, {'a', 'd'}, {'a', 'e'}, {'a', 'f'}, {'b', '0'}, {'b', '1'}, {'b', '2'}, {'b', '3'}, {'b', '4'}, {'b', '5'}, {'b', '6'}, {'b', '7'}, {'b', '8'}, {'b', '9'}, {'b', 'a'}, {'b', 'b'}, {'b', 'c'}, {'b', 'd'}, {'b', 'e'}, {'b', 'f'}, {'c', '0'}, {'c', '1'}, {'c', '2'}, {'c', '3'}, {'c', '4'}, {'c', '5'}, {'c', '6'}, {'c', '7'}, {'c', '8'}, {'c', '9'}, {'c', 'a'}, {'c', 'b'}, {'c', 'c'}, {'c', 'd'}, {'c', 'e'}, {'c', 'f'}, {'d', '0'}, {'d', '1'}, {'d', '2'}, {'d', '3'}, {'d', '4'}, {'d', '5'}, {'d', '6'}, {'d', '7'}, {'d', '8'}, {'d', '9'}, {'d', 'a'}, {'d', 'b'}, {'d', 'c'}, {'d', 'd'}, {'d', 'e'}, {'d', 'f'}, {'e', '0'}, {'e', '1'}, {'e', '2'}, {'e', '3'}, {'e', '4'}, {'e', '5'}, {'e', '6'}, {'e', '7'}, {'e', '8'}, {'e', '9'}, {'e', 'a'}, {'e', 'b'}, {'e', 'c'}, {'e', 'd'}, {'e', 'e'}, {'e', 'f'}, {'f', '0'}, {'f', '1'}, {'f', '2'}, {'f', '3'}, {'f', '4'}, {'f', '5'}, {'f', '6'}, {'f', '7'}, {'f', '8'}, {'f', '9'}, {'f', 'a'}, {'f', 'b'}, {'f', 'c'}, {'f', 'd'}, {'f', 'e'}, {'f', 'f'}}

// --- 200IQ OPTIMIZATION: LOP3.LUT ---
// We force the compiler to use the 3-input logic operation
// instruction, which is the heart of SHA-256 performance.

// ROTR(x, n) = bitwise right rotate
// Using __funnelshift_r (shf.r PTX instruction) is the most direct way.
__device__ __forceinline__ uint32_t ROTR(uint32_t x, uint32_t n)
{
    return __funnelshift_r(x, x, n);
}
#define SHR(x, n) ((x) >> (n))

// CH(x, y, z) = (x & y) ^ (~x & z)
// PTX: lop3.b32 d, x, y, z, 0xcc;
// The "truth table" 0xcc represents this exact function.
__device__ __forceinline__ uint32_t CH(uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, 0xcc;" : "=r"(ret) : "r"(x), "r"(y), "r"(z));
    return ret;
}

// MAJ(x, y, z) = (x & y) ^ (x & z) ^ (y & z)
// PTX: lop3.b32 d, x, y, z, 0xe8;
// The "truth table" 0xe8 represents this exact function.
__device__ __forceinline__ uint32_t MAJ(uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(ret) : "r"(x), "r"(y), "r"(z));
    return ret;
}

// Standard SHA-256 macros now use our LOP3-optimized functions
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

// --- SHA-256 Core Functions ---

// Single SHA-256 transform (now blazing fast)
__device__ __forceinline__ void sha256_transform(sha256_ctx *ctx)
{
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

// 1. Prepare message schedule
#pragma unroll
    for (int i = 0, j = 0; i < 16; i++, j += 4)
    {
        W[i] = (ctx->M[j] << 24) | (ctx->M[j + 1] << 16) | (ctx->M[j + 2] << 8) | ctx->M[j + 3];
    }
#pragma unroll
    for (int i = 16; i < 64; i++)
    {
        W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];
    }

    // 2. Initialize working registers
    a = ctx->H[0];
    b = ctx->H[1];
    c = ctx->H[2];
    d = ctx->H[3];
    e = ctx->H[4];
    f = ctx->H[5];
    g = ctx->H[6];
    h = ctx->H[7];

// 3. Main compression loop
#pragma unroll
    for (int i = 0; i < 64; i++)
    {
        uint32_t T1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
        uint32_t T2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // 4. Add to hash value
    ctx->H[0] += a;
    ctx->H[1] += b;
    ctx->H[2] += c;
    ctx->H[3] += d;
    ctx->H[4] += e;
    ctx->H[5] += f;
    ctx->H[6] += g;
    ctx->H[7] += h;
}

// (sha256_init and sha256_update_global are ported from your OpenCL kernel)
__device__ __forceinline__ void sha256_init(sha256_ctx *ctx)
{
    ctx->H[0] = 0x6a09e667;
    ctx->H[1] = 0xbb67ae85;
    ctx->H[2] = 0x3c6ef372;
    ctx->H[3] = 0xa54ff53a;
    ctx->H[4] = 0x510e527f;
    ctx->H[5] = 0x9b05688c;
    ctx->H[6] = 0x1f83d9ab;
    ctx->H[7] = 0x5be0cd19;
    ctx->N = 0;
}

__device__ __forceinline__ void sha256_update_global(sha256_ctx *ctx, const uint8_t *data, int len)
{
    int pos = 0;
    while (pos < len)
    {
        int n_mod_64 = (int)(ctx->N % 64);
        int copy_len = min(64 - n_mod_64, len - pos);

        for (int i = 0; i < copy_len; i++)
        {
            ctx->M[n_mod_64 + i] = data[pos + i];
        }

        ctx->N += copy_len;
        pos += copy_len;

        if ((ctx->N % 64) == 0)
        {
            sha256_transform(ctx);
        }
    }
}

// (sha256_final_and_check_fast and write_hash_from_H are ported)
__device__ __forceinline__ bool sha256_final_and_check_fast(sha256_ctx *ctx, uint32_t difficulty)
{
    sha256_transform(ctx);

    // OPTIMIZATION: Unrolled Difficulty Check
    uint32_t bits = 0;
    uint32_t h_val, leading_zeros;

    h_val = ctx->H[0];
    leading_zeros = __clz(h_val);
    bits += leading_zeros;
    if (leading_zeros < 32)
        return (bits >= difficulty);
    if (bits >= difficulty)
        return true;
    h_val = ctx->H[1];
    leading_zeros = __clz(h_val);
    bits += leading_zeros;
    if (leading_zeros < 32)
        return (bits >= difficulty);
    if (bits >= difficulty)
        return true;
    h_val = ctx->H[2];
    leading_zeros = __clz(h_val);
    bits += leading_zeros;
    if (leading_zeros < 32)
        return (bits >= difficulty);
    if (bits >= difficulty)
        return true;
    h_val = ctx->H[3];
    leading_zeros = __clz(h_val);
    bits += leading_zeros;
    if (leading_zeros < 32)
        return (bits >= difficulty);
    if (bits >= difficulty)
        return true;
    h_val = ctx->H[4];
    leading_zeros = __clz(h_val);
    bits += leading_zeros;
    if (leading_zeros < 32)
        return (bits >= difficulty);
    if (bits >= difficulty)
        return true;
    h_val = ctx->H[5];
    leading_zeros = __clz(h_val);
    bits += leading_zeros;
    if (leading_zeros < 32)
        return (bits >= difficulty);
    if (bits >= difficulty)
        return true;
    h_val = ctx->H[6];
    leading_zeros = __clz(h_val);
    bits += leading_zeros;
    if (leading_zeros < 32)
        return (bits >= difficulty);
    if (bits >= difficulty)
        return true;
    h_val = ctx->H[7];
    leading_zeros = __clz(h_val);
    bits += leading_zeros;

    return (bits >= difficulty);
}

__device__ __forceinline__ void write_hash_from_H(uint8_t *hash_out, sha256_ctx *ctx)
{
    for (int i = 0; i < 8; i++)
    {
        hash_out[i * 4 + 0] = (uint8_t)(ctx->H[i] >> 24);
        hash_out[i * 4 + 1] = (uint8_t)(ctx->H[i] >> 16);
        hash_out[i * 4 + 2] = (uint8_t)(ctx->H[i] >> 8);
        hash_out[i * 4 + 3] = (uint8_t)(ctx->H[i]);
    }
}

/*
 * == KERNEL 1: Calculate Midstate AND Final Block Template (Run ONCE) ==
 */
extern "C" __global__ void calculate_setup_kernel(
    const uint8_t *base_line,
    int base_len,
    sha256_ctx *midstate_out,          // Output 1
    uint8_t *final_block_template_out, // Output 2
    uint32_t *n_mod_64_nonce_start_out // Output 3
)
{
    // This kernel is run with 1 thread, it's not performance critical.
    // The logic is identical to your OpenCL kernel.

    // 1. Calculate midstate
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update_global(&ctx, base_line, base_len);
    *midstate_out = ctx;

    // 2. Prepare final block template
    uint64_t final_N = ctx.N + 16;
    int n_mod_64_final = (int)(final_N % 64);
    uint32_t n_mod_64_start = (uint32_t)(ctx.N % 64);

    uint8_t M_template[64];

    // 2a. Copy partial data
    for (int i = 0; i < n_mod_64_start; i++)
    {
        M_template[i] = ctx.M[i];
    }
    // 2b. Zero-init rest
    for (int i = n_mod_64_start; i < 64; i++)
    {
        M_template[i] = 0;
    }
    // 2c. Add padding byte
    M_template[n_mod_64_final] = 0x80;

    // 2d/e. Write 64-bit length
    uint64_t n_bits = final_N * 8;
    M_template[56] = (uint8_t)(n_bits >> 56);
    M_template[57] = (uint8_t)(n_bits >> 48);
    M_template[58] = (uint8_t)(n_bits >> 40);
    M_template[59] = (uint8_t)(n_bits >> 32);
    M_template[60] = (uint8_t)(n_bits >> 24);
    M_template[61] = (uint8_t)(n_bits >> 16);
    M_template[62] = (uint8_t)(n_bits >> 8);
    M_template[63] = (uint8_t)(n_bits);

    // 3. Write template to global memory
    for (int j = 0; j < 64; j++)
    {
        final_block_template_out[j] = M_template[j];
    }

    // 4. Write nonce insertion offset
    *n_mod_64_nonce_start_out = n_mod_64_start;
}

/*
 * == KERNEL 2: Find Hash (Run 10M+ times) ==
 *
 * (HYPER-OPTIMIZED for NVIDIA)
 */

// --- Global __constant__ inputs for the main kernel ---
// This is the fastest memory for read-only, kernel-wide data.
// We memcpy to these symbols from the host.
__device__ __constant__ sha256_ctx c_midstate;
__device__ __constant__ uint8_t c_final_block_template[64];

// __launch_bounds__(256, 4)
// - 256: Block size (must match LOCAL_WORK_SIZE)
// - 4:   Min blocks per SM. This hints the compiler to
//        limit register usage to maximize occupancy.
extern "C" __global__ __launch_bounds__(256, 4) void find_hash_kernel(
    // Inputs 1 & 2 are now in __constant__ memory
    uint32_t n_mod_64_nonce_start,
    uint64_t start_nonce,
    uint32_t difficulty,
    uint32_t *result_local_id,
    uint8_t *result_hash)
{
    uint64_t global_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + global_id;

    // Removed the early-exit global read, as it's a perf bottleneck.

    // 1. Load the pre-calculated H-state from __constant__ memory
    sha256_ctx ctx;
    ctx.H[0] = c_midstate.H[0];
    ctx.H[1] = c_midstate.H[1];
    ctx.H[2] = c_midstate.H[2];
    ctx.H[3] = c_midstate.H[3];
    ctx.H[4] = c_midstate.H[4];
    ctx.H[5] = c_midstate.H[5];
    ctx.H[6] = c_midstate.H[6];
    ctx.H[7] = c_midstate.H[7];
    // ctx.N is not needed

    // 2. Load the pre-padded final M-block (fast 64-byte vector copy)
    // Use ulonglong4 (4 x 16 bytes) for a 64-byte copy.
    *((ulonglong4 *)ctx.M) = *((const ulonglong4 *)c_final_block_template);

    // 3. Write 16-byte hex nonce DIRECTLY into ctx.M
    uchar2 *hex_ptr = (uchar2 *)&ctx.M[n_mod_64_nonce_start];

    hex_ptr[0] = hex_lookup[(nonce >> 56) & 0xFF];
    hex_ptr[1] = hex_lookup[(nonce >> 48) & 0xFF];
    hex_ptr[2] = hex_lookup[(nonce >> 40) & 0xFF];
    hex_ptr[3] = hex_lookup[(nonce >> 32) & 0xFF];
    hex_ptr[4] = hex_lookup[(nonce >> 24) & 0xFF];
    hex_ptr[5] = hex_lookup[(nonce >> 16) & 0xFF];
    hex_ptr[6] = hex_lookup[(nonce >> 8) & 0xFF];
    hex_ptr[7] = hex_lookup[nonce & 0xFF];

    // 4. Finalize hash and check difficulty (now extremely fast)
    if (sha256_final_and_check_fast(&ctx, difficulty))
    {
        // We found a winner!
        // atomicCAS (Compare And Swap)
        if (atomicCAS(result_local_id, (uint32_t)(-1), (uint32_t)global_id) == (uint32_t)(-1))
        {
            // We were first! Write our hash.
            write_hash_from_H(result_hash, &ctx);
        }
    }
}