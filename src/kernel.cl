/*
 * OpenCL SHA-256 Kernel (Optimized for Midstate & Hex-String Nonce)
 *
 * (Version 3: Heavily Optimized for fixed-pattern workload)
 *
 * This kernel is for protocols that hash:
 * <64_byte_parent> + ' ' + <name_1-16byte> + ' ' + <16byte_hex_nonce>
 *
 * Optimizations Applied:
 * 1. find_hash_kernel:
 * - Inlines the ulong_to_hex_fixed16 conversion.
 * - Calls a new, specialized `sha256_update_local_16_fast`.
 * - Calls a new, specialized `sha256_final_and_check_fast`.
 *
 * 2. sha256_update_local_16_fast:
 * - Removed all boundary-check logic (which is dead code for this workload).
 * - Performs a simple, unrolled 16-byte copy.
 *
 * 3. sha256_final_and_check_fast:
 * - Removed the `if (n_mod_64 > 56)` branch (dead code for this workload).
 * - Replaced the byte-wise zero-padding loop with a fast, word-aligned
 * (uint-based) "memset".
 * - Fully unrolled the final clz() difficulty check.
 */

// --- SHA-256 Constants and Macros ---
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHR(x, n) ((x) >> (n))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

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

// --- SHA-256 Context Struct ---
typedef struct {
    uint H[8];
    uchar M[64];
    ulong N;
} sha256_ctx;


// --- SHA-256 Core Functions ---

// Single SHA-256 transform (Unchanged)
inline void sha256_transform(sha256_ctx *ctx) {
    uint W[64];
    
    // Portable manual byte-swapping
    #pragma unroll
    for (int i = 0, j = 0; i < 16; i++, j += 4) {
        W[i] = (ctx->M[j] << 24) | (ctx->M[j + 1] << 16) | (ctx->M[j + 2] << 8) | ctx->M[j + 3];
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];
    }

    uint8 H_vec = (uint8)(ctx->H[0], ctx->H[1], ctx->H[2], ctx->H[3],
                         ctx->H[4], ctx->H[5], ctx->H[6], ctx->H[7]);

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint T1 = H_vec.s7 + EP1(H_vec.s4) + CH(H_vec.s4, H_vec.s5, H_vec.s6) + K[i] + W[i];
        uint T2 = EP0(H_vec.s0) + MAJ(H_vec.s0, H_vec.s1, H_vec.s2);
        
        H_vec = (uint8)(T1 + T2, H_vec.s0, H_vec.s1, H_vec.s2,
                        H_vec.s3 + T1, H_vec.s4, H_vec.s5, H_vec.s6);
    }

    uint8 H_in = (uint8)(ctx->H[0], ctx->H[1], ctx->H[2], ctx->H[3],
                         ctx->H[4], ctx->H[5], ctx->H[6], ctx->H[7]);
    H_vec += H_in;

    ctx->H[0] = H_vec.s0; ctx->H[1] = H_vec.s1; ctx->H[2] = H_vec.s2; ctx->H[3] = H_vec.s3;
    ctx->H[4] = H_vec.s4; ctx->H[5] = H_vec.s5; ctx->H[6] = H_vec.s6; ctx->H[7] = H_vec.s7;
}

// Initialize a new context (Unchanged)
inline void sha256_init(sha256_ctx *ctx) {
    ctx->H[0] = 0x6a09e667; ctx->H[1] = 0xbb67ae85; ctx->H[2] = 0x3c6ef372; ctx->H[3] = 0xa54ff53a;
    ctx->H[4] = 0x510e527f; ctx->H[5] = 0x9b05688c; ctx->H[6] = 0x1f83d9ab; ctx->H[7] = 0x5be0cd19;
    ctx->N = 0;
}

// Update context with GLOBAL data (for midstate)
inline void sha256_update_global(sha256_ctx *ctx, __global const uchar* data, int len) {
    int pos = 0;
    while(pos < len) {
        int n_mod_64 = (int)(ctx->N % 64);
        int copy_len = min(64 - n_mod_64, len - pos);
        
        for(int i=0; i < copy_len; i++) {
            ctx->M[n_mod_64 + i] = data[pos + i];
        }

        ctx->N += copy_len;
        pos += copy_len;

        if ((ctx->N % 64) == 0) {
            sha256_transform(ctx);
        }
    }
}

// --- OPTIMIZATION: Specialized 16-byte update for nonce ---
// This function assumes it is *only* for the 16-byte hex nonce
// and that (N % 64) + 16 < 64, which we proved is true.
inline void sha256_update_local_16_fast(sha256_ctx *ctx, const uchar* data) {
    int n_mod_64 = (int)(ctx->N % 64);

    // Unrolled 16-byte copy
    ctx->M[n_mod_64 + 0]  = data[0];
    ctx->M[n_mod_64 + 1]  = data[1];
    ctx->M[n_mod_64 + 2]  = data[2];
    ctx->M[n_mod_64 + 3]  = data[3];
    ctx->M[n_mod_64 + 4]  = data[4];
    ctx->M[n_mod_64 + 5]  = data[5];
    ctx->M[n_mod_64 + 6]  = data[6];
    ctx->M[n_mod_64 + 7]  = data[7];
    ctx->M[n_mod_64 + 8]  = data[8];
    ctx->M[n_mod_64 + 9]  = data[9];
    ctx->M[n_mod_64 + 10] = data[10];
    ctx->M[n_mod_64 + 11] = data[11];
    ctx->M[n_mod_64 + 12] = data[12];
    ctx->M[n_mod_64 + 13] = data[13];
    ctx->M[n_mod_64 + 14] = data[14];
    ctx->M[n_mod_64 + 15] = data[15];

    ctx->N += 16;
}

// --- OPTIMIZATION: Specialized finalization ---
// This function removes the `if (n_mod_64 > 56)` branch (dead code)
// and uses a word-aligned "fast memset" for zero-padding.
inline bool sha256_final_and_check_fast(sha256_ctx *ctx, uint difficulty) {
    // 1. Padding
    int n_mod_64 = (int)(ctx->N % 64);
    ctx->M[n_mod_64++] = 0x80;
    
    // NOTE: The `if (n_mod_64 > 56)` block is REMOVED as it's
    // provably dead code for this specific workload.
    
    // 2. OPTIMIZATION: Fast zero-pad using uint
    
    // 2a. Handle unaligned start (max 3 bytes)
    while(n_mod_64 < 56 && (n_mod_64 % 4 != 0)) {
        ctx->M[n_mod_64++] = 0;
    }

    // 2b. Write aligned 32-bit (uint) chunks
    // We are now at n_mod_64, which is a multiple of 4.
    uint* M_u = (uint*)ctx->M;
    int i_u = n_mod_64 / 4;
    
    #pragma unroll
    while(i_u < 14) { // 14 * 4 = 56 bytes
        M_u[i_u++] = 0;
    }
    
    // 3. Write 64-bit length
    ulong n_bits = ctx->N * 8;
    ctx->M[56] = (uchar)(n_bits >> 56);
    ctx->M[57] = (uchar)(n_bits >> 48);
    ctx->M[58] = (uchar)(n_bits >> 40);
    ctx->M[59] = (uchar)(n_bits >> 32);
    ctx->M[60] = (uchar)(n_bits >> 24);
    ctx->M[61] = (uchar)(n_bits >> 16);
    ctx->M[62] = (uchar)(n_bits >> 8);
    ctx->M[63] = (uchar)(n_bits);

    // 4. Run the final transform
    sha256_transform(ctx);
    
    // 5. OPTIMIZATION: Unrolled Difficulty Check
    uint bits = 0;
    uint h_val, leading_zeros;

    h_val = ctx->H[0]; leading_zeros = clz(h_val); bits += leading_zeros; if (leading_zeros < 32) return (bits >= difficulty); if (bits >= difficulty) return true;
    h_val = ctx->H[1]; leading_zeros = clz(h_val); bits += leading_zeros; if (leading_zeros < 32) return (bits >= difficulty); if (bits >= difficulty) return true;
    h_val = ctx->H[2]; leading_zeros = clz(h_val); bits += leading_zeros; if (leading_zeros < 32) return (bits >= difficulty); if (bits >= difficulty) return true;
    h_val = ctx->H[3]; leading_zeros = clz(h_val); bits += leading_zeros; if (leading_zeros < 32) return (bits >= difficulty); if (bits >= difficulty) return true;
    h_val = ctx->H[4]; leading_zeros = clz(h_val); bits += leading_zeros; if (leading_zeros < 32) return (bits >= difficulty); if (bits >= difficulty) return true;
    h_val = ctx->H[5]; leading_zeros = clz(h_val); bits += leading_zeros; if (leading_zeros < 32) return (bits >= difficulty); if (bits >= difficulty) return true;
    h_val = ctx->H[6]; leading_zeros = clz(h_val); bits += leading_zeros; if (leading_zeros < 32) return (bits >= difficulty); if (bits >= difficulty) return true;
    h_val = ctx->H[7]; leading_zeros = clz(h_val); bits += leading_zeros;
    
    return (bits >= difficulty);
}

// Helper to write the final hash to global memory (Unchanged)
inline void write_hash_from_H(__global uchar* hash_out, sha256_ctx *ctx) {
    for(int i = 0; i < 8; i++) {
        hash_out[i*4 + 0] = (uchar)(ctx->H[i] >> 24);
        hash_out[i*4 + 1] = (uchar)(ctx->H[i] >> 16);
        hash_out[i*4 + 2] = (uchar)(ctx->H[i] >> 8);
        hash_out[i*4 + 3] = (uchar)(ctx->H[i]);
    }
}

/*
 * == KERNEL 1: Calculate Midstate (Run ONCE) ==
 */
__kernel void calculate_midstate_kernel(
    __global const uchar* base_line,
    int base_len,
    __global sha256_ctx* midstate_out
) {
    sha256_ctx ctx;
    sha256_init(&ctx);
    // Use the global update function
    sha256_update_global(&ctx, base_line, base_len);
    *midstate_out = ctx;
}


/*
 * == KERNEL 2: Find Hash (Run 10M+ times) ==
 *
 * (HEAVILY OPTIMIZED)
 */
__kernel void find_hash_kernel(
    __constant const sha256_ctx* midstate, // <-- Use __constant
    ulong start_nonce,
    uint difficulty,
    __global uint* result_local_id,
    __global uchar* result_hash
) {
    ulong global_id = get_global_id(0);
    ulong nonce = start_nonce + global_id;

    if (*result_local_id != (uint)(-1)) {
        return;
    }

    // 1. Load the pre-calculated midstate (fast read from constant cache)
    sha256_ctx ctx = *midstate;

    // 2. --- OPTIMIZATION: Inlined ulong_to_hex_fixed16 ---
    uchar nonce_hex[16]; // Use uchar directly
    const char hex_chars[] = "0123456789abcdef";
    
    nonce_hex[0]  = hex_chars[(nonce >> 60) & 0xF];
    nonce_hex[1]  = hex_chars[(nonce >> 56) & 0xF];
    nonce_hex[2]  = hex_chars[(nonce >> 52) & 0xF];
    nonce_hex[3]  = hex_chars[(nonce >> 48) & 0xF];
    nonce_hex[4]  = hex_chars[(nonce >> 44) & 0xF];
    nonce_hex[5]  = hex_chars[(nonce >> 40) & 0xF];
    nonce_hex[6]  = hex_chars[(nonce >> 36) & 0xF];
    nonce_hex[7]  = hex_chars[(nonce >> 32) & 0xF];
    nonce_hex[8]  = hex_chars[(nonce >> 28) & 0xF];
    nonce_hex[9]  = hex_chars[(nonce >> 24) & 0xF];
    nonce_hex[10] = hex_chars[(nonce >> 20) & 0xF];
    nonce_hex[11] = hex_chars[(nonce >> 16) & 0xF];
    nonce_hex[12] = hex_chars[(nonce >> 12) & 0xF];
    nonce_hex[13] = hex_chars[(nonce >> 8)  & 0xF];
    nonce_hex[14] = hex_chars[(nonce >> 4)  & 0xF];
    nonce_hex[15] = hex_chars[nonce & 0xF];
    
    // 3. Update context with the 16-char hex string (fast path)
    sha256_update_local_16_fast(&ctx, (const uchar*)nonce_hex);

    // 4. Finalize hash and check difficulty (fast path)
    if (sha256_final_and_check_fast(&ctx, difficulty)) {
        // We found a winner!
        if (atomic_cmpxchg(result_local_id, (uint)(-1), (uint)global_id) == (uint)(-1)) {
            // We were first! Write our hash to the output buffer.
            write_hash_from_H(result_hash, &ctx);
        }
    }
}
