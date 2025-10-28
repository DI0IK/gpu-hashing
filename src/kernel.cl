/*
 * OpenCL SHA-256 Kernel (Optimized for Midstate & Hex-String Nonce)
 *
 * (Version 2: Reverted bswap32 to manual byte-swap for portability)
 *
 * This kernel is for protocols that require the nonce to be
 * appended as its hex-string representation, not raw bytes.
 *
 * Key Optimizations Applied:
 * 1. find_hash_kernel:
 * - Midstate loaded from `__constant` memory.
 * - Nonce (u64) is converted to a 16-char hex string
 * using a fast, branchless, unrolled function.
 * 2. sha256_final_and_check:
 * - Uses `clz()` (Count Leading Zeros) intrinsic for difficulty check.
 * 3. sha256_transform:
 * - State registers (a-h) vectorized into a `uint8`.
 * - Message loading uses portable, manual byte-swapping.
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


// --- SHA-256 Core Functions (Refactored) ---

// Single SHA-256 transform
inline void sha256_transform(sha256_ctx *ctx) {
    uint W[64];
    
    // --- FIX: Reverted from bswap32 to manual, portable byte-swapping ---
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

// Initialize a new context
inline void sha256_init(sha256_ctx *ctx) {
    ctx->H[0] = 0x6a09e667; ctx->H[1] = 0xbb67ae85; ctx->H[2] = 0x3c6ef372; ctx->H[3] = 0xa54ff53a;
    ctx->H[4] = 0x510e527f; ctx->H[5] = 0x9b05688c; ctx->H[6] = 0x1f83d9ab; ctx->H[7] = 0x5be0cd19;
    ctx->N = 0;
}

// Update context with data (processes full blocks)
inline void sha256_update(sha256_ctx *ctx, __global const uchar* data, int len) {
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

// Update context with local char data (for nonce)
inline void sha256_update_local(sha256_ctx *ctx, const uchar* data, int len) {
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

// Finalize hash: Apply padding and run last transform(s)
inline bool sha256_final_and_check(sha256_ctx *ctx, uint difficulty) {
    // Padding
    int n_mod_64 = (int)(ctx->N % 64);
    ctx->M[n_mod_64++] = 0x80;
    
    if (n_mod_64 > 56) {
        for(int i=n_mod_64; i < 64; i++) {
            ctx->M[i] = 0;
        }
        sha256_transform(ctx);
        n_mod_64 = 0;
    }
    
    for(int i=n_mod_64; i < 56; i++) {
        ctx->M[i] = 0;
    }
    
    ulong n_bits = ctx->N * 8;
    ctx->M[56] = (uchar)(n_bits >> 56);
    ctx->M[57] = (uchar)(n_bits >> 48);
    ctx->M[58] = (uchar)(n_bits >> 40);
    ctx->M[59] = (uchar)(n_bits >> 32);
    ctx->M[60] = (uchar)(n_bits >> 24);
    ctx->M[61] = (uchar)(n_bits >> 16);
    ctx->M[62] = (uchar)(n_bits >> 8);
    ctx->M[63] = (uchar)(n_bits);

    sha256_transform(ctx);
    
    // --- OPTIMIZATION: Early Exit & Difficulty Check using clz() ---
    uint bits = 0;
    for (int i = 0; i < 8; i++) {
        uint h_val = ctx->H[i];
        uint leading_zeros = clz(h_val); 
        bits += leading_zeros;
        if (leading_zeros < 32) {
            return (bits >= difficulty);
        }
        if (bits >= difficulty) return true;
    }
    return (bits >= difficulty);
}

// Helper to write the final hash to global memory
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
    sha256_update(&ctx, base_line, base_len);
    *midstate_out = ctx;
}


// --- ADDED: Fast, branchless u64-to-hex converter ---
inline void ulong_to_hex_fixed16(ulong n, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    
    // Unroll the loop manually for 16 characters (64 bits)
    out[0]  = hex_chars[(n >> 60) & 0xF];
    out[1]  = hex_chars[(n >> 56) & 0xF];
    out[2]  = hex_chars[(n >> 52) & 0xF];
    out[3]  = hex_chars[(n >> 48) & 0xF];
    out[4]  = hex_chars[(n >> 44) & 0xF];
    out[5]  = hex_chars[(n >> 40) & 0xF];
    out[6]  = hex_chars[(n >> 36) & 0xF];
    out[7]  = hex_chars[(n >> 32) & 0xF];
    out[8]  = hex_chars[(n >> 28) & 0xF];
    out[9]  = hex_chars[(n >> 24) & 0xF];
    out[10] = hex_chars[(n >> 20) & 0xF];
    out[11] = hex_chars[(n >> 16) & 0xF];
    out[12] = hex_chars[(n >> 12) & 0xF];
    out[13] = hex_chars[(n >> 8)  & 0xF];
    out[14] = hex_chars[(n >> 4)  & 0xF];
    out[15] = hex_chars[n & 0xF];
}


/*
 * == KERNEL 2: Find Hash (Run 10M+ times) ==
 *
 * (CORRECTED to use hex-string nonce)
 */
__kernel void find_hash_kernel(
    __constant const sha256_ctx* midstate, // <-- OPTIMIZATION: Use __constant
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

    // 2. --- MODIFICATION: Convert u64 nonce to 16-char hex string ---
    char nonce_hex[16];
    ulong_to_hex_fixed16(nonce, nonce_hex);
    
    // 3. Update context with the 16-char hex string
    sha256_update_local(&ctx, (const uchar*)nonce_hex, 16);

    // 4. Finalize hash and check difficulty (now much faster)
    if (sha256_final_and_check(&ctx, difficulty)) {
        // We found a winner!
        if (atomic_cmpxchg(result_local_id, (uint)(-1), (uint)global_id) == (uint)(-1)) {
            // We were first! Write our hash to the output buffer.
            write_hash_from_H(result_hash, &ctx);
        }
    }
}
