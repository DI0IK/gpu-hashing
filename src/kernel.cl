/*
 * OpenCL SHA-256 Kernel (Optimized for Midstate)
 *
 * This kernel is split into a pre-calculator for the constant
 * 'base_line' and a main kernel for checking nonces.
 */

// --- SHA-256 Constants and Macros ---
// (Same as before)
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
// (Must match the Rust struct)
typedef struct {
    uint H[8];
    uchar M[64];
    ulong N;
} sha256_ctx;


// --- SHA-256 Core Functions (Refactored) ---

// Single SHA-256 transform
inline void sha256_transform(sha256_ctx *ctx) {
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
// --- MODIFIED: Returns 'bool' (pass/fail) ---
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
    
    // --- OPTIMIZATION: Early Exit & Difficulty Check ---
    // Check H registers directly, avoid creating hash_output array
    
    int bits = 0;
    for (int i = 0; i < 8; i++) {
        uint h_val = ctx->H[i];
        
        // Check 4 bytes of this H register
        uchar b;
        
        b = (uchar)(h_val >> 24); // Byte 0
        if (b == 0) { bits += 8; } 
        else { while ((b & 0x80) == 0) { bits++; b <<= 1; } break; }
        if (bits >= difficulty) return true;

        b = (uchar)(h_val >> 16); // Byte 1
        if (b == 0) { bits += 8; } 
        else { while ((b & 0x80) == 0) { bits++; b <<= 1; } break; }
        if (bits >= difficulty) return true;

        b = (uchar)(h_val >> 8); // Byte 2
        if (b == 0) { bits += 8; } 
        else { while ((b & 0x80) == 0) { bits++; b <<= 1; } break; }
        if (bits >= difficulty) return true;

        b = (uchar)(h_val); // Byte 3
        if (b == 0) { bits += 8; } 
        else { while ((b & 0x80) == 0) { bits++; b <<= 1; } break; }
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

// --- Helper: ulong to hex string (Same as before) ---
inline int ulong_to_hex(ulong n, char* out) {
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

    for (int j = 0; j < i; j++) {
        out[j] = buf[i - 1 - j];
    }
    return i;
}


/*
 * == KERNEL 1: Calculate Midstate (Run ONCE) ==
 *
 * Takes the constant base_line, hashes it, and saves the
 * resulting SHA-256 context (H, M, N) as the "midstate".
 */
__kernel void calculate_midstate_kernel(
    __global const uchar* base_line,
    int base_len,
    __global sha256_ctx* midstate_out
) {
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, base_line, base_len);
    
    // Write the entire context struct to global memory
    *midstate_out = ctx;
}


/*
 * == KERNEL 2: Find Hash (Run 10M+ times) ==
 *
 * Starts from the pre-calculated midstate, adds the
 * unique nonce, and checks for a valid hash.
 */
__kernel void find_hash_kernel(
    __global const sha256_ctx* midstate, // <-- INPUT
    ulong start_nonce,
    uint difficulty,
    __global uint* result_local_id,
    __global uchar* result_hash
) {
    ulong global_id = get_global_id(0);
    ulong nonce = start_nonce + global_id;

    // --- Optimization: Check if already found ---
    if (*result_local_id != (uint)(-1)) {
        return;
    }

    // 1. Load the pre-calculated midstate
    sha256_ctx ctx = *midstate; // Fast struct copy

    // 2. Construct nonce hex
    char nonce_hex[17];
    int nonce_len = ulong_to_hex(nonce, nonce_hex);
    
    // 3. Update context with nonce_hex
    sha256_update_local(&ctx, (const uchar*)nonce_hex, nonce_len);

    // 4. Finalize hash and check difficulty
    if (sha256_final_and_check(&ctx, difficulty)) {
        // We found a winner!
        // Atomically write our global_id (if no one else has)
        if (atomic_cmpxchg(result_local_id, (uint)(-1), (uint)global_id) == (uint)(-1)) {
            // We were first! Write our hash to the output buffer.
            write_hash_from_H(result_hash, &ctx);
        }
    }
}
