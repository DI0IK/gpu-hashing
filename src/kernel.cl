/*
 * OpenCL SHA-256 Kernel (Version 4: Final Block Caching)
 *
 * This kernel is hyper-optimized for the specific workload.
 *
 * All padding logic (`0x80`, zero-padding, bit-length) is
 * pre-calculated by `calculate_setup_kernel` and stored in
 * `final_block_template`.
 *
 * The main `find_hash_kernel` is now just a high-speed assembler:
 * 1. Copy H-state (from midstate)
 * 2. Copy M-template (from final_block_template)
 * 3. Write 16-byte hex nonce into M
 * 4. Run ONE transform
 * 5. Check difficulty
 */

// --- SHA-256 Constants and Macros (Unchanged) ---
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

// --- Byte-to-Hex Lookup Table (from v3) ---
constant uchar2 hex_lookup[256] = {
    (uchar2)('0','0'), (uchar2)('0','1'), (uchar2)('0','2'), (uchar2)('0','3'),
    (uchar2)('0','4'), (uchar2)('0','5'), (uchar2)('0','6'), (uchar2)('0','7'),
    (uchar2)('0','8'), (uchar2)('0','9'), (uchar2)('0','a'), (uchar2)('0','b'),
    (uchar2)('0','c'), (uchar2)('0','d'), (uchar2)('0','e'), (uchar2)('0','f'),
    (uchar2)('1','0'), (uchar2)('1','1'), (uchar2)('1','2'), (uchar2)('1','3'),
    (uchar2)('1','4'), (uchar2)('1','5'), (uchar2)('1','6'), (uchar2)('1','7'),
    (uchar2)('1','8'), (uchar2)('1','9'), (uchar2)('1','a'), (uchar2)('1','b'),
    (uchar2)('1','c'), (uchar2)('1','d'), (uchar2)('1','e'), (uchar2)('1','f'),
    (uchar2)('2','0'), (uchar2)('2','1'), (uchar2)('2','2'), (uchar2)('2','3'),
    (uchar2)('2','4'), (uchar2)('2','5'), (uchar2)('2','6'), (uchar2)('2','7'),
    (uchar2)('2','8'), (uchar2)('2','9'), (uchar2)('2','a'), (uchar2)('2','b'),
    (uchar2)('2','c'), (uchar2)('2','d'), (uchar2)('2','e'), (uchar2)('2','f'),
    (uchar2)('3','0'), (uchar2)('3','1'), (uchar2)('3','2'), (uchar2)('3','3'),
    (uchar2)('3','4'), (uchar2)('3','5'), (uchar2)('3','6'), (uchar2)('3','7'),
    (uchar2)('3','8'), (uchar2)('3','9'), (uchar2)('3','a'), (uchar2)('3','b'),
    (uchar2)('3','c'), (uchar2)('3','d'), (uchar2)('3','e'), (uchar2)('3','f'),
    (uchar2)('4','0'), (uchar2)('4','1'), (uchar2)('4','2'), (uchar2)('4','3'),
    (uchar2)('4','4'), (uchar2)('4','5'), (uchar2)('4','6'), (uchar2)('4','7'),
    (uchar2)('4','8'), (uchar2)('4','9'), (uchar2)('4','a'), (uchar2)('4','b'),
    (uchar2)('4','c'), (uchar2)('4','d'), (uchar2)('4','e'), (uchar2)('4','f'),
    (uchar2)('5','0'), (uchar2)('5','1'), (uchar2)('5','2'), (uchar2)('5','3'),
    (uchar2)('5','4'), (uchar2)('5','5'), (uchar2)('5','6'), (uchar2)('5','7'),
    (uchar2)('5','8'), (uchar2)('5','9'), (uchar2)('5','a'), (uchar2)('5','b'),
    (uchar2)('5','c'), (uchar2)('5','d'), (uchar2)('5','e'), (uchar2)('5','f'),
    (uchar2)('6','0'), (uchar2)('6','1'), (uchar2)('6','2'), (uchar2)('6','3'),
    (uchar2)('6','4'), (uchar2)('6','5'), (uchar2)('6','6'), (uchar2)('6','7'),
    (uchar2)('6','8'), (uchar2)('6','9'), (uchar2)('6','a'), (uchar2)('6','b'),
    (uchar2)('6','c'), (uchar2)('6','d'), (uchar2)('6','e'), (uchar2)('6','f'),
    (uchar2)('7','0'), (uchar2)('7','1'), (uchar2)('7','2'), (uchar2)('7','3'),
    (uchar2)('7','4'), (uchar2)('7','5'), (uchar2)('7','6'), (uchar2)('7','7'),
    (uchar2)('7','8'), (uchar2)('7','9'), (uchar2)('7','a'), (uchar2)('7','b'),
    (uchar2)('7','c'), (uchar2)('7','d'), (uchar2)('7','e'), (uchar2)('7','f'),
    (uchar2)('8','0'), (uchar2)('8','1'), (uchar2)('8','2'), (uchar2)('8','3'),
    (uchar2)('8','4'), (uchar2)('8','5'), (uchar2)('8','6'), (uchar2)('8','7'),
    (uchar2)('8','8'), (uchar2)('8','9'), (uchar2)('8','a'), (uchar2)('8','b'),
    (uchar2)('8','c'), (uchar2)('8','d'), (uchar2)('8','e'), (uchar2)('8','f'),
    (uchar2)('9','0'), (uchar2)('9','1'), (uchar2)('9','2'), (uchar2)('9','3'),
    (uchar2)('9','4'), (uchar2)('9','5'), (uchar2)('9','6'), (uchar2)('9','7'),
    (uchar2)('9','8'), (uchar2)('9','9'), (uchar2)('9','a'), (uchar2)('9','b'),
    (uchar2)('9','c'), (uchar2)('9','d'), (uchar2)('9','e'), (uchar2)('9','f'),
    (uchar2)('a','0'), (uchar2)('a','1'), (uchar2)('a','2'), (uchar2)('a','3'),
    (uchar2)('a','4'), (uchar2)('a','5'), (uchar2)('a','6'), (uchar2)('a','7'),
    (uchar2)('a','8'), (uchar2)('a','9'), (uchar2)('a','a'), (uchar2)('a','b'),
    (uchar2)('a','c'), (uchar2)('a','d'), (uchar2)('a','e'), (uchar2)('a','f'),
    (uchar2)('b','0'), (uchar2)('b','1'), (uchar2)('b','2'), (uchar2)('b','3'),
    (uchar2)('b','4'), (uchar2)('b','5'), (uchar2)('b','6'), (uchar2)('b','7'),
    (uchar2)('b','8'), (uchar2)('b','9'), (uchar2)('b','a'), (uchar2)('b','b'),
    (uchar2)('b','c'), (uchar2)('b','d'), (uchar2)('b','e'), (uchar2)('b','f'),
    (uchar2)('c','0'), (uchar2)('c','1'), (uchar2)('c','2'), (uchar2)('c','3'),
    (uchar2)('c','4'), (uchar2)('c','5'), (uchar2)('c','6'), (uchar2)('c','7'),
    (uchar2)('c','8'), (uchar2)('c','9'), (uchar2)('c','a'), (uchar2)('c','b'),
    (uchar2)('c','c'), (uchar2)('c','d'), (uchar2)('c','e'), (uchar2)('c','f'),
    (uchar2)('d','0'), (uchar2)('d','1'), (uchar2)('d','2'), (uchar2)('d','3'),
    (uchar2)('d','4'), (uchar2)('d','5'), (uchar2)('d','6'), (uchar2)('d','7'),
    (uchar2)('d','8'), (uchar2)('d','9'), (uchar2)('d','a'), (uchar2)('d','b'),
    (uchar2)('d','c'), (uchar2)('d','d'), (uchar2)('d','e'), (uchar2)('d','f'),
    (uchar2)('e','0'), (uchar2)('e','1'), (uchar2)('e','2'), (uchar2)('e','3'),
    (uchar2)('e','4'), (uchar2)('e','5'), (uchar2)('e','6'), (uchar2)('e','7'),
    (uchar2)('e','8'), (uchar2)('e','9'), (uchar2)('e','a'), (uchar2)('e','b'),
    (uchar2)('e','c'), (uchar2)('e','d'), (uchar2)('e','e'), (uchar2)('e','f'),
    (uchar2)('f','0'), (uchar2)('f','1'), (uchar2)('f','2'), (uchar2)('f','3'),
    (uchar2)('f','4'), (uchar2)('f','5'), (uchar2)('f','6'), (uchar2)('f','7'),
    (uchar2)('f','8'), (uchar2)('f','9'), (uchar2)('f','a'), (uchar2)('f','b'),
    (uchar2)('f','c'), (uchar2)('f','d'), (uchar2)('f','e'), (uchar2)('f','f')
};

// --- SHA-256 Context Struct (Unchanged) ---
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

// Update context with GLOBAL data (for setup kernel) (Unchanged)
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

// --- OPTIMIZATION: Gutted finalization ---
// This function assumes ctx->M is ALREADY the fully padded final block.
// Its only jobs are to transform and check.
inline bool sha256_final_and_check_fast(sha256_ctx *ctx, uint difficulty) {
    // 4. Run the final transform
    sha256_transform(ctx);
    
    // 5. OPTIMIZATION: Unrolled Difficulty Check (Unchanged)
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
 * == KERNEL 1: Calculate Midstate AND Final Block Template (Run ONCE) ==
 */
__kernel void calculate_setup_kernel( // Renamed
    __global const uchar* base_line,
    int base_len,
    __global sha256_ctx* midstate_out,           // Output 1
    __global uchar* final_block_template_out, // Output 2
    __global uint* n_mod_64_nonce_start_out   // Output 3
) {
    // 1. Calculate midstate as before
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update_global(&ctx, base_line, base_len);
    *midstate_out = ctx;

    // 2. Now, prepare the final block template
    
    // We add 16 bytes for the future hex nonce
    ulong final_N = ctx.N + 16; 
    int n_mod_64_final = (int)(final_N % 64);
    uint n_mod_64_start = (uint)(ctx.N % 64);

    uchar M_template[64];
    
    // 2a. --- FIX: Copy the partial data from ctx.M ---
    // This copies the name part.
    for(int i=0; i<n_mod_64_start; i++) {
        M_template[i] = ctx.M[i];
    }
    
    // 2b. Zero-init the *rest* of the template.
    // This zeroes the 16 bytes for the nonce, and all padding areas.
    for(int i=n_mod_64_start; i<64; i++) {
        M_template[i] = 0;
    }

    // 2c. Add padding byte
    M_template[n_mod_64_final] = 0x80;
    
    // 2d. Zero pad (from padding byte to byte 56)
    // 2e. Write 64-bit length
    ulong n_bits = final_N * 8;
    M_template[56] = (uchar)(n_bits >> 56);
    M_template[57] = (uchar)(n_bits >> 48);
    M_template[58] = (uchar)(n_bits >> 40);
    M_template[59] = (uchar)(n_bits >> 32);
    M_template[60] = (uchar)(n_bits >> 24);
    M_template[61] = (uchar)(n_bits >> 16);
    M_template[62] = (uchar)(n_bits >> 8);
    M_template[63] = (uchar)(n_bits);

    // 3. Write the template to global memory
    for(int j=0; j<64; j++) {
        final_block_template_out[j] = M_template[j];
    }
    
    // 4. Write the *nonce insertion offset*
    *n_mod_64_nonce_start_out = (uint)(ctx.N % 64);
}


/*
 * == KERNEL 2: Find Hash (Run 10M+ times) ==
 *
 * (HYPER-OPTIMIZED)
 */
// Force the compiler to optimize for this exact work-group size
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void find_hash_kernel(
    __constant const sha256_ctx* midstate,         // Input 1
    __constant const uchar* final_block_template, // Input 2
    uint n_mod_64_nonce_start,                     // Input 3
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

    // 1. Load the pre-calculated H-state into private registers
    sha256_ctx ctx;
    ctx.H[0] = midstate->H[0];
    ctx.H[1] = midstate->H[1];
    ctx.H[2] = midstate->H[2];
    ctx.H[3] = midstate->H[3];
    ctx.H[4] = midstate->H[4];
    ctx.H[5] = midstate->H[5];
    ctx.H[6] = midstate->H[6];
    ctx.H[7] = midstate->H[7];
    // ctx.N is not needed for the final transform

    // 2. Load the pre-padded final M-block (fast 64-byte vector copy)
    *(__private ulong8*)ctx.M = *(__constant ulong8*)final_block_template;
    
    // 3. --- OPTIMIZATION: Write hex nonce DIRECTLY into ctx.M ---
    // We overwrite the (zeroed) nonce-bytes in the template
    __private uchar2* hex_ptr = (__private uchar2*)&ctx.M[n_mod_64_nonce_start];

    hex_ptr[0] = hex_lookup[(nonce >> 56) & 0xFF];
    hex_ptr[1] = hex_lookup[(nonce >> 48) & 0xFF];
    hex_ptr[2] = hex_lookup[(nonce >> 40) & 0xFF];
    hex_ptr[3] = hex_lookup[(nonce >> 32) & 0xFF];
    hex_ptr[4] = hex_lookup[(nonce >> 24) & 0xFF];
    hex_ptr[5] = hex_lookup[(nonce >> 16) & 0xFF];
    hex_ptr[6] = hex_lookup[(nonce >> 8)  & 0xFF];
    hex_ptr[7] = hex_lookup[ nonce        & 0xFF];
    
    // 4. Finalize hash and check difficulty (now extremely fast)
    if (sha256_final_and_check_fast(&ctx, difficulty)) {
        // We found a winner!
        if (atomic_cmpxchg(result_local_id, (uint)(-1), (uint)global_id) == (uint)(-1)) {
            // We were first! Write our hash to the output buffer.
            write_hash_from_H(result_hash, &ctx);
        }
    }
}
