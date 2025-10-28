## 1. The Core Hashing Problem & Insight

First, let's analyze the precise input format, as it is the key to all subsequent optimizations.

**Input String:** `<parent_64_byte><space><username_1_to_16_byte><space><nonce_16_byte>`

Let's break this down in terms of 64-byte (512-bit) SHA-256 blocks:

1.  **`base_line`:** This is the portion passed from Rust: `<parent_64_byte><space><username...><space>`.
    - Its length is `64 + 1 + len(username) + 1`, so it's between **67 and 82 bytes**.
2.  **`nonce_str`:** This is the 16-byte hex representation of the `u64` nonce, generated _inside the kernel_.
3.  **Total Data:** The full string to be hashed is `base_line + nonce_str`.
    - Total length is between **83 bytes** (67+16) and **98 bytes** (82+16).

**The "Aha!" Moment:**
Since the total data is _always_ between 83 and 98 bytes, the entire SHA-256 operation for _any_ nonce will _always_ consist of exactly two SHA-256 transforms:

- **Block 1:** The first 64 bytes of the `base_line` (which is just the `<parent_64_byte>`).
- **Block 2:** The remaining `base_line` (3 to 18 bytes) + the 16-byte `nonce_str` + SHA-256 padding.

This structure is predictable and exploitable. The _entire_ first transform (Block 1) is **100% constant** for the entire job. The _second_ transform (Block 2) is _mostly_ constant, except for the 16 bytes where the nonce is injected.

This code exploits this by splitting the problem into two kernels.

---

## 2. The Two-Kernel "Architect & Sprinter" Architecture

### 🚀 Kernel 1: `calculate_setup_kernel` (The "Architect")

This kernel runs **only once** per `find_seed_gpu` call (on a single thread). Its job is to build a "kit" for the main mining kernel. It pre-calculates three critical pieces of data.

**Input:** `base_line` (e.g., "ParentHash... user1 ")
**Output 1: The Midstate (`midstate_out`)**

- The kernel initializes a SHA-256 context (`sha256_init`).
- It feeds the _entire_ `base_line` into `sha256_update_global`.
- Crucially, this function processes all _full blocks_. Since our `base_line` is 67-82 bytes, it processes the first 64-byte block (the parent hash) and runs `sha256_transform` on it.
- It stops with the remaining 3-18 bytes (`<space><username><space>`) sitting in the `ctx.M` buffer.
- The kernel then saves the _entire_ `sha256_ctx` struct. The 8 `H` values (`ctx.H`) in this struct are the **"midstate"**: the internal state of the SHA-256 algorithm _after_ having processed the parent hash.

**Output 2: The Final Block Template (`final_block_template_out`)**

- This is the most brilliant part. The kernel _constructs the entire second data block_ before it even knows the nonce.
- It creates a 64-byte `M_template` buffer.
- **Step (a):** It copies the partial data (the 3-18 bytes of `<space><username><space>`) from `ctx.M` into the _start_ of the template.
- **Step (b):** It **zero-fills** the _rest_ of the 64-byte template. This zeroes out the 16 bytes where the nonce _will go_ and all the subsequent padding area.
- **Step (c):** It calculates the _final_ total length (`final_N = base_len + 16`). It uses this to place the `0x80` padding byte at the correct position (e.g., at byte `18 + 16 = 34`).
- **Step (d):** It calculates the final bit-length (`n_bits = final_N * 8`) and writes this 8-byte value into the _last 8 bytes_ of the template (bytes 56-63).
- The result is a perfect, 64-byte, fully-padded final block, with a 16-byte hole of zeroes where the nonce will be written.

**Output 3: The Nonce Offset (`n_mod_64_nonce_start_out`)**

- This is simply `ctx.N % 64`, which is the length of the partial data (3-18 bytes).
- It tells the main kernel _exactly_ what byte offset to start writing the 16-byte nonce into the template.

### 🏃 Kernel 2: `find_hash_kernel` (The "Sprinter")

This kernel is the beating heart of the miner. It is launched with **10.4 million** parallel threads (`GLOBAL_WORK_SIZE`). It is "hyper-optimized" because its only job is to assemble the pre-built kit and run a _single_ SHA-256 transform.

Here is its entire logic, step-by-step:

1.  **Get ID:** `ulong nonce = start_nonce + get_global_id(0);` (Each thread gets a unique nonce to check).
2.  **Load Midstate:** `ctx.H[0] = midstate->H[0]; ...` (8 register loads).
    - **Optimization:** It loads the 8 pre-calculated `H` values from the `midstate`. It _skips_ `sha256_init` and the _entire first transform_. It starts the race halfway to the finish line.
3.  **Load Template:** `*(__private ulong8*)ctx.M = *(__constant ulong8*)final_block_template;`
    - **Optimization:** This is a **vectorized 64-byte copy**. Instead of a 64-iteration loop, it tells the GPU to move 8x8-bytes (`ulong8`) from `__constant` memory (fast, cached) to `__private` memory (registers) in a single instruction. This copies the _entire_ pre-padded final block.
4.  **Inject Nonce:** `hex_ptr[0] = hex_lookup[(nonce >> 56) & 0xFF]; ...`
    - **Optimization:** This is the _only_ real work. It does _not_ do `sprintf` or any slow string conversion. It uses bitwise shifts (`>> 56`) and masks (`& 0xFF`) to isolate each of the 8 bytes of the `u64` nonce.
    - It uses each byte as an index into the `hex_lookup` table (which is in `__constant` memory).
    - `hex_lookup` instantly returns the 2-char hex string (e.g., `0xAB` -> `('a', 'b')`).
    - It writes these 8 `uchar2` values (16 bytes total) _directly_ into the `ctx.M` template, overwriting the 16 bytes of zeroes.
5.  **Hash & Check:** `if (sha256_final_and_check_fast(&ctx, difficulty)) { ... }`
    - **Optimization:** This `sha256_final_and_check_fast` function is "gutted." It contains _no padding logic_. It just runs `sha256_transform` _once_ and then immediately checks the result.
    - The difficulty check itself is unrolled (`clz(h_val)`) for maximum speed.
6.  **Report Victory:** `if (atomic_cmpxchg(result_local_id, ...)) { ... }`
    - This is a standard atomic operation to ensure that if multiple threads find a hash in the same batch, only the _first_ one gets to write its result and stop the miner.

---

## 3. Host-Side (Rust) Driver Optimizations

The Rust code is not just a simple launcher; it's a highly efficient driver that enables the kernel's speed.

1.  **Persistent Buffers:** In `GpuMiner::new()`, the `final_template_buf` and `n_mod_64_buf` are created _once_. This avoids allocating and deallocating GPU memory on every job, which is a slow operation.
2.  **`__constant` Memory Hinting:** This is a subtle but key optimization in `find_seed_gpu`.
    - The setup kernel writes the midstate to `midstate_buf_rw` (Read-Write).
    - The host then performs a **device-to-device copy** into `midstate_buf_const` (Read-Only).
    - The main `find_hash_kernel` is given this `_const` buffer.
    - This hints to the OpenCL driver that this data is _immutable_ for the duration of the 10.4M-thread kernel launch. The driver can then broadcast this buffer to the compute units' special, ultra-fast **`__constant` cache**, which is much faster than global VRAM. The same applies to `final_block_template`.
3.  **Pass-by-Value Argument:** The `n_mod_64_val` (the nonce offset) is read back to the _host_ (CPU) _once_. Then, inside the main loop, it's passed to the kernel as a _value_ (`.set_arg(2, n_mod_64_val)`), not a buffer.
    - **Optimization:** This means the kernel gets this value (e.g., `6`) loaded _directly into a register_ as part of its startup arguments. The alternative would be forcing all 10.4 million threads to perform a global memory read from a buffer just to find out where to write the nonce. This saves 10.4 million memory accesses per batch.
4.  **Tuned Work Sizes:** `LOCAL_WORK_SIZE = 256` is a large, well-tuned work-group size. It's a multiple of 64, fitting modern GPU "wavefronts" or "warps" perfectly, which ensures maximum hardware occupancy and efficiency. `GLOBAL_WORK_SIZE = 10_485_760` is simply a very large number of batches to keep the GPU busy for a measurable amount of time before returning to the host.
5.  **Compiler Flags:**
    - `-cl-mad-enable`: Enables **Fused Multiply-Add** (MAD). The SHA-256 transform is full of additions (`T1 = ... + K[i] + W[i]`). This flag allows the compiler to fuse `(a * b) + c` operations into a single instruction, reducing total instruction count and improving throughput.
    - `-cl-no-signed-zeros`: Not relevant for integer-only SHA-256, but harmless.

## Summary of Improvements

| Improvement           | Location      | Naive Way                                            | Optimized Way                                                |
| :-------------------- | :------------ | :--------------------------------------------------- | :----------------------------------------------------------- |
| **Midstate Caching**  | Kernel        | Re-hash `parent` block for every nonce.              | Hash `parent` _once_, load midstate from `__constant` cache. |
| **Block Templating**  | Kernel        | Build, pad, and write length for every nonce.        | Build template _once_, copy from `__constant` cache.         |
| **Nonce Generation**  | Kernel        | `sprintf(..., "%016lx", nonce)` (string formatting). | 8 bit-shifts and 8 `hex_lookup` table reads.                 |
| **Memory Access**     | Kernel        | Byte-by-byte loops for copies.                       | Single-instruction `ulong8` vector copies.                   |
| **Kernel Arguments**  | Rust / Kernel | Pass nonce offset in a `__global` buffer.            | Pass nonce offset _by value_ into a private register.        |
| **Buffer Management** | Rust          | Allocate/free `template_buf` on every job.           | Allocate _once_ in `GpuMiner::new()` and re-use.             |
| **Final Hash Logic**  | Kernel        | Full `sha256_final` with padding logic.              | Gutted `_fast` version with _only_ one transform.            |
