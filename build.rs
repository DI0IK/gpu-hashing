use rust_cuda_nvcc::Compiler;

fn main() {
    let ptx_path = Compiler::new()
        .gpu_arch("sm_50") // Target Maxwell (2014) and newer for LOP3.LUT
        .source("src/kernel.cu")
        .compile()
        .expect("Failed to compile CUDA kernel");

    // This environment variable is the magic.
    // We can now access this path from our main Rust code.
    println!("cargo:rustc-env=NVCC_PTX_PATH={}", ptx_path.display());

    // Re-run this build script if the kernel changes
    println!("cargo:rerun-if-changed=src/kernel.cu");
}
