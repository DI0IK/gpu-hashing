/*
A Rust-based hashing game client, rewritten from the Java original.

This version has been updated to support BOTH:
1. CPU Mining (using `rayon` for all cores)
2. GPU Mining (using `ocl` for OpenCL devices)

Set the `USE_GPU` const in `main()` to switch between them.

It also includes a 5-second logging thread to report the
combined hash rate from all miners.

== How to Compile ==
1. Install Rust: https://www.rust-lang.org/tools/install
2. (For GPU) Install OpenCL drivers for your GPU:
   - NVIDIA: Install the "CUDA Toolkit" (includes OpenCL)
   - AMD: Install the latest "AMD Software: Adrenalin Edition"
   - Intel: Install the "Intel SDK for OpenCL"
3. Create a new project: `cargo new rust_miner`
4. Enter the project: `cd rust_miner`
5. Copy `Cargo.toml` into your `Cargo.toml`
6. Create `src/gpu_miner.rs` and copy the code into it.
7. Copy this code into `src/main.rs`
8. Run: `cargo run --release` (release mode is crucial for performance)

*/

use rand::{thread_rng, Rng};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
// --- FIXED: Removed unused prelude ---
// use rayon::prelude::*;
// --- ADDED: Use rayon's parallel iterator trait ---
use rayon::iter::ParallelIterator;

// --- ADDED ---
// This brings the new `gpu_miner.rs` file into our project as a module.
mod gpu_miner;

// Use lazy_static to create a global, mutable timestamp for rate limiting.
lazy_static::lazy_static! {
    static ref LAST_REQUEST: Mutex<Instant> = Mutex::new(Instant::now() - Duration::from_secs(5));
    // --- ADDED: Global counter for hashes ---
    pub static ref TOTAL_HASHES: AtomicU64 = AtomicU64::new(0);
}

const SERVER_URL: &str = "http://hash.h10a.de/?raw";
const MIN_REQUEST_GAP: Duration = Duration::from_millis(2100);

struct HashClient {
    http_client: Client,
    name: String,
}

#[derive(Debug)]
struct ServerState {
    difficulty: usize,
    parent_hash: String,
}

/// Counts the number of leading zero bits in a byte array (hash).
fn count_zero_bits(bytes: &[u8]) -> usize {
    let mut bits = 0;
    for &byte in bytes {
        if byte == 0 {
            bits += 8;
        } else {
            // Count leading zeros in the first non-zero byte
            let mut b = byte;
            while (b & 0x80) == 0 {
                bits += 1;
                b <<= 1;
            }
            break; // Stop after the first non-zero byte
        }
    }
    bits
}

impl HashClient {
    fn new(name: String) -> Self {
        HashClient {
            http_client: Client::new(),
            name,
        }
    }

    /// Creates the input string to be hashed: "parent name seed"
    fn get_line(&self, parent: &str, seed: &str) -> String {
        format!("{} {} {}", parent, self.name, seed)
    }

    /// Performs a single SHA-256 hash.
    fn hash(&self, parent: &str, seed: &str) -> Vec<u8> {
        let line = self.get_line(parent, seed);
        let mut hasher = Sha256::new();
        hasher.update(line.as_bytes());
        hasher.finalize().to_vec()
    }

    /// Connects to the server (with rate-limiting) to get the current state.
    fn get_server_state(&self, url: &str) -> Result<ServerState, String> {
        // --- Rate Limiting ---
        {
            let mut last_req = LAST_REQUEST.lock().unwrap();
            let elapsed = last_req.elapsed();
            if elapsed < MIN_REQUEST_GAP {
                let sleep_time = MIN_REQUEST_GAP - elapsed;
                println!("[Net] Rate limiting: sleeping for {:?}", sleep_time);
                thread::sleep(sleep_time);
            }
            *last_req = Instant::now();
        } // Mutex lock is released here
          // --- End Rate Limiting ---

        println!("\n[Net] Connecting to server: {}", url);
        let mut res = self
            .http_client
            .get(url)
            .timeout(Duration::from_secs(10))
            .send()
            .map_err(|e| e.to_string())?;

        let mut body = String::new();
        res.read_to_string(&mut body).map_err(|e| e.to_string())?;

        let mut lines = body.lines();

        // 1. Parse Difficulty
        let difficulty_line = lines.next().ok_or("Empty server response")?;
        let difficulty = difficulty_line
            .trim()
            .parse::<usize>()
            .map_err(|e| format!("Failed to parse difficulty: {}", e))?;
        println!("[Net] Difficulty: {}", difficulty);

        // 2. Parse Parent Hash
        let mut best_level = -1;
        let mut parent_hash = String::new();

        for line in lines {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                if let Ok(level) = parts[1].trim().parse::<i32>() {
                    if level >= best_level {
                        best_level = level;
                        parent_hash = parts[0].trim().to_string();
                    }
                }
            }
        }

        if parent_hash.is_empty() {
            Err("Failed to find a parent hash from server response".to_string())
        } else {
            println!(
                "[Net] New parent hash: {} (Level: {})",
                parent_hash, best_level
            );
            Ok(ServerState {
                difficulty,
                parent_hash,
            })
        }
    }

    /// Gets the latest parent hash from the default server URL.
    fn get_latest_parent(&self) -> Result<ServerState, String> {
        self.get_server_state(SERVER_URL)
    }

    /// Submits a found seed to the server.
    fn send_seed(&self, parent: &str, seed: &str) -> Result<ServerState, String> {
        println!("[Net] Submitting seed to server...");
        let url = format!("{}&Z={}&P={}&R={}", SERVER_URL, parent, self.name, seed);
        self.get_server_state(&url)
    }

    /// Finds a valid seed by parallel searching across all CPU cores.
    /// (Renamed from `find_seed` to `find_seed_cpu`)
    fn find_seed_cpu(&self, parent: &str, difficulty: usize) -> (String, String) {
        println!(
            "[CPU] Finding seed for parent {} with difficulty {} (using {} threads)...",
            parent,
            difficulty,
            rayon::current_num_threads()
        );

        // --- FIXED: Replaced AtomicOption with a simple Mutex ---
        // This will store Some((seed, hash_hex)) once found.
        let result = Arc::new(Mutex::new(None::<(String, String)>));
        // This Atomic is a flag to tell other threads to stop working.
        let found = Arc::new(AtomicBool::new(false));

        // --- FIXED: Use `rayon::iter::repeat` for a parallel infinite iterator ---
        // --- FIXED: Clone the Arc `result` for the closure ---
        rayon::iter::repeat(()).for_each_with(
            (result.clone(), found),
            |(result_clone, found_clone), _| {
                // Check if another thread has already found a solution.
                if found_clone.load(Ordering::SeqCst) {
                    return; // Stop this thread's work
                }

                // Get a thread-local random number generator.
                let mut rng = thread_rng();
                // Generate a random u64 and format it as a hex string.
                let seed = format!("{:x}", rng.gen::<u64>());

                // Perform the hash
                let hash_bytes = self.hash(parent, &seed);
                let bits = count_zero_bits(&hash_bytes);

                // --- ADDED: Increment global hash counter ---
                TOTAL_HASHES.fetch_add(1, Ordering::Relaxed);

                // Check if we found a valid hash
                if bits >= difficulty {
                    // Try to set the 'found' flag.
                    if found_clone
                        .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                        .is_ok()
                    {
                        let hash_hex = hex::encode(&hash_bytes);
                        println!(
                            "\n+++ [CPU] Found valid seed! (Thread {}) +++",
                            rayon::current_thread_index().unwrap_or(0)
                        );
                        println!("    Seed: {}", seed);
                        println!("    Hash: {} (Bits: {})", hash_hex, bits);

                        // --- FIXED: Store the result in the Mutex ---
                        let mut res_guard = result_clone.lock().unwrap();
                        *res_guard = Some((seed, hash_hex));
                    }
                }
            },
        );

        // --- FIXED: Wait for the Mutex to be populated ---
        loop {
            // Check if the Option inside the Mutex is Some
            if let Some(res) = result.lock().unwrap().clone() {
                // clone the (String, String) out and return it
                return res;
            }
            // Yield to the scheduler to avoid a busy-wait spin
            thread::sleep(Duration::from_millis(10));
        }
    }
}

// --- ADDED: Helper function for formatting hash rate ---
fn format_hash_rate(rate: f64) -> String {
    if rate < 1_000.0 {
        format!("{:.2} H/s", rate)
    } else if rate < 1_000_000.0 {
        format!("{:.2} kH/s", rate / 1_000.0)
    } else if rate < 1_000_000_000.0 {
        format!("{:.2} MH/s", rate / 1_000_000.0)
    } else {
        format!("{:.2} GH/s", rate / 1_000_000_000.0)
    }
}

fn main() {
    // --- CHOOSE YOUR MINER ---
    // `true` = Use OpenCL GPU Miner
    // `false` = Use Rayon CPU Miner
    const USE_GPU: bool = true;
    // -------------------------

    // --- ADDED: Spawn hash rate logging thread ---
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(5));
            // Atomically read the count and reset it to 0
            let hashes = TOTAL_HASHES.swap(0, Ordering::SeqCst);
            // Calculate rate (hashes per second)
            let rate = (hashes as f64) / 5.0;
            println!("[Stats] Local hash rate: {}", format_hash_rate(rate));
        }
    });
    // ---------------------------------------------

    // 1. Get client name
    let mut name = hostname::get().map_or("RustClient".to_string(), |s| {
        s.to_string_lossy().to_string()
    });

    // Allow overriding name with an environment variable
    if let Ok(env_name) = std::env::var("HASH_NAME") {
        if !env_name.is_empty() {
            name = env_name;
        }
    }

    // ---
    // --- Set your nickname here if you don't want to use hostname:
    // name = "MyMiner".to_string();
    // ---

    println!("Starting Rust HashClient...");
    println!("Using name: {}", name);

    let client = HashClient::new(name);
    let mut current_parent: String;
    let mut current_difficulty: usize;

    // --- ADDED: Initialize GPU Miner (if selected) ---
    // This is a bit of a trick: we create an `Option` to hold the miner.
    // `gpu_miner::GpuMiner::new()` will find the GPU and compile the kernel.
    // This can take a few seconds, so we do it once at the start.
    let mut gpu_miner_instance = if USE_GPU {
        println!("[GPU] Initializing OpenCL GPU miner...");
        match gpu_miner::GpuMiner::new() {
            Ok(miner) => {
                println!("[GPU] GPU Miner initialized successfully!");
                Some(miner)
            }
            Err(e) => {
                eprintln!("[GPU] FATAL: Failed to initialize GPU miner: {}", e);
                eprintln!("[GPU] Check your OpenCL drivers. Falling back to CPU.");
                None
            }
        }
    } else {
        None
    };
    // ------------------------------------------------

    // Get initial parent
    match client.get_latest_parent() {
        Ok(state) => {
            current_parent = state.parent_hash;
            current_difficulty = state.difficulty;
        }
        Err(e) => {
            eprintln!("Fatal Error: Could not get initial parent: {}", e);
            return;
        }
    }

    // Main game loop
    loop {
        // --- MODIFIED: Choose CPU or GPU path ---
        let (seed, our_hash_hex) = if let Some(miner) = &mut gpu_miner_instance {
            // --- GPU Path ---
            println!(
                "[GPU] Finding seed for parent {} with difficulty {}...",
                current_parent, current_difficulty
            );
            // We create the "base" string for the GPU: "parent name "
            let base_line = client.get_line(&current_parent, "");

            match miner.find_seed_gpu(&base_line, current_difficulty) {
                Ok(gpu_result) => {
                    // gpu_result is (seed, hash_hex)
                    gpu_result
                }
                Err(e) => {
                    eprintln!(
                        "[GPU] GPU miner failed: {}. Falling back to CPU for this round.",
                        e
                    );
                    // Fallback to CPU
                    client.find_seed_cpu(&current_parent, current_difficulty)
                }
            }
        } else {
            // --- CPU Path (Original) ---
            client.find_seed_cpu(&current_parent, current_difficulty)
        };
        // -----------------------------------------

        // 2. Submit the seed to the server
        match client.send_seed(&current_parent, &seed) {
            Ok(new_state) => {
                // 3. Verify the server accepted our hash
                if new_state.parent_hash == our_hash_hex {
                    println!(
                        "[Net] Server accepted our hash! New parent is: {}",
                        new_state.parent_hash
                    );
                } else {
                    println!("[Net] !!! Server did NOT accept our hash (someone was faster). !!!");
                    println!("    Our hash:   {}", our_hash_hex);
                    println!("    New parent: {}", new_state.parent_hash);
                }
                // Update state for the next round
                current_parent = new_state.parent_hash;
                current_difficulty = new_state.difficulty;
            }
            Err(e) => {
                eprintln!("[Net] Error submitting seed: {}. Retrying...", e);
                // On error, just try to get the latest parent and continue
                match client.get_latest_parent() {
                    Ok(state) => {
                        current_parent = state.parent_hash;
                        current_difficulty = state.difficulty;
                    }
                    Err(e) => {
                        eprintln!("Fatal Error: Could not re-sync with server: {}", e);
                        thread::sleep(Duration::from_secs(10)); // Wait before retrying
                    }
                }
            }
        }
    }
}
