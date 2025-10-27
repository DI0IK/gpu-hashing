use rand::{thread_rng, Rng};
use rayon::iter::ParallelIterator;
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

mod gpu_miner;

// Use lazy_static to create a global, mutable timestamp for rate limiting.
lazy_static::lazy_static! {
    static ref LAST_REQUEST: Mutex<Instant> = Mutex::new(Instant::now() - Duration::from_secs(5));
    pub static ref TOTAL_HASHES: AtomicU64 = AtomicU64::new(0);
}

const SERVER_URL: &str = "http://hash.h10a.de/?raw";
const MIN_REQUEST_GAP: Duration = Duration::from_millis(2100);
// --- ADDED: How often to check for a new parent hash ---
const CHECK_INTERVAL: Duration = Duration::from_secs(10);

// --- MODIFIED: HashClient is now Clone ---
// We need to clone it for the new checker thread.
#[derive(Clone)]
struct HashClient {
    // --- MODIFIED: Use Arc to allow cheap cloning ---
    http_client: Arc<Client>,
    name: Arc<String>,
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
            // --- MODIFIED: Wrap in Arc ---
            http_client: Arc::new(Client::new()),
            name: Arc::new(name),
        }
    }

    /// Creates the input string to be hashed: "parent name seed"
    fn get_line(&self, parent: &str, seed: &str) -> String {
        // --- MODIFIED: Dereference Arc<String> ---
        format!("{} {} {}", parent, *self.name, seed)
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
        // --- MODIFIED: Don't print, too noisy for checker
        // println!("[Net] Difficulty: {}", difficulty);

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
            // --- MODIFIED: Don't print, too noisy for checker
            // println!(
            //     "[Net] New parent hash: {} (Level: {})",
            //     parent_hash, best_level
            // );
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
        // --- MODIFIED: Call get_server_state and print results here ---
        let result = self.get_server_state(&url);
        if let Ok(state) = &result {
            println!("[Net] Difficulty: {}", state.difficulty);
            println!("[Net] New parent hash: {}", state.parent_hash);
        }
        result
    }

    /// --- MODIFIED: Updated signature and return type ---
    /// Finds a valid seed by parallel searching across all CPU cores.
    /// Returns `Some((seed, hash))` if found, `None` if interrupted.
    fn find_seed_cpu(
        &self,
        parent: &str,
        difficulty: usize,
        stop_signal: Arc<AtomicBool>,
    ) -> Option<(String, String)> {
        println!(
            "[CPU] Finding seed for parent {} with difficulty {} (using {} threads)...",
            parent,
            difficulty,
            rayon::current_num_threads()
        );

        let result = Arc::new(Mutex::new(None::<(String, String)>));

        // --- MODIFIED: Use `for_each_with` to pass in `result` ---
        rayon::iter::repeat(()).for_each_with(result.clone(), |result_clone, _| {
            // --- MODIFIED: Check the external stop_signal ---
            if stop_signal.load(Ordering::Relaxed) {
                return; // Stop this thread's work
            }

            // Get a thread-local random number generator.
            let mut rng = thread_rng();
            // Generate a random u64 and format it as a hex string.
            let seed = format!("{:x}", rng.gen::<u64>());

            // Perform the hash
            let hash_bytes = self.hash(parent, &seed);
            let bits = count_zero_bits(&hash_bytes);

            TOTAL_HASHES.fetch_add(1, Ordering::Relaxed);

            // Check if we found a valid hash
            if bits >= difficulty {
                // --- MODIFIED: Use the stop_signal as the "found" flag ---
                // Try to set the 'stop' flag. This tells all other threads
                // (CPU) and the checker thread to stop.
                if stop_signal
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

                    let mut res_guard = result_clone.lock().unwrap();
                    *res_guard = Some((seed, hash_hex));
                }
            }
        });

        // --- MODIFIED: Return the content of the Mutex ---
        // It will be `Some` if we found it, or `None` if we were stopped.
        let final_result = result.lock().unwrap().clone();
        final_result
    }
}

/// --- ADDED: Helper function for formatting hash rate ---
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

/// --- ADDED: Spawns the checker thread ---
/// This thread polls the server and sets `stop_signal` if the parent changes.
fn spawn_checker_thread(
    client: HashClient,
    current_parent: String,
    stop_signal: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        loop {
            // 1. Check if we should stop (e.g., miner found a solution)
            if stop_signal.load(Ordering::Relaxed) {
                break;
            }

            // 2. Sleep for the interval
            // --- MODIFIED: Sleep in small chunks to be responsive ---
            let sleep_end = Instant::now() + CHECK_INTERVAL;
            while Instant::now() < sleep_end {
                if stop_signal.load(Ordering::Relaxed) {
                    return; // Exit immediately
                }
                thread::sleep(Duration::from_millis(100));
            }

            // 3. Check the server
            match client.get_latest_parent() {
                Ok(state) => {
                    // 4. Compare parent hash
                    if state.parent_hash != current_parent {
                        println!(
                            "\n[Net] Stale parent detected! Server parent is {}, we are on {}.",
                            state.parent_hash, current_parent
                        );
                        // 5. Signal the miner to stop
                        if stop_signal
                            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                            .is_ok()
                        {
                            println!("[Net] Stop signal sent to miner.");
                        }
                        break; // Stop this checker thread
                    } else {
                        // println!("[Net] Parent check OK."); // Uncomment for debugging
                    }
                }
                Err(e) => {
                    println!("[Net] Checker thread error: {}. Ignoring.", e);
                }
            }
        }
    })
}

fn main() {
    // --- CHOOSE YOUR MINER ---
    const USE_GPU: bool = true;
    // -------------------------

    // --- ADDED: Spawn hash rate logging thread ---
    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(5));
        let hashes = TOTAL_HASHES.swap(0, Ordering::SeqCst);
        let rate = (hashes as f64) / 5.0;
        println!("[Stats] Local hash rate: {}", format_hash_rate(rate));
    });
    // ---------------------------------------------

    // 1. Get client name
    let mut name = hostname::get().map_or("RustClient".to_string(), |s| {
        s.to_string_lossy().to_string()
    });

    if let Ok(env_name) = std::env::var("HASH_NAME") {
        if !env_name.is_empty() {
            name = env_name;
        }
    }

    println!("Starting Rust HashClient...");
    println!("Using name: {}", name);

    let client = HashClient::new(name);
    let mut current_parent: String;
    let mut current_difficulty: usize;

    // --- ADDED: Initialize GPU Miner (if selected) ---
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
            println!("[Net] Difficulty: {}", state.difficulty);
            println!("[Net] New parent hash: {}", state.parent_hash);
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
        // --- MODIFIED: Main loop logic completely rewritten ---

        // 1. Create a new stop signal for this round
        let stop_signal = Arc::new(AtomicBool::new(false));

        // 2. Spawn the checker thread
        let checker_handle =
            spawn_checker_thread(client.clone(), current_parent.clone(), stop_signal.clone());

        // 3. Start the miner
        let mining_result = if let Some(miner) = &mut gpu_miner_instance {
            // --- GPU Path ---
            println!(
                "[GPU] Finding seed for parent {} with difficulty {}...",
                current_parent, current_difficulty
            );
            let base_line = client.get_line(&current_parent, "");

            // --- MODIFIED: Call new GPU function & handle OclResult ---
            match miner.find_seed_gpu(&base_line, current_difficulty, stop_signal.clone()) {
                Ok(Some(result)) => Some(result), // Found
                Ok(None) => None,                 // Interrupted
                Err(e) => {
                    eprintln!(
                        "[GPU] GPU miner failed: {}. Falling back to CPU for this round.",
                        e
                    );
                    // Fallback to CPU
                    client.find_seed_cpu(&current_parent, current_difficulty, stop_signal.clone())
                }
            }
        } else {
            // --- CPU Path (Original) ---
            client.find_seed_cpu(&current_parent, current_difficulty, stop_signal.clone())
        };

        // 4. Stop the checker thread
        // (It may have already stopped, this is safe)
        stop_signal.store(true, Ordering::SeqCst);
        checker_handle.join().unwrap(); // Wait for it to exit

        // 5. Process the result
        if let Some((seed, our_hash_hex)) = mining_result {
            // --- We found a hash! ---
            // 5a. Submit the seed to the server
            match client.send_seed(&current_parent, &seed) {
                Ok(new_state) => {
                    // 5b. Verify if we won the round
                    if new_state.parent_hash == our_hash_hex {
                        println!(
                            "[Net] Server accepted our hash! New parent is: {}",
                            new_state.parent_hash
                        );
                    } else {
                        println!(
                            "[Net] !!! Server did NOT accept our hash (someone was faster). !!!"
                        );
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
                            println!("[Net] Difficulty: {}", state.difficulty);
                            println!("[Net] New parent hash: {}", state.parent_hash);
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
        } else {
            // --- We were interrupted by the checker ---
            println!("[Main] Miner interrupted (stale parent). Fetching new parent...");
            match client.get_latest_parent() {
                Ok(state) => {
                    println!("[Net] Difficulty: {}", state.difficulty);
                    println!("[Net] New parent hash: {}", state.parent_hash);
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
