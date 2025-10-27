use rand::{thread_rng, Rng};
use rayon::iter::ParallelIterator;
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs;
use std::io::stdout;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// TUI deps
use crossterm::event::{Event as CEvent, KeyCode, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use tui::backend::CrosstermBackend;
use tui::layout::{Constraint, Direction, Layout};
use tui::text::{Span, Spans};
use tui::widgets::{Block, Borders, List, ListItem, Paragraph};
use tui::Terminal;

mod events;
mod gpu_miner;

// Use lazy_static to create a global, mutable timestamp for rate limiting.
lazy_static::lazy_static! {
    static ref LAST_REQUEST: Mutex<Instant> = Mutex::new(Instant::now() - Duration::from_secs(5));
    pub static ref TOTAL_HASHES: AtomicU64 = AtomicU64::new(0);
    // Global app exit flag (set by TUI 'q' or Ctrl-C)
    pub static ref APP_EXIT: AtomicBool = AtomicBool::new(false);
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

// UI state shared with the TUI thread
const BLOCKS_FILE: &str = "blocks_mined.txt";

struct UIState {
    recent_events: VecDeque<String>,
    hash_rate: f64,
    username: String,
    parent: String,
    total_blocks_mined: u64,
}

fn load_blocks_mined_from_disk() -> u64 {
    let p = Path::new(BLOCKS_FILE);
    if !p.exists() {
        return 0;
    }
    match fs::read_to_string(p) {
        Ok(s) => s.trim().parse::<u64>().unwrap_or(0),
        Err(_) => 0,
    }
}

fn save_blocks_mined_to_disk(n: u64) {
    // Write atomically if possible: write to temp then rename
    let tmp = format!("{}.tmp", BLOCKS_FILE);
    if fs::write(&tmp, n.to_string()).is_ok() {
        let _ = fs::rename(tmp, BLOCKS_FILE);
    }
}

impl UIState {
    fn new(username: String, parent: String) -> Self {
        let total_blocks_mined = load_blocks_mined_from_disk();
        UIState {
            recent_events: VecDeque::with_capacity(16),
            hash_rate: 0.0,
            username,
            parent,
            total_blocks_mined,
        }
    }

    fn push_event(&mut self, e: String) {
        const MAX_EVENTS: usize = 16;
        self.recent_events.push_front(e);
        while self.recent_events.len() > MAX_EVENTS {
            self.recent_events.pop_back();
        }
    }

    fn increment_blocks(&mut self) {
        self.total_blocks_mined = self.total_blocks_mined.saturating_add(1);
        save_blocks_mined_to_disk(self.total_blocks_mined);
    }
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
                events::publish_event(&format!(
                    "[Net] Rate limiting: sleeping for {:?}",
                    sleep_time
                ));
                thread::sleep(sleep_time);
            }
            *last_req = Instant::now();
        } // Mutex lock is released here
          // --- End Rate Limiting ---

        // println!("\n[Net] Connecting to server: {}", url);
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
        events::publish_event("[Net] Submitting seed to server...");
        let url = format!("{}&Z={}&P={}&R={}", SERVER_URL, parent, self.name, seed);
        // --- MODIFIED: Call get_server_state and print results here ---
        let result = self.get_server_state(&url);
        if let Ok(state) = &result {
            events::publish_event(&format!("[Net] Difficulty: {}", state.difficulty));
            events::publish_event(&format!("[Net] New parent hash: {}", state.parent_hash));
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
        events::publish_event(&format!(
            "[CPU] Finding seed for parent {} with difficulty {} (using {} threads)...",
            parent,
            difficulty,
            rayon::current_num_threads()
        ));

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
                    events::publish_event(&format!(
                        "+++ [CPU] Found valid seed! (Thread {}) Seed: {} Hash: {} (Bits: {})",
                        rayon::current_thread_index().unwrap_or(0),
                        seed,
                        hash_hex,
                        bits
                    ));

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
    ui_state: Arc<Mutex<UIState>>,
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
                        // Push a stale-parent event into UI state
                        if let Ok(mut ui) = ui_state.lock() {
                            ui.push_event(format!(
                                "Stale parent: server {} != local {}",
                                state.parent_hash, current_parent
                            ));
                            // Show new parent in UI (server value)
                            ui.parent = state.parent_hash.clone();
                        }

                        // 5. Signal the miner to stop
                        if stop_signal
                            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                            .is_ok()
                        {
                            // no console print; UI shows event
                        }
                        break; // Stop this checker thread
                    } else {
                        // parent unchanged
                    }
                }
                Err(_e) => {
                    // Ignore checker errors; UI isn't updated for these
                }
            }
        }
    })
}

/// Spawn a simple TUI that renders username, parent, hash rate and recent events.
fn spawn_tui(ui_state: Arc<Mutex<UIState>>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // Terminal setup
        let mut stdout = stdout();
        enable_raw_mode().ok();
        execute!(stdout, EnterAlternateScreen).ok();
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).unwrap();

        // clear current terminal
        terminal.clear().unwrap();

        // Render loop
        while !APP_EXIT.load(Ordering::Relaxed) {
            // draw
            let ui_clone = ui_state.clone();
            let draw_result = terminal.draw(|f| {
                let size = f.size();
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .margin(1)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Length(3),
                        Constraint::Min(3),
                    ])
                    .split(size);

                // Header: username and parent and rate
                let (username, parent, rate, events, blocks) = if let Ok(ui) = ui_clone.lock() {
                    (
                        ui.username.clone(),
                        ui.parent.clone(),
                        ui.hash_rate,
                        ui.recent_events.clone(),
                        ui.total_blocks_mined,
                    )
                } else {
                    ("-".to_string(), "-".to_string(), 0.0, VecDeque::new(), 0u64)
                };

                let header = Paragraph::new(Spans::from(vec![
                    Span::raw(format!("User: {}    ", username)),
                    Span::raw(format!("Parent: {}    ", parent)),
                    Span::raw(format!("Rate: {}    ", format_hash_rate(rate))),
                    Span::raw(format!("Blocks: {}", blocks)),
                ]))
                .block(Block::default().borders(Borders::ALL).title("Status"));
                f.render_widget(header, chunks[0]);

                let info = Paragraph::new("Recent events (stale parent / block success)")
                    .block(Block::default().borders(Borders::ALL).title("Events"));
                f.render_widget(info, chunks[1]);

                // Events list
                let items: Vec<ListItem> = events
                    .iter()
                    .map(|s| ListItem::new(Spans::from(vec![Span::raw(s.clone())])))
                    .collect();
                let list =
                    List::new(items).block(Block::default().borders(Borders::ALL).title("Recent"));
                f.render_widget(list, chunks[2]);
            });

            if draw_result.is_err() {
                // ignore render errors briefly
            }

            // Poll for key events (200ms)
            if crossterm::event::poll(Duration::from_millis(200)).unwrap_or(false) {
                if let Ok(ev) = crossterm::event::read() {
                    if let CEvent::Key(key) = ev {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Char('Q') => {
                                APP_EXIT.store(true, Ordering::SeqCst);
                                // exit entire process immediately to avoid blocked threads
                                std::process::exit(0);
                            }
                            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                APP_EXIT.store(true, Ordering::SeqCst);
                                std::process::exit(0);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Restore terminal
        disable_raw_mode().ok();
        let mut stdout = std::io::stdout();
        execute!(stdout, LeaveAlternateScreen).ok();
    })
}

fn main() {
    // --- CHOOSE YOUR MINER ---
    const USE_GPU: bool = true;
    // -------------------------

    // (stats thread moved later after UI state is created)

    // 1. Get client name
    let mut name = hostname::get().map_or("RustClient".to_string(), |s| {
        s.to_string_lossy().to_string()
    });

    if let Ok(env_name) = std::env::var("HASH_NAME") {
        if !env_name.is_empty() {
            name = env_name;
        }
    }

    events::publish_event("Starting Rust HashClient...");
    events::publish_event(&format!("Using name: {}", name));

    if !name.ends_with("-B6") {
        if name.len() > 13 {
            name.truncate(13);
        }
        name.push_str("-B6");
    }

    let client = HashClient::new(name);
    // --- ADDED: Debug mode from environment ---
    // If DEBUG_MODE is set to "1" (or any non-empty value), we enable debug.
    // If DEBUG_DIFFICULTY is set, use it as a fixed difficulty for mining.
    let debug_mode = std::env::var("DEBUG_MODE").unwrap_or_default();
    let debug_enabled = !debug_mode.is_empty();
    let debug_difficulty: Option<usize> = std::env::var("DEBUG_DIFFICULTY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    if debug_enabled {
        if let Some(d) = debug_difficulty {
            events::publish_event(&format!(
                "[Debug] Debug mode enabled. Forcing difficulty = {}",
                d
            ));
        } else {
            events::publish_event(
                "[Debug] Debug mode enabled. No DEBUG_DIFFICULTY set; will use server difficulty.",
            );
        }
        events::publish_event("[Debug] Found seeds will NOT be submitted to server.");
    }
    let mut current_parent: String;
    let mut current_difficulty: usize;

    // --- ADDED: Initialize GPU Miner (if selected) ---
    let mut gpu_miner_instance = if USE_GPU {
        println!("[GPU] Initializing OpenCL GPU miner...");
        match gpu_miner::GpuMiner::new() {
            Ok(miner) => {
                events::publish_event("[GPU] GPU Miner initialized successfully!");
                Some(miner)
            }
            Err(e) => {
                events::publish_event(&format!(
                    "[GPU] FATAL: Failed to initialize GPU miner: {}",
                    e
                ));
                events::publish_event("[GPU] Check your OpenCL drivers. Falling back to CPU.");
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
            events::publish_event(&format!("[Net] Difficulty: {}", state.difficulty));
            events::publish_event(&format!("[Net] New parent hash: {}", state.parent_hash));
            current_parent = state.parent_hash;
            current_difficulty = state.difficulty;
        }
        Err(e) => {
            eprintln!("Fatal Error: Could not get initial parent: {}", e);
            return;
        }
    }

    // Create UI state and spawn the TUI and stats threads
    let ui_state = Arc::new(Mutex::new(UIState::new(
        client.name.as_ref().to_string(),
        current_parent.clone(),
    )));

    // Create an event channel and initialize the global event sender
    let (ev_tx, ev_rx) = std::sync::mpsc::channel::<String>();
    events::init_event_sender(ev_tx.clone());

    // Forward events from channel into the UI state
    {
        let ui_clone = ui_state.clone();
        thread::spawn(move || {
            for msg in ev_rx {
                if let Ok(mut ui) = ui_clone.lock() {
                    ui.push_event(msg);
                }
            }
        });
    }

    // Spawn TUI
    let _tui_handle = spawn_tui(ui_state.clone());

    // Spawn hash rate logging thread -> update UI state instead of printing
    {
        let ui_clone = ui_state.clone();
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(5));
            let hashes = TOTAL_HASHES.swap(0, Ordering::SeqCst);
            let rate = (hashes as f64) / 5.0;
            if let Ok(mut ui) = ui_clone.lock() {
                ui.hash_rate = rate;
            }
        });
    }

    // Main game loop
    loop {
        // Exit if TUI requested quit
        if APP_EXIT.load(Ordering::Relaxed) {
            println!("Exiting (requested by UI)");
            break;
        }
        // --- MODIFIED: Main loop logic completely rewritten ---

        // 1. Create a new stop signal for this round
        let stop_signal = Arc::new(AtomicBool::new(false));

        // 2. Spawn the checker thread
        let checker_handle = spawn_checker_thread(
            client.clone(),
            current_parent.clone(),
            stop_signal.clone(),
            ui_state.clone(),
        );

        // Determine effective difficulty for this round (may be overridden in debug)
        let effective_difficulty = if debug_enabled {
            debug_difficulty.unwrap_or(current_difficulty)
        } else {
            current_difficulty
        };

        // 3. Start the miner
        let mining_result = if let Some(miner) = &mut gpu_miner_instance {
            // --- GPU Path ---
            println!(
                "[GPU] Finding seed for parent {} with difficulty {}...",
                current_parent, effective_difficulty
            );
            let base_line = client.get_line(&current_parent, "");

            // --- MODIFIED: Call new GPU function & handle OclResult ---
            match miner.find_seed_gpu(&base_line, effective_difficulty, stop_signal.clone()) {
                Ok(found) => found, // found: Option<(String, String)>
                Err(e) => {
                    events::publish_event(&format!(
                        "[GPU] GPU miner failed: {}. Falling back to CPU for this round.",
                        e
                    ));
                    // Fallback to CPU
                    client.find_seed_cpu(&current_parent, effective_difficulty, stop_signal.clone())
                }
            }
        } else {
            // --- CPU Path (Original) ---
            client.find_seed_cpu(&current_parent, effective_difficulty, stop_signal.clone())
        };

        // 4. Stop the checker thread
        // (It may have already stopped, this is safe)
        stop_signal.store(true, Ordering::SeqCst);
        checker_handle.join().unwrap(); // Wait for it to exit

        // 5. Process the result
        if let Some((seed, our_hash_hex)) = mining_result {
            // --- We found a hash! ---
            if debug_enabled {
                // In debug mode we don't submit to server.
                println!("[Debug] Found seed (not submitting):");
                println!("    Parent: {}", current_parent);
                println!("    Seed: {}", seed);
                println!("    Hash: {}", our_hash_hex);
                // Simulate acceptance locally by updating parent.
                current_parent = our_hash_hex.clone();
                if let Ok(mut ui) = ui_state.lock() {
                    ui.push_event(format!("(debug) Block accepted: {}", current_parent));
                    ui.parent = current_parent.clone();
                }
                if let Some(d) = debug_difficulty {
                    current_difficulty = d;
                }
            } else {
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
                        if new_state.parent_hash == our_hash_hex {
                            if let Ok(mut ui) = ui_state.lock() {
                                ui.push_event(format!("Block accepted: {}", new_state.parent_hash));
                                ui.increment_blocks();
                            }
                        }
                        current_parent = new_state.parent_hash.clone();
                        if let Ok(mut ui) = ui_state.lock() {
                            ui.parent = new_state.parent_hash.clone();
                        }
                        current_difficulty = new_state.difficulty;
                    }
                    Err(e) => {
                        events::publish_event(&format!(
                            "[Net] Error submitting seed: {}. Retrying...",
                            e
                        ));
                        // On error, just try to get the latest parent and continue
                        match client.get_latest_parent() {
                            Ok(state) => {
                                println!("[Net] Difficulty: {}", state.difficulty);
                                println!("[Net] New parent hash: {}", state.parent_hash);
                                current_parent = state.parent_hash.clone();
                                if let Ok(mut ui) = ui_state.lock() {
                                    ui.parent = current_parent.clone();
                                }
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
        } else {
            // --- We were interrupted by the checker ---
            println!("[Main] Miner interrupted (stale parent). Fetching new parent...");
            match client.get_latest_parent() {
                Ok(state) => {
                    events::publish_event(&format!("[Net] Difficulty: {}", state.difficulty));
                    events::publish_event(&format!("[Net] New parent hash: {}", state.parent_hash));
                    current_parent = state.parent_hash;
                    current_difficulty = state.difficulty;
                }
                Err(e) => {
                    events::publish_event(&format!(
                        "Fatal Error: Could not re-sync with server: {}",
                        e
                    ));
                    thread::sleep(Duration::from_secs(10)); // Wait before retrying
                }
            }
        }
    }
}
