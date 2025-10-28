use rand::{thread_rng, Rng};
// rayon used for thread count; we don't need ParallelIterator here
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
use crossterm::event::{Event as CEvent, KeyCode, KeyModifiers, MouseButton, MouseEventKind};
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
    // Which device is being used (e.g. "CPU" or "GPU: GeForce ...")
    device: String,
    // When true, show the device selection menu overlay
    show_device_menu: bool,
    // Index of currently highlighted item in the device menu
    device_menu_index: usize,
    // Available device choices displayed in the menu
    available_devices: Vec<String>,
    // Last rendered device card rectangle (x, y, width, height) for mouse clicks
    device_rect: Option<(u16, u16, u16, u16)>,
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
    fn new(username: String, parent: String, device: String) -> Self {
        let total_blocks_mined = load_blocks_mined_from_disk();
        UIState {
            recent_events: VecDeque::with_capacity(32),
            hash_rate: 0.0,
            username,
            parent,
            device,
            show_device_menu: false,
            device_menu_index: 0,
            available_devices: Vec::new(),
            device_rect: None,
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

        // Capture client name for composing the line: "parent name seed"
        let client_name = self.name.as_ref().to_string();

        // Spawn worker threads (std::thread) so we can check the shared stop
        // signal frequently and respond to aborts when switching devices.
        let num_threads = rayon::current_num_threads();
        let mut handles = Vec::with_capacity(num_threads);

        for _ in 0..num_threads {
            let parent = parent.to_string();
            let cname = client_name.clone();
            let stop = stop_signal.clone();
            let result_clone = result.clone();
            handles.push(thread::spawn(move || {
                // Each thread uses its own RNG
                let mut rng = thread_rng();
                const BATCH: usize = 256; // check stop signal every BATCH hashes
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }

                    for _ in 0..BATCH {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let seed = format!("{:x}", rng.gen::<u64>());
                        let hash_bytes = {
                            // compute the hash for this seed: "parent name seed"
                            let mut hasher = Sha256::new();
                            let line = format!("{} {} {}", parent, cname, seed);
                            hasher.update(line.as_bytes());
                            hasher.finalize().to_vec()
                        };
                        let bits = count_zero_bits(&hash_bytes);

                        TOTAL_HASHES.fetch_add(1, Ordering::Relaxed);

                        if bits >= difficulty {
                            // Try to claim victory by setting the stop flag.
                            if stop
                                .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                                .is_ok()
                            {
                                let hash_hex = hex::encode(&hash_bytes);
                                events::publish_event(&format!(
                                    "+++ [CPU] Found valid seed! Seed: {} Hash: {} (Bits: {})",
                                    seed, hash_hex, bits
                                ));
                                let mut guard = result_clone.lock().unwrap();
                                *guard = Some((seed, hash_hex));
                                break;
                            } else {
                                // Another thread or the checker claimed it
                                break;
                            }
                        }
                    }

                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }));
        }

        // Wait for workers to finish
        for h in handles {
            let _ = h.join();
        }

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
fn spawn_tui(
    ui_state: Arc<Mutex<UIState>>,
    shared_stop_signal: Arc<Mutex<Arc<AtomicBool>>>,
) -> thread::JoinHandle<()> {
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
                    .constraints([Constraint::Length(3), Constraint::Min(3)])
                    .split(size);

                // Header: username and parent and rate
                // Lock UIState (mutable) so we can also record the device card rectangle
                let (username, parent, rate, device, events, blocks) =
                    if let Ok(ui) = ui_clone.lock() {
                        (
                            ui.username.clone(),
                            ui.parent.clone(),
                            ui.hash_rate,
                            ui.device.clone(),
                            ui.recent_events.clone(),
                            ui.total_blocks_mined,
                        )
                    } else {
                        (
                            "-".to_string(),
                            "-".to_string(),
                            0.0,
                            "-".to_string(),
                            VecDeque::new(),
                            0u64,
                        )
                    };

                let top_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(18), // User
                        Constraint::Percentage(40), // Parent (wider)
                        Constraint::Percentage(16), // Rate
                        Constraint::Percentage(12), // Blocks
                        Constraint::Percentage(14), // Device
                    ])
                    .split(chunks[0]);

                let user_card =
                    Paragraph::new(Spans::from(vec![Span::raw(format!("{}", username))]))
                        .block(Block::default().borders(Borders::ALL).title("User"));
                f.render_widget(user_card, top_chunks[0]);

                let parent_card =
                    Paragraph::new(Spans::from(vec![Span::raw(format!("{}", parent))]))
                        .block(Block::default().borders(Borders::ALL).title("Parent"));
                f.render_widget(parent_card, top_chunks[1]);

                let rate_card = Paragraph::new(Spans::from(vec![Span::raw(format!(
                    "{}",
                    format_hash_rate(rate)
                ))]))
                .block(Block::default().borders(Borders::ALL).title("Rate"));
                f.render_widget(rate_card, top_chunks[2]);

                let blocks_card =
                    Paragraph::new(Spans::from(vec![Span::raw(format!("{}", blocks))]))
                        .block(Block::default().borders(Borders::ALL).title("Blocks"));
                f.render_widget(blocks_card, top_chunks[3]);

                let device_card =
                    Paragraph::new(Spans::from(vec![Span::raw(format!("{}", device))]))
                        .block(Block::default().borders(Borders::ALL).title("Device"));
                f.render_widget(device_card, top_chunks[4]);

                // Save the device card rectangle for mouse click detection
                if let Ok(mut ui) = ui_clone.lock() {
                    let rect = top_chunks[4];
                    ui.device_rect = Some((rect.x, rect.y, rect.width, rect.height));
                }

                // If device menu is active, render a centered popup with choices
                if let Ok(ui) = ui_clone.lock() {
                    if ui.show_device_menu {
                        let popup_width = 40u16.min(size.width.saturating_sub(4));
                        let popup_height =
                            (ui.available_devices.len() as u16).saturating_add(2).max(3);
                        let popup_x = (size.width.saturating_sub(popup_width)) / 2;
                        let popup_y = (size.height.saturating_sub(popup_height)) / 2;

                        let mut lines: Vec<Spans> = Vec::new();
                        for (i, d) in ui.available_devices.iter().enumerate() {
                            if i == ui.device_menu_index {
                                lines.push(Spans::from(vec![Span::raw(format!("> {}", d))]));
                            } else {
                                lines.push(Spans::from(vec![Span::raw(format!("  {}", d))]));
                            }
                        }

                        let popup = Paragraph::new(lines).block(
                            Block::default()
                                .borders(Borders::ALL)
                                .title("Select Device"),
                        );

                        use tui::layout::Rect;
                        let rect = Rect::new(popup_x, popup_y, popup_width, popup_height);
                        f.render_widget(popup, rect);
                    }
                }

                // Events list
                let items: Vec<ListItem> = events
                    .iter()
                    .map(|s| ListItem::new(Spans::from(vec![Span::raw(s.clone())])))
                    .collect();
                let list =
                    List::new(items).block(Block::default().borders(Borders::ALL).title("Recent"));
                f.render_widget(list, chunks[1]);
            });

            if draw_result.is_err() {
                // ignore render errors briefly
            }

            // Poll for key events (200ms)
            if crossterm::event::poll(Duration::from_millis(200)).unwrap_or(false) {
                if let Ok(ev) = crossterm::event::read() {
                    match ev {
                        CEvent::Key(key) => {
                            // If the device menu is open, handle navigation and selection
                            if let Ok(mut ui) = ui_clone.lock() {
                                if ui.show_device_menu {
                                    match key.code {
                                        KeyCode::Down => {
                                            if !ui.available_devices.is_empty() {
                                                ui.device_menu_index = (ui.device_menu_index + 1)
                                                    % ui.available_devices.len();
                                            }
                                        }
                                        KeyCode::Up => {
                                            if !ui.available_devices.is_empty() {
                                                if ui.device_menu_index == 0 {
                                                    ui.device_menu_index =
                                                        ui.available_devices.len() - 1;
                                                } else {
                                                    ui.device_menu_index -= 1;
                                                }
                                            }
                                        }
                                        KeyCode::Enter => {
                                            if let Some(choice) =
                                                ui.available_devices.get(ui.device_menu_index)
                                            {
                                                ui.device = choice.clone();
                                                ui.show_device_menu = false;
                                                events::publish_event(&format!(
                                                    "Device selected: {}",
                                                    ui.device
                                                ));
                                                // Signal the current miner to stop so main will restart with new device
                                                if let Ok(shared) = shared_stop_signal.lock() {
                                                    shared.store(true, Ordering::SeqCst);
                                                }
                                            }
                                        }
                                        KeyCode::Esc => {
                                            ui.show_device_menu = false;
                                        }
                                        _ => {}
                                    }
                                    continue; // menu handled
                                }
                            }

                            // Global key bindings
                            match key.code {
                                KeyCode::Char('q') | KeyCode::Char('Q') => {
                                    APP_EXIT.store(true, Ordering::SeqCst);
                                    // exit entire process immediately to avoid blocked threads
                                    std::process::exit(0);
                                }
                                KeyCode::Char('c')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    APP_EXIT.store(true, Ordering::SeqCst);
                                    std::process::exit(0);
                                }
                                KeyCode::Char('d') => {
                                    if let Ok(mut ui) = ui_clone.lock() {
                                        ui.show_device_menu = true;
                                        ui.device_menu_index = ui
                                            .available_devices
                                            .iter()
                                            .position(|d| *d == ui.device)
                                            .unwrap_or(0);
                                    }
                                }
                                _ => {}
                            }
                        }
                        CEvent::Mouse(me) => {
                            // Left-click on device card opens the device menu
                            if matches!(me.kind, MouseEventKind::Down(MouseButton::Left)) {
                                if let Ok(mut ui) = ui_clone.lock() {
                                    if let Some((x, y, w, h)) = ui.device_rect {
                                        let col = me.column as u16;
                                        let row = me.row as u16;
                                        if col >= x && col < x + w && row >= y && row < y + h {
                                            ui.show_device_menu = true;
                                            ui.device_menu_index = ui
                                                .available_devices
                                                .iter()
                                                .position(|d| *d == ui.device)
                                                .unwrap_or(0);
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
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
        events::publish_event("[GPU] Initializing OpenCL GPU miner...");
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
            events::publish_event(&format!("Fatal Error: Could not get initial parent: {}", e));
            return;
        }
    }

    // Create UI state and spawn the TUI and stats threads
    // Determine selected device string for UI
    let selected_device = if let Some(miner) = &gpu_miner_instance {
        format!("GPU: {}", miner.device_name)
    } else {
        "CPU".to_string()
    };

    let ui_state = Arc::new(Mutex::new(UIState::new(
        client.name.as_ref().to_string(),
        current_parent.clone(),
        selected_device.clone(),
    )));

    // Populate available devices for the device menu
    if let Ok(mut ui) = ui_state.lock() {
        ui.available_devices.clear();
        ui.available_devices.push("CPU".to_string());
        if let Some(miner) = &gpu_miner_instance {
            ui.available_devices
                .push(format!("GPU: {}", miner.device_name));
        }
        // Make sure current device is one of the available ones
        if !ui.available_devices.iter().any(|d| d == &ui.device) {
            ui.device = ui.available_devices[0].clone();
        }
    }

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

    // Create a shared holder for the current round's stop-signal so the TUI
    // can request the running miner to stop when the user switches devices.
    let shared_stop_signal: Arc<Mutex<Arc<AtomicBool>>> =
        Arc::new(Mutex::new(Arc::new(AtomicBool::new(false))));

    // Spawn TUI (give it access to the shared stop-signal)
    let _tui_handle = spawn_tui(ui_state.clone(), shared_stop_signal.clone());

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
    // Track the last-seen device to detect user switches
    let mut previous_device = selected_device.clone();

    loop {
        // Exit if TUI requested quit
        if APP_EXIT.load(Ordering::Relaxed) {
            events::publish_event("Exiting (requested by UI)");
            break;
        }
        // --- MODIFIED: Main loop logic completely rewritten ---

        // 1. Create a new stop signal for this round
        let stop_signal = Arc::new(AtomicBool::new(false));

        // Publish this stop signal into the shared holder so the TUI can set it
        // to request the miner be interrupted (e.g. when the user changes device).
        if let Ok(mut holder) = shared_stop_signal.lock() {
            *holder = stop_signal.clone();
        }

        // 2. Spawn the checker thread
        let checker_handle = spawn_checker_thread(
            client.clone(),
            current_parent.clone(),
            stop_signal.clone(),
            ui_state.clone(),
        );

        // Honor device selection from the UI: switch GPU on/off as requested
        let ui_device_choice = if let Ok(ui) = ui_state.lock() {
            ui.device.clone()
        } else {
            "CPU".to_string()
        };

        // If the user changed device since last round, reset counters and notify
        if ui_device_choice != previous_device {
            events::publish_event(&format!(
                "Switching device: {} -> {}",
                previous_device, ui_device_choice
            ));
            // Reset total hashes so the displayed hash-rate quickly reflects the new device
            TOTAL_HASHES.store(0, Ordering::SeqCst);
            if let Ok(mut ui) = ui_state.lock() {
                ui.hash_rate = 0.0;
            }
            previous_device = ui_device_choice.clone();
        }

        if ui_device_choice.starts_with("GPU") {
            // Ensure we have a GPU miner instance; try to initialize on demand
            if gpu_miner_instance.is_none() {
                events::publish_event("[Main] User requested GPU — attempting to initialize...");
                match gpu_miner::GpuMiner::new() {
                    Ok(miner) => {
                        events::publish_event("[GPU] GPU miner initialized on-demand.");
                        let devname = miner.device_name.clone();
                        gpu_miner_instance = Some(miner);
                        // Update UI available devices and current device string
                        if let Ok(mut ui) = ui_state.lock() {
                            ui.available_devices.retain(|d| d != "GPU: Unknown");
                            if !ui.available_devices.iter().any(|d| d.starts_with("GPU:")) {
                                ui.available_devices.push(format!("GPU: {}", devname));
                            }
                            ui.device = format!("GPU: {}", devname);
                        }
                    }
                    Err(e) => {
                        events::publish_event(&format!(
                            "[GPU] Failed to initialize GPU on-demand: {}. Staying on CPU.",
                            e
                        ));
                        // Remove GPU option from available devices
                        if let Ok(mut ui) = ui_state.lock() {
                            ui.available_devices.retain(|d| !d.starts_with("GPU:"));
                            ui.device = "CPU".to_string();
                        }
                    }
                }
            }
        } else {
            // User requested CPU — drop GPU instance if present
            if gpu_miner_instance.is_some() {
                events::publish_event("[Main] Switching to CPU per UI selection.");
                gpu_miner_instance = None;
            }
        }

        // Determine effective difficulty for this round (may be overridden in debug)
        let effective_difficulty = if debug_enabled {
            debug_difficulty.unwrap_or(current_difficulty)
        } else {
            current_difficulty
        };

        // 3. Start the miner
        let mining_result = if let Some(miner) = &mut gpu_miner_instance {
            // --- GPU Path ---
            events::publish_event(&format!(
                "[GPU] Finding seed for parent {} with difficulty {}...",
                current_parent, effective_difficulty,
            ));
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
                events::publish_event("[Debug] Found seed (not submitting):");
                events::publish_event(&format!("    Parent: {}", current_parent));
                events::publish_event(&format!("    Seed: {}", seed));
                events::publish_event(&format!("    Hash: {}", our_hash_hex));
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
                            events::publish_event(&format!(
                                "[Net] Server accepted our hash! New parent is: {}",
                                new_state.parent_hash
                            ));
                        } else {
                            events::publish_event(&format!(
                                "[Net] !!! Server did NOT accept our hash (someone was faster). !!!"
                            ));
                            events::publish_event(&format!("    Our hash:   {}", our_hash_hex));
                            events::publish_event(&format!(
                                "    New parent: {}",
                                new_state.parent_hash
                            ));
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
                                events::publish_event(&format!(
                                    "[Net] Difficulty: {}",
                                    state.difficulty
                                ));
                                events::publish_event(&format!(
                                    "[Net] New parent hash: {}",
                                    state.parent_hash
                                ));
                                current_parent = state.parent_hash.clone();
                                if let Ok(mut ui) = ui_state.lock() {
                                    ui.parent = current_parent.clone();
                                }
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
        } else {
            // --- We were interrupted by the checker ---
            events::publish_event(
                "[Main] Miner interrupted (stale parent). Fetching new parent...",
            );
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
