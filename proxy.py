import requests
import time
import json
import threading
import queue
import os
import sqlite3
import logging
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
# Use environment variables for flexibility, with sensible defaults
DATA_DIR = os.environ.get('DATA_DIR', '/data')
DB_FILE = os.path.join(DATA_DIR, 'hash_tree.db')
BASE_URL = "http://hash.h10a.de/"
HEADERS = {'User-Agent': 'Python HashGame Analyzer (Bot) v2.0'}

# --- Timing & Concurrency Configuration ---
# NEW: Global rate limit setting
MIN_REQUEST_INTERVAL_SEC = 1.0 # 1 request per second
# How often to poll the main page for new active hashes
POLL_INTERVAL_SEC = 60
# How long a worker sleeps after successfully scraping a node
WORKER_SLEEP_SEC = 0.05 # Reduced: Rate limiting is now global
# How long a worker sleeps after a *failed* scrape
WORKER_RETRY_SLEEP_SEC = 3.0
# Number of parallel scraper workers
NUM_WORKERS = 5
# Request timeout
REQUEST_TIMEOUT_SEC = 10
# Max retries for a single hash
MAX_SCRAPE_RETRIES = 3

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Database Management ---
# We use a single, thread-safe queue for DB writes.
# Reads can happen from any thread, but writes are serialized
# to avoid 'database is locked' errors with SQLite in WAL mode.
# An alternative is one connection per thread, but this is simpler.
db_write_queue = queue.Queue()

def get_db_conn():
    """
    Creates a new database connection.
    Enables Write-Ahead Logging (WAL) for better concurrency.
    """
    try:
        # Ensure the data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn
    except sqlite3.Error as e:
        logging.critical(f"Failed to connect to database at {DB_FILE}: {e}")
        raise

def init_db():
    """Initializes the database tables if they don't exist."""
    logging.info(f"Initializing database at {DB_FILE}...")
    with get_db_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                hash TEXT PRIMARY KEY,
                parent TEXT,
                user TEXT,
                height INTEGER,
                children TEXT,
                scraped INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS active_hashes (
                hash TEXT PRIMARY KEY
            )
        ''')
        # Indexes for faster lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_parent ON nodes (parent);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scraped ON nodes (scraped);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_height ON nodes (height);")
        conn.commit()
    logging.info("Database initialization complete.")

def db_writer_worker():
    """
    A dedicated thread that serializes all database writes.
    This safely handles SQLite's concurrency limitations.
    It receives (query, params) tuples from db_write_queue.
    """
    conn = get_db_conn()
    while True:
        try:
            query, params = db_write_queue.get()
            if query is None:
                logging.info("DB writer thread shutting down.")
                break
            
            conn.execute(query, params)
            conn.commit()
        except sqlite3.Error as e:
            logging.error(f"DB write error: {e}. Query: {query}, Params: {params}")
        except Exception as e:
            logging.error(f"Unexpected error in DB writer: {e}")
        finally:
            db_write_queue.task_done()
    conn.close()

def db_write(query, params=()):
    """Queues a write operation for the DB writer thread."""
    db_write_queue.put((query, params))

# --- Global State (Minimal & Thread-Safe) ---
# Queue for hashes to be fetched
fetch_queue = queue.Queue()
# Set for fast, in-memory check of what's *already* in the queue
# This prevents adding the same hash to the queue thousands of times.
# We still use a lock for this set, as it's shared across threads.
in_queue_set = set()
in_queue_lock = threading.Lock()

# NEW: Rate limiting state
request_lock = threading.Lock()
last_request_time = 0.0

# Use a single session for connection pooling
http_session = requests.Session()
http_session.headers.update(HEADERS)

# --- Flask App ---
app = Flask(__name__, static_folder='.', static_url_path='')

# --- Core Scraping & Logic ---

def get_page_soup(url):
    """
    Fetches a URL and returns a BeautifulSoup object using the global session.
    This function is NOW RATE-LIMITED to 1 request per MIN_REQUEST_INTERVAL_SEC.
    """
    global last_request_time
    
    with request_lock:
        # Calculate how long to sleep
        current_time = time.monotonic()
        time_since_last = current_time - last_request_time
        
        if time_since_last < MIN_REQUEST_INTERVAL_SEC:
            sleep_duration = MIN_REQUEST_INTERVAL_SEC - time_since_last
            logging.debug(f"Rate limiting: sleeping for {sleep_duration:.2f}s")
            time.sleep(sleep_duration)
        
        # Now, make the request
        try:
            response = http_session.get(url, timeout=REQUEST_TIMEOUT_SEC)
            last_request_time = time.monotonic() # Update time *after* request
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            last_request_time = time.monotonic() # Update time even on failure
            logging.warning(f"Error fetching {url}: {e}")
            return None

def scrape_node_data(hash_value):
    """
    Scrapes a single hash page.
    Returns (parent_hash, user_name, children_list, height) or (None, None, None, None) on failure.
    """
    logging.debug(f"Scraping new node: {hash_value}")
    soup = get_page_soup(f"{BASE_URL}?check={hash_value}")
    if not soup:
        return None, None, None, None

    parent_hash, user, height = None, "Unknown", 0
    children_list = []

    try:
        parent_header = soup.find(string='Parent: ')
        if parent_header:
            parent_link = parent_header.find_next_sibling('a')
            if parent_link and parent_link.find('tt'):
                parent_hash = parent_link.find('tt').text.strip()
    except Exception as e:
        logging.warning(f"Error parsing parent for {hash_value}: {e}")

    try:
        main_tt = soup.find('tt', string=hash_value)
        if main_tt:
            info_text = main_tt.next_sibling
            if info_text and ' by ' in info_text:
                user = info_text.split(' by ')[1].split(' at length')[0].strip()
                height_str = info_text.split(' at length ')[1].split(' numbered ')[0].strip()
                height = int(height_str)
    except Exception as e:
        logging.warning(f"Error parsing user/height for {hash_value}: {e}")

    try:
        children_header = soup.find(string='Children:')
        if children_header:
            children_ul = children_header.find_next_sibling('ul')
            if children_ul:
                for li in children_ul.find_all('li'):
                    child_link = li.find('a')
                    child_hash = child_link.find('tt').text.strip()
                    child_user = "Unknown"
                    user_text_node = child_link.next_sibling
                    if user_text_node and ' by ' in user_text_node:
                        child_user = user_text_node.split(' by ')[1].strip()
                    children_list.append({"hash": child_hash, "user": child_user})
    except Exception as e:
        logging.warning(f"Error parsing children for {hash_value}: {e}")

    # Check for a "successful" scrape (at least some data was found)
    if user != "Unknown" or parent_hash or children_list or height > 0:
        return parent_hash, user, children_list, height
    else:
        logging.warning(f"Scrape for {hash_value} yielded no data.")
        return None, None, None, None

def add_to_fetch_queue(hash_value, conn):
    """
    Adds a hash to the fetch queue if it's not already scraped or queued.
    This function performs a DB read, so it needs a connection.
    """
    if not hash_value:
        return

    # 1. Fast in-memory check
    with in_queue_lock:
        if hash_value in in_queue_set:
            return  # Already queued

    # 2. Database check (is it already scraped?)
    try:
        cursor = conn.execute("SELECT scraped FROM nodes WHERE hash = ?", (hash_value,))
        node = cursor.fetchone()
        if node and node['scraped'] == 1:
            return  # Already scraped and saved
    except sqlite3.Error as e:
        logging.error(f"DB error checking node {hash_value}: {e}")
        return

    # 3. Add to queue and tracking set
    with in_queue_lock:
        if hash_value not in in_queue_set: # Double-check after DB read
            fetch_queue.put(hash_value)
            in_queue_set.add(hash_value)
            logging.debug(f"Queued for fetch: {hash_value}")

def process_single_hash(hash_to_fetch):
    """
    Fetches a single hash, updates the DB, and queues its unscraped parent/children.
    This is the core logic for the worker threads.
    """
    conn = get_db_conn()
    try:
        # --- Check if it still needs scraping ---
        cursor = conn.execute("SELECT scraped FROM nodes WHERE hash = ?", (hash_to_fetch,))
        node = cursor.fetchone()
        if node and node['scraped'] == 1:
            logging.debug(f"Node {hash_to_fetch} was already scraped. Skipping.")
            return

        # --- Core Scrape with Retries ---
        parent_hash, user, children_list, height = None, None, None, None
        for i in range(MAX_SCRAPE_RETRIES):
            parent_hash, user, children_list, height = scrape_node_data(hash_to_fetch)
            if user is not None: # `scrape_node_data` returns None on total failure
                break
            logging.warning(f"Failed to scrape {hash_to_fetch}. Retry {i+1}/{MAX_SCRAPE_RETRIES}.")
            time.sleep(WORKER_RETRY_SLEEP_SEC * (i + 1))
        
        if user is None:
            logging.error(f"Failed to scrape node {hash_to_fetch} after {MAX_SCRAPE_RETRIES} retries. Skipping.")
            return

        # --- Queue Parent and Children ---
        add_to_fetch_queue(parent_hash, conn)
        for child_info in children_list:
            add_to_fetch_queue(child_info['hash'], conn)

        # --- Update the Database (via write queue) ---
        
        # 1. Update the node itself (scraped)
        children_json = json.dumps([c['hash'] for c in children_list])
        db_write(
            '''
            INSERT INTO nodes (hash, parent, user, height, children, scraped, last_updated)
            VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(hash) DO UPDATE SET
                parent = excluded.parent,
                user = excluded.user,
                height = excluded.height,
                children = excluded.children,
                scraped = 1,
                last_updated = CURRENT_TIMESTAMP
            ''',
            (hash_to_fetch, parent_hash, user, height, children_json)
        )

        # 2. Create/Update stubs for children
        for child_info in children_list:
            db_write(
                '''
                INSERT INTO nodes (hash, parent, user)
                VALUES (?, ?, ?)
                ON CONFLICT(hash) DO UPDATE SET
                    parent = excluded.parent,
                    user = excluded.user
                WHERE scraped = 0
                ''',
                (child_info['hash'], hash_to_fetch, child_info['user'])
            )
        
        # 3. Create/Update stub for parent
        if parent_hash:
            db_write(
                '''
                INSERT INTO nodes (hash) VALUES (?) ON CONFLICT(hash) DO NOTHING
                ''',
                (parent_hash,)
            )
        
        logging.info(f"Successfully processed and queued node {hash_to_fetch} (Height: {height}). Queue: {fetch_queue.qsize()}")

    finally:
        # Mark as processed (remove from in-queue set)
        with in_queue_lock:
            if hash_to_fetch in in_queue_set:
                in_queue_set.remove(hash_to_fetch)
        # Close the thread-local connection
        conn.close()


def check_for_updates():
    """Checks the main page for new/removed active hashes and updates the DB."""
    logging.info("Checking for new active hashes...")
    soup = get_page_soup(BASE_URL)
    if not soup:
        return

    current_active_hashes = set()
    try:
        active_table = soup.find('h1', string='Active Hashes').find_next_sibling('table')
        for row in active_table.find_all('tr')[1:]:
            hash_link = row.find('a')
            if hash_link and hash_link.find('tt'):
                current_active_hashes.add(hash_link.find('tt').text.strip())
    except Exception as e:
        logging.error(f"Error parsing active hashes table: {e}")
        return

    with get_db_conn() as conn:
        cursor = conn.execute("SELECT hash FROM active_hashes")
        prev_active = {row['hash'] for row in cursor.fetchall()}
        
        new_hashes = current_active_hashes - prev_active
        removed_hashes = prev_active - current_active_hashes

        if new_hashes:
            logging.info(f"Found {len(new_hashes)} new hashes. Adding to queue and DB.")
            for new_hash in new_hashes:
                add_to_fetch_queue(new_hash, conn)
                db_write("INSERT OR IGNORE INTO active_hashes (hash) VALUES (?)", (new_hash,))

        if removed_hashes:
            logging.info(f"Found {len(removed_hashes)} removed hashes. Removing from active list.")
            for old_hash in removed_hashes:
                db_write("DELETE FROM active_hashes WHERE hash = ?", (old_hash,))
    
    if not new_hashes and not removed_hashes:
        logging.info("No changes to active hashes.")

def populate_queue_from_stubs():
    """Scans the DB at startup to populate the queue with unscraped nodes."""
    logging.info("Populating fetch queue from unscraped stubs in DB...")
    with get_db_conn() as conn:
        cursor = conn.execute("SELECT hash FROM nodes WHERE scraped = 0")
        stubs_found = 0
        for row in cursor.fetchall():
            hash_val = row['hash']
            # Use a connection for the DB read inside add_to_fetch_queue
            with get_db_conn() as add_conn:
                add_to_fetch_queue(hash_val, add_conn)
            stubs_found += 1
        
        logging.info(f"Queued {stubs_found} stubs from DB. (Actual queue size: {fetch_queue.qsize()})")

def find_best_b6_node(max_diff=None):
    """
    Searches the DB for the best B6 path, optionally filtering by height.
    This function is ITERATIVE and immune to recursion depth limits.
    It builds an in-memory cache *for this search only* to avoid re-walking paths.
    """
    logging.info("Searching for best '-B6' path...")
    
    # This cache will be populated iteratively for the duration of this call.
    path_counts_cache = {}
    
    # We need a reusable connection
    conn = get_db_conn()
    
    def get_node_from_db(node_hash):
        """Helper to fetch a single node from the DB."""
        if not node_hash:
            return None
        try:
            cursor = conn.execute("SELECT hash, parent, user, height FROM nodes WHERE hash = ?", (node_hash,))
            return cursor.fetchone()
        except sqlite3.Error as e:
            logging.error(f"DB error in get_node_from_db: {e}")
            return None

    try:
        # --- Find Max Height & Filter ---
        max_height = 0
        cursor = conn.execute("SELECT MAX(height) as max_h FROM nodes")
        result = cursor.fetchone()
        if result and result['max_h']:
            max_height = int(result['max_h'])
            
        min_height_filter = 0
        if max_diff is not None:
            try:
                min_height_filter = max_height - int(max_diff)
                logging.info(f"Filtering search: MaxHeight ({max_height}) - MaxDiff ({max_diff}) = MinHeight ({min_height_filter})")
            except Exception:
                pass

        # --- Iterative Path Calculation ---
        # Get *all* nodes that might be part of a path
        cursor = conn.execute("SELECT hash, parent, user, height FROM nodes WHERE scraped = 1 AND height >= ?", (min_height_filter,))
        all_nodes = cursor.fetchall()
        if not all_nodes:
            logging.warning("No nodes found in DB to search for B6 path.")
            return None, 0, 0, 0
        
        logging.info(f"Searching for B6 path among {len(all_nodes)} candidate nodes...")

        max_b6_count = -1
        best_node = None

        for node in all_nodes:
            node_hash = node['hash']
            
            if node_hash in path_counts_cache:
                continue

            # --- Start Iterative Walk ---
            current_hash = node_hash
            path_to_resolve = [] # Stack
            visited_in_path = set() # Cycle detection
            
            current_node_data = node
            
            while current_hash:
                if current_hash in path_counts_cache:
                    break
                
                if current_hash in visited_in_path:
                    logging.warning(f"Cycle detected at node {current_hash}. Breaking loop.")
                    break
                
                visited_in_path.add(current_hash)
                path_to_resolve.append(current_hash)
                
                # Fetch parent data
                parent_hash = current_node_data['parent'] if current_node_data else None
                if not parent_hash:
                    break
                    
                current_node_data = get_node_from_db(parent_hash)
                current_hash = current_node_data['hash'] if current_node_data else None

            # --- Resolve The Stack ---
            parent_score = path_counts_cache.get(current_hash, 0)
            
            for node_to_calc_hash in reversed(path_to_resolve):
                # We fetch again to get user/height, or use the 'node' if it's the start
                if node_to_calc_hash == node['hash']:
                    node_data = node
                else:
                    node_data = get_node_from_db(node_to_calc_hash)
                
                if not node_data:
                    my_b6_score = 0
                else:
                    my_b6_score = 1 if (node_data['user'] or '').endswith('-B6') else 0
                
                total_score = my_b6_score + parent_score
                path_counts_cache[node_to_calc_hash] = total_score
                parent_score = total_score
                
                # Check if this node is the new best
                current_height = node_data['height'] if node_data else 0
                if total_score > max_b6_count and current_height >= min_height_filter:
                    max_b6_count = total_score
                    best_node = node_to_calc_hash

        if best_node:
            best_node_height = get_node_from_db(best_node)['height']
            logging.info(f"🏆 Best Node Found: {best_node} (Height: {best_node_height})")
            logging.info(f"   Path has {max_b6_count} '-B6' users.")
        else:
            logging.info(f"No nodes with '-B6' users found (or none met filter >={min_height_filter}).")
            
        return best_node, max_b6_count, max_height, min_height_filter

    finally:
        conn.close() # Close the connection for this search

# --- REST API Endpoints ---

@app.route('/api/stats', methods=['GET'])
def get_tree_stats():
    """Returns basic stats about the tree from the DB."""
    try:
        with get_db_conn() as conn:
            node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            active_count = conn.execute("SELECT COUNT(*) FROM active_hashes").fetchone()[0]
        queue_size = fetch_queue.qsize()
        return jsonify({
            "total_nodes_stored": node_count,
            "known_active_hashes": active_count,
            "nodes_in_fetch_queue": queue_size,
            "nodes_in_queue_set": len(in_queue_set), # For debugging
            "db_write_queue_size": db_write_queue.qsize() # For debugging
        })
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

@app.route('/api/node/<string:hash_value>', methods=['GET'])
def get_node(hash_value):
    """Gets all known data for a single hash node."""
    try:
        with get_db_conn() as conn:
            node = conn.execute("SELECT * FROM nodes WHERE hash = ?", (hash_value,)).fetchone()
        
        if node:
            # Convert row object to a plain dict for JSON serialization
            return jsonify(dict(node))
        else:
            return jsonify({"error": "Node not found"}), 404
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

@app.route('/api/best_b6_path', methods=['GET'])
def get_best_b6_path():
    """Finds the node with the most '-B6' users in its path."""
    max_diff = request.args.get('max_diff')
    if max_diff is not None:
        try:
            max_diff = int(max_diff)
        except ValueError:
            return jsonify({"error": "max_diff must be an integer"}), 400
    
    # This function is now computationally expensive, but memory-safe.
    # Consider caching its result for a few minutes.
    best_node, count, max_height, min_height = find_best_b6_node(max_diff)
    
    if best_node:
        with get_db_conn() as conn:
            node_height = conn.execute("SELECT height FROM nodes WHERE hash = ?", (best_node,)).fetchone()['height']
        
        return jsonify({
            "best_node": best_node,
            "b6_count": count,
            "node_height": node_height,
            "filter_applied": {
                "max_diff": max_diff,
                "tree_max_height": max_height,
                "min_height_required": min_height
            }
        })
    else:
        return jsonify({"error": "No nodes found or tree is empty"}), 404

@app.route('/api/tree_data', methods=['GET'])
def get_tree_data():
    """
    Returns the entire tree as a flat list, formatted for d3.stratify.
    This is memory-safe as it streams from the DB, but can be slow for clients.
    """
    logging.info("Starting /api/tree_data generation...")
    try:
        with get_db_conn() as conn:
            all_hashes = {row['hash'] for row in conn.execute("SELECT hash FROM nodes")}
            
            nodes_list = [{
                "id": "root", "parentId": "", "user": "ROOT",
                "height": -1, "scraped": True
            }]
            
            # Fetch all nodes. This might be large, but it's just rows,
            # not a giant nested dict.
            all_nodes = conn.execute("SELECT hash, parent, user, height, scraped FROM nodes").fetchall()
            
            for node_data in all_nodes:
                parent_id = node_data["parent"]
                if parent_id is None or parent_id not in all_hashes:
                    parent_id = "root"
                
                nodes_list.append({
                    "id": node_data["hash"],
                    "parentId": parent_id,
                    "user": node_data["user"],
                    "height": node_data["height"],
                    "scraped": bool(node_data["scraped"])
                })
        
        logging.info(f"Returning {len(nodes_list)} nodes for d3.stratify.")
        return jsonify(nodes_list)
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

# --- Background Threads ---

def queue_worker_thread(worker_id):
    """Worker thread that processes the fetch_queue."""
    logging.info(f"Worker-{worker_id} started.")
    while True:
        try:
            hash_to_fetch = fetch_queue.get()
            
            process_single_hash(hash_to_fetch)
            
            fetch_queue.task_done()
            time.sleep(WORKER_SLEEP_SEC)
        
        except Exception as e:
            logging.error(f"Error in Worker-{worker_id}: {e}", exc_info=True)
            try:
                fetch_queue.task_done()
            except ValueError:
                pass # Already done

def poller_thread():
    """Main scraper loop (poller) to check for active hashes."""
    logging.info("Poller thread started.")
    while True:
        try:
            if fetch_queue.empty() and db_write_queue.empty():
                check_for_updates()
            else:
                logging.info(f"Skipping poll. FetchQueue: {fetch_queue.qsize()}, WriteQueue: {db_write_queue.qsize()}")

            # Log best path periodically
            find_best_b6_node(max_diff=None)
            
            logging.info(f"Poller waiting {POLL_INTERVAL_SEC}s. FetchQueue: {fetch_queue.qsize()}, WriteQueue: {db_write_queue.qsize()}")
            time.sleep(POLL_INTERVAL_SEC)
        except Exception as e:
            logging.error(f"Error in poller_thread: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL_SEC) # Don't crash loop

# --- Main Execution ---

def main():
    """Main function to load data and start threads."""
    logging.info("Starting Hash Scraper v2.0...")
    init_db()
    
    # Start the dedicated DB writer thread
    threading.Thread(target=db_writer_worker, daemon=True, name="DBWriter").start()

    # Populate queue from any unscraped nodes in the DB
    populate_queue_from_stubs()
    
    # Start the parallel queue worker threads
    for i in range(NUM_WORKERS):
        threading.Thread(
            target=queue_worker_thread,
            daemon=True,
            name=f"Worker-{i+1}",
            args=(i+1,)
        ).start()

    # Start the main page poller thread
    threading.Thread(target=poller_thread, daemon=True, name="Poller").start()
    
    logging.info("\n========================================================")
    logging.info("Starting Flask API server on http://0.0.0.0:5000/")
    logging.info("View the visualization at: http://127.0.0.1:5000/")
    logging.info("========================================================\n")
    
    # Run the Flask app in the main thread
    # Turn off reloader to prevent threads from being started twice
    app.run(host='0.0.0.0', port=5000, use_reloader=False, threaded=True)

if __name__ == "__main__":
    main()
