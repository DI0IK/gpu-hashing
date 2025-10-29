import requests
import time
import json
from bs4 import BeautifulSoup
import threading
import queue
from flask import Flask, jsonify, request
import os

# --- Configuration ---
BASE_URL = "http://hash.h10a.de/"
HEADERS = {'User-Agent': 'Python HashGame Analyzer (Bot)'}
# Store JSON data under a single data directory (can be mounted into the container)
DATA_DIR = '/data'
TREE_FILE = os.path.join(DATA_DIR, 'hash_tree.json')
ACTIVE_HASHES_FILE = os.path.join(DATA_DIR, 'active_hashes.json')
CHECK_INTERVAL = 5  # Seconds
WORKER_SLEEP = 5.0   # Seconds to sleep between fetches in the worker

# --- Global Shared State & Lock ---
shared_tree = {}
shared_active_hashes = set()
fetch_queue = queue.Queue() # <-- New queue
in_queue_set = set()      # <-- New set to track items in queue
data_lock = threading.Lock()

# --- Flask App Initialization ---
app = Flask(__name__, static_folder='.', static_url_path='')

# --- Data Persistence (Uses Global State) ---

def load_data():
    """Loads the tree and active hashes from disk into global state."""
    global shared_tree, shared_active_hashes
    try:
        with open(TREE_FILE, 'r') as f:
            shared_tree = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        shared_tree = {}
    try:
        with open(ACTIVE_HASHES_FILE, 'r') as f:
            shared_active_hashes = set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        shared_active_hashes = set()

def save_data():
    """Saves the global tree and active hashes to disk."""
    with data_lock:
        tree_copy = shared_tree.copy()
        active_hashes_copy = list(shared_active_hashes)
    
    try:
        with open(TREE_FILE, 'w') as f:
            json.dump(tree_copy, f, indent=2)
        with open(ACTIVE_HASHES_FILE, 'w') as f:
            json.dump(active_hashes_copy, f, indent=2)
    except IOError as e:
        print(f"Error saving data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during save: {e}")

# --- Web Scraping Functions ---

def get_page_soup(url):
    """Fetches a URL and returns a BeautifulSoup object."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_node_data(hash_value):
    """
    Scrapes a single hash page for its parent, user, children, and height.
    Returns (parent_hash, user_name, children_list, height)
    """
    print(f"Scraping new node: {hash_value}")
    soup = get_page_soup(f"{BASE_URL}?check={hash_value}")
    if not soup:
        return None, "Unknown", [], 0

    parent_hash = None
    user = "Unknown"
    children_list = []
    height = 0

    parent_header = soup.find(string='Parent: ')
    if parent_header:
        parent_link = parent_header.find_next_sibling('a')
        if parent_link and parent_link.find('tt'):
            parent_hash = parent_link.find('tt').text.strip()

    main_tt = soup.find('tt', string=hash_value)
    if main_tt:
        info_text = main_tt.next_sibling
        if info_text and ' by ' in info_text:
            try:
                user = info_text.split(' by ')[1].split(' at length')[0].strip()
                height_str = info_text.split(' at length ')[1].split(' numbered ')[0].strip()
                height = int(height_str)
            except Exception as e:
                print(f"Could not parse user or height from info text: {e}")
                
    children_header = soup.find(string='Children:')
    if children_header:
        children_ul = children_header.find_next_sibling('ul')
        if children_ul:
            for li in children_ul.find_all('li'):
                try:
                    child_link = li.find('a')
                    child_hash = child_link.find('tt').text.strip()
                    child_user = "Unknown"
                    user_text_node = child_link.next_sibling
                    if user_text_node and ' by ' in user_text_node:
                        child_user = user_text_node.split(' by ')[1].strip()
                    children_list.append({"hash": child_hash, "user": child_user})
                except Exception as e:
                    print(f"Could not parse child list item: {e}")

    return parent_hash, user, children_list, height

# --- NEW: Queue Management ---

def add_to_fetch_queue(hash_value):
    """Adds a hash to the fetch queue if it's not already scraped or queued."""
    if not hash_value:
        return

    with data_lock:
        # Check if already scraped
        node_data = shared_tree.get(hash_value)
        if node_data and node_data.get('scraped', False):
            return  # Already done

        # Check if already in queue
        if hash_value in in_queue_set:
            return  # Already queued
        
        # Add to queue and tracking set
        fetch_queue.put(hash_value)
        in_queue_set.add(hash_value)
        # print(f"Queued for fetch: {hash_value}") # (Optional: can be very noisy)

# --- Scraper Logic (MODIFIED) ---

def process_single_hash(hash_to_fetch):
    """
    Fetches a single hash, updates the tree, and queues its
    unscraped parent/children. This function does NOT loop.
    """
    global shared_tree
    
    with data_lock:
        needs_scrape = (hash_to_fetch not in shared_tree or 
                        not shared_tree[hash_to_fetch].get("scraped", False))
    
    if not needs_scrape:
        # print(f"Node {hash_to_fetch} was already scraped. Skipping.")
        with data_lock:
            if hash_to_fetch in in_queue_set:
                in_queue_set.remove(hash_to_fetch)
        return

    # --- This is the core scrape ---
    parent_hash, user, children_list, height = scrape_node_data(hash_to_fetch)
    
    if user == "Unknown" and parent_hash is None and not children_list and height == 0:
        print(f"Failed to scrape {hash_to_fetch}. Will not be retried unless re-queued.")
        with data_lock:
            if hash_to_fetch in in_queue_set:
                in_queue_set.remove(hash_to_fetch)
        return
        
    # --- Queue parent and children ---
    add_to_fetch_queue(parent_hash)
    for child_info in children_list:
        add_to_fetch_queue(child_info['hash'])

    # --- Update the tree (with lock) ---
    with data_lock:
        shared_tree.setdefault(hash_to_fetch, {})
        shared_tree[hash_to_fetch].update({
            "parent": parent_hash,
            "user": user,
            "height": height,
            "children": [c['hash'] for c in children_list],
            "scraped": True
        })
        
        # Update children stubs
        for child_info in children_list:
            shared_tree.setdefault(child_info['hash'], {})
            shared_tree[child_info['hash']].update({
                "parent": hash_to_fetch,
                "user": child_info['user']
            })
        
        # Update parent stub
        if parent_hash:
            shared_tree.setdefault(parent_hash, {})
            if "children" not in shared_tree[parent_hash]:
                shared_tree[parent_hash]["children"] = []
            if hash_to_fetch not in shared_tree[parent_hash]["children"]:
                shared_tree[parent_hash]["children"].append(hash_to_fetch)
        
        # Mark as processed
        if hash_to_fetch in in_queue_set:
            in_queue_set.remove(hash_to_fetch)
    
    save_data() # Save after *every* successful scrape
    print(f"Persisted node {hash_to_fetch} (Height: {height}). Queue: {fetch_queue.qsize()}")

def check_for_updates():
    """Checks the main page for new hashes and adds them to the fetch queue."""
    global shared_active_hashes
    print("Checking for new active hashes...")
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
        print(f"Error parsing active hashes table: {e}")
        return

    with data_lock:
        prev_active = set(shared_active_hashes)
        new_hashes = current_active_hashes - prev_active
        removed_hashes = prev_active - current_active_hashes

    if new_hashes:
        print(f"Found {len(new_hashes)} new hashes. Adding to queue.")
        for new_hash in new_hashes:
            add_to_fetch_queue(new_hash)

    if removed_hashes:
        print(f"Found {len(removed_hashes)} removed/inactive hashes. Removing from active list.")

    if new_hashes or removed_hashes:
        with data_lock:
            # Add new active hashes
            if new_hashes:
                shared_active_hashes.update(new_hashes)
            # Remove hashes that are no longer active
            if removed_hashes:
                shared_active_hashes.difference_update(removed_hashes)

        save_data() # Save the updated active_hashes list
    else:
        print("No changes to active hashes.")
        
def populate_queue_from_stubs():
    """
    Scans the loaded tree *once* at startup to populate
    the fetch queue with all unscraped nodes.
    """
    global shared_tree
    
    # Take a snapshot once (no lock held while we iterate and call add_to_fetch_queue).
    with data_lock:
        tree_snapshot = shared_tree.copy()

    print(f"Scanning {len(tree_snapshot)} loaded nodes for missing stubs...")

    stubs_found = 0

    # We use the local tree_snapshot for lookups so we don't need to acquire
    # data_lock while calling add_to_fetch_queue (which itself acquires the lock).
    for node_hash, node_data in tree_snapshot.items():
        # 1. Check self (if this node is just a stub)
        if not node_data.get('scraped', False):
            add_to_fetch_queue(node_hash)
            stubs_found += 1
            continue  # No need to check its children if it's not scraped

        # 2. Check parent using the snapshot
        parent_hash = node_data.get('parent')
        if parent_hash:
            parent_node = tree_snapshot.get(parent_hash)
            if not parent_node or not parent_node.get('scraped', False):
                add_to_fetch_queue(parent_hash)
                stubs_found += 1

        # 3. Check children using the snapshot
        for child_hash in node_data.get('children', []):
            child_node = tree_snapshot.get(child_hash)
            if not child_node or not child_node.get('scraped', False):
                add_to_fetch_queue(child_hash)
                stubs_found += 1

    print(f"Found and queued {stubs_found} stubs from loaded data. (Actual queue size: {fetch_queue.qsize()})")

# --- Search Function (Iterative, No Recursion) ---
# (This function is unchanged)
def find_best_b6_node(tree_snapshot, max_diff=None):
    """
    Searches a tree snapshot for the best B6 path, optionally
    filtering by height (max_diff).
    This function is ITERATIVE and immune to recursion depth limits.
    """
    print("Searching for best '-B6' path...")
    if not tree_snapshot:
        return None, 0, 0, 0

    max_b6_count = -1
    best_node = None
    
    # This cache will be populated iteratively.
    path_counts_cache = {}
    
    # --- Find Max Height & Filter (No change) ---
    max_height = 0
    for node_data in tree_snapshot.values():
        height = node_data.get('height', 0)
        if height > max_height:
            max_height = height
            
    min_height_filter = 0
    if max_diff is not None:
        try:
            min_height_filter = max_height - int(max_diff)
            print(f"Filtering search: MaxHeight ({max_height}) - MaxDiff ({max_diff}) = MinHeight ({min_height_filter})")
        except Exception:
            pass 

    # --- Iterative Path Calculation ---
    # We iterate through all nodes to ensure the cache is built
    # for every single one.
    for node_hash in tree_snapshot.keys():
        
        # If we've already calculated this path (as part of another's)
        if node_hash in path_counts_cache:
            continue

        # --- Start Iterative Walk ---
        current_hash = node_hash
        path_to_resolve = [] # This will act as our "call stack"
        visited_in_path = set() # For cycle detection
        
        while current_hash and current_hash in tree_snapshot:
            # 1. We hit a sub-path we've already computed
            if current_hash in path_counts_cache:
                break
            
            # 2. We found a cycle in this path
            if current_hash in visited_in_path:
                print(f"Warning: Cycle detected at node {current_hash}. Breaking loop.")
                break
            
            # 3. Add to stack and continue up
            visited_in_path.add(current_hash)
            path_to_resolve.append(current_hash)
            current_hash = tree_snapshot[current_hash].get('parent')
        
        # --- Resolve The Stack ---
        # Get the score of the path *below* our stack
        # (This is 0 if we hit the root, or the cached score)
        parent_score = path_counts_cache.get(current_hash, 0)
        
        # Now, resolve the stack in reverse order, populating the cache
        for node_to_calc in reversed(path_to_resolve):
            node_data = tree_snapshot[node_to_calc]
            my_b6_score = 1 if node_data.get('user', '').endswith('-B6') else 0
            
            total_score = my_b6_score + parent_score
            
            path_counts_cache[node_to_calc] = total_score
            parent_score = total_score # This node's score is the "parent" for the next one down

    # --- End of all cache calculation ---

    # --- Find Best Node (using the now-full cache) ---
    for node_hash, node_data in tree_snapshot.items():
        count = path_counts_cache.get(node_hash, 0)
        current_height = node_data.get('height', 0)
        
        if count > max_b6_count and current_height >= min_height_filter:
            max_b6_count = count
            best_node = node_hash
            
    if best_node:
        print(f"🏆 Best Node Found: {best_node}")
        print(f"   Path to root has {max_b6_count} '-B6' users.")
        print(f"   Node height is {tree_snapshot[best_node].get('height', 0)} (Filter was >={min_height_filter})")
    else:
        print(f"No nodes with '-B6' users found (or none met height filter >={min_height_filter}).")
        
    return best_node, max_b6_count, max_height, min_height_filter

# --- REST API Endpoints ---

@app.route('/api/stats', methods=['GET'])
def get_tree_stats():
    """Returns basic stats about the tree."""
    with data_lock:
        node_count = len(shared_tree)
        active_count = len(shared_active_hashes)
        queue_size = fetch_queue.qsize()
    return jsonify({
        "total_nodes_stored": node_count,
        "known_active_hashes": active_count,
        "nodes_in_fetch_queue": queue_size
    })

@app.route('/api/node/<string:hash_value>', methods=['GET'])
def get_node(hash_value):
    """Gets all known data for a single hash node."""
    with data_lock:
        node_data = shared_tree.get(hash_value, {}).copy()
    
    if node_data:
        return jsonify(node_data)
    else:
        return jsonify({"error": "Node not found"}), 404

@app.route('/api/best_b6_path', methods=['GET'])
def get_best_b6_path():
    """Finds the node with the most '-B6' users in its path."""
    max_diff = request.args.get('max_diff')
    
    if max_diff is not None:
        try:
            max_diff = int(max_diff)
        except ValueError:
            return jsonify({"error": "max_diff must be an integer"}), 400

    with data_lock:
        tree_copy = shared_tree.copy()
    
    best_node, count, max_height, min_height = find_best_b6_node(tree_copy, max_diff) 
    
    if best_node:
        return jsonify({
            "best_node": best_node,
            "b6_count": count,
            "node_height": tree_copy[best_node].get('height', 0),
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
    Includes a virtual root node to tie all branches together.
    """
    with data_lock:
        tree_copy = shared_tree.copy()
    
    nodes_list = []
    all_hashes = set(tree_copy.keys())
    
    nodes_list.append({
        "id": "root", 
        "parentId": "", 
        "user": "ROOT", 
        "height": -1, 
        "scraped": True
    })
    
    for hash_val, node_data in tree_copy.items():
        parent_id = node_data.get("parent")
        
        if parent_id is None or parent_id not in all_hashes:
            parent_id = "root"
            
        nodes_list.append({
            "id": hash_val,
            "parentId": parent_id,
            "user": node_data.get("user", "Unknown"),
            "height": node_data.get("height", 0),
            "scraped": node_data.get("scraped", False)
        })
        
    return jsonify(nodes_list)

# --- Main Execution (MODIFIED) ---

def queue_worker():
    """Worker thread that processes the fetch_queue."""
    print("Queue worker thread started.")
    while True:
        try:
            # Get a hash to fetch. This will block until a hash is available.
            hash_to_fetch = fetch_queue.get()
            
            # process_single_hash does its own check, so we
            # can just call it.
            process_single_hash(hash_to_fetch)
            
            # Signal that this item is complete
            fetch_queue.task_done()
            
            # Sleep to be nice to the server
            time.sleep(WORKER_SLEEP) 
        
        except Exception as e:
            print(f"Error in queue_worker: {e}")
            try:
                # Ensure task_done is called even on error
                fetch_queue.task_done()
            except ValueError:
                pass # Already done

def scraper_loop():
    """Main scraper loop (poller) to run in a background thread."""
    print("Scraper thread (poller) started.")
    
    # --- Main Polling Loop ---
    while True:
        try:
            # 1. Check for new hashes from the main page
            print("--- Checking for new active hashes ---")
            check_for_updates() # This now adds to the queue
            
            # 2. Log best path (still useful to see)
            with data_lock:
                tree_copy = shared_tree.copy()
            best_node, count, max_h, min_h = find_best_b6_node(tree_copy, max_diff=None)
            
            # 3. Wait
            print(f"\n--- Waiting {CHECK_INTERVAL} seconds for next check... (Queue size: {fetch_queue.qsize()}) ---")
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print(f"Error in scraper loop: {e}")
            time.sleep(CHECK_INTERVAL) # Don't crash loop

def main():
    """Main function to load data and start threads."""
    print("Loading initial data...")
    load_data()
    print(f"Loaded {len(shared_tree)} nodes and {len(shared_active_hashes)} known active hashes.")
    
    # --- NEW: Populate queue from loaded data ---
    populate_queue_from_stubs()
    
    # --- NEW: Start the queue worker thread ---
    worker_thread = threading.Thread(target=queue_worker, daemon=True)
    worker_thread.start()

    # --- Start the main page poller thread ---
    scraper_thread = threading.Thread(target=scraper_loop, daemon=True)
    scraper_thread.start()
    
    print("\n========================================================")
    print("Starting Flask API server...")
    print("View the visualization at: http://127.0.0.1:5000/")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

if __name__ == "__main__":
    main()