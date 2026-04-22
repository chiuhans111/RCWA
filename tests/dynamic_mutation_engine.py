import subprocess
import os
import shutil
import sys
import re
import random
import signal
import time

# --- Configuration & Rules ---
LOCK_FILE = os.path.join(os.path.dirname(__file__), "mutation.lock")
TARGET_DIR = "rcwa"

RULES = [
    (r' \+ ', ' - ', 'SUBST_ADD_SUB'),
    (r' \- ', ' + ', 'SUBST_SUB_ADD'),
    (r' \* ', ' / ', 'SUBST_MUL_DIV'),
    (r'np\.sin', 'np.cos', 'FUNC_SIN_COS'),
    (r'np\.cos', 'np.sin', 'FUNC_COS_SIN'),
    (r'np\.sqrt', 'np.square', 'FUNC_SQRT_SQR'),
]

# --- Global State for Cleanup ---
class MutationState:
    def __init__(self):
        self.active_file = None
        self.backup_path = None

state = MutationState()

def restore_all_backups(silent=False):
    """Scans the TARGET_DIR for any legacy .original files and restores them."""
    found = False
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith('.original'):
                original_path = os.path.join(root, file)
                target_path = original_path.replace('.original', '')
                if not silent:
                    print(f"  [RECOVERY] Restoring {target_path} from backup...")
                shutil.move(original_path, target_path)
                found = True
    return found

def signal_handler(sig, frame):
    """Handles Ctrl+C and ensures everything is restored before exiting."""
    print("\n\n[INTERRUPT] Caught signal. Restoring source code...")
    cleanup_and_exit(1)

def cleanup_and_exit(exit_code=0):
    if state.active_file and state.backup_path and os.path.exists(state.backup_path):
        shutil.move(state.backup_path, state.active_file)
    
    # Final safety sweep
    restore_all_backups(silent=True)
    
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    
    sys.exit(exit_code)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)

# --- Core Logic ---

def run_tests():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "tests"],
            capture_output=True, text=True, timeout=90
        )
        return "OK" in result.stderr and result.returncode == 0
    except Exception:
        return False

def discover_mutation_sites(directory):
    sites = []
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py') or file == '__init__.py':
                continue
            
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith('#') or clean_line.startswith(('import', 'from')):
                    continue
                if '"""' in line or "'''" in line:
                    continue
                
                for pattern, replacement, label in RULES:
                    if re.search(pattern, line):
                        mutated = re.sub(pattern, replacement, line, count=1)
                        sites.append({
                            "file": path, "line_idx": i,
                            "original": line, "mutated": mutated, "label": label
                        })
    return sites

def apply_mutation(file_path, line_idx, new_line):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines[line_idx] = new_line
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def main():
    print("=== RCWA Bulletproof Dynamic Mutation Engine ===")
    
    # 1. Lock Protection
    if os.path.exists(LOCK_FILE):
        print(f"ERROR: Lock file detected at {LOCK_FILE}.")
        print("This prevents concurrent mutation runs that could corrupt the source.")
        print("If you are sure no other instance is running, delete the lock file manually.")
        sys.exit(1)
        
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))

    try:
        # 2. Self-Healing Startup
        print("Performing pre-flight health check...")
        if restore_all_backups():
            print("  System recovered from a previous crash state.")
        else:
            print("  System is healthy.")

        # 3. Discovery
        all_sites = discover_mutation_sites(TARGET_DIR)
        print(f"Discovered {len(all_sites)} total mutation sites.")
        
        count = 10
        random.shuffle(all_sites)
        test_sites = all_sites[:count]
        
        print(f"Executing randomized sweep of {len(test_sites)} sites...\n")
        results = []
        
        for site in test_sites:
            file_path = site["file"]
            state.active_file = file_path
            state.backup_path = file_path + ".original"
            
            rel_path = os.path.relpath(file_path, os.getcwd())
            description = f"{rel_path}:{site['line_idx']+1} [{site['label']}]"
            print(f"Mutating {description}...")
            
            shutil.copy(file_path, state.backup_path)
            try:
                apply_mutation(file_path, site["line_idx"], site["mutated"])
                passed = run_tests()
                status = "MISSED" if passed else "CAUGHT"
                results.append((description, status))
                print(f"  Result: {status}")
            finally:
                # Cleanup active mutation immediately
                if os.path.exists(state.backup_path):
                    shutil.move(state.backup_path, state.active_file)
                state.active_file = None
                state.backup_path = None
                
        # 4. Reporting
        print("\n=== Bulletproof Mutation Report ===")
        print(f"{'Site (File:Line [Mutation Type])':<60} | {'Status':<10}")
        print("-" * 75)
        for desc, status in results:
            indicator = "[PASS]" if status == "CAUGHT" else "[FAIL]"
            print(f"{desc:<60} | {status:<10} {indicator}")
        
        caught = sum(1 for _, s in results if s == "CAUGHT")
        print(f"\nSensitivity Score: {caught}/{len(results)} bugs detected.")
        if caught == len(results):
            print("EXCELLENT: Test suite caught all random mutations.")
        else:
            print("NOTE: Investigate MISSED sites for missing logic coverage.")

    finally:
        # Final safety cleanup
        cleanup_and_exit(0)

if __name__ == "__main__":
    main()
