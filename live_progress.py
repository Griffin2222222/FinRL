import time
import os
import sys
import pickle
import numpy as np

def live_progress(log_path="checkpoints/training_log.pkl", refresh=2):
    """
    Live terminal graphic for FinRL training progress.
    Reads the training log and prints a live progress bar and stats.
    """
    last_ep = -1
    while True:
        os.system('clear')
        print("=== FinRL Training Progress ===\n")
        if not os.path.exists(log_path):
            print("Waiting for training log...")
            time.sleep(refresh)
            continue
        try:
            with open(log_path, "rb") as f:
                log = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read log: {e}")
            time.sleep(refresh)
            continue
        if not log:
            print("No progress yet...")
            time.sleep(refresh)
            continue
        last = log[-1]
        ep = last.get('episode', 0)
        print(f"Episode: {ep}")
        print(f"Last update: {last.get('timestamp', '')}")
        # Add more stats as needed (PnL, drawdown, etc.)
        # Progress bar
        bar_len = 40
        progress = min(ep / 1000, 1.0)  # Assume 1000 episodes for full bar
        bar = '[' + '#' * int(bar_len * progress) + '-' * (bar_len - int(bar_len * progress)) + ']'
        print(f"Progress: {bar} {progress*100:.1f}%\n")
        # Optionally, print recent stats
        print("Recent episodes:")
        for l in log[-5:]:
            print(f"Ep {l.get('episode', '')}: {l.get('timestamp', '')}")
        sys.stdout.flush()
        time.sleep(refresh)

if __name__ == "__main__":
    live_progress()
