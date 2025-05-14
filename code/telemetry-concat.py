# ─── k_window_builder.py ─────────────────────────────────────────────
from collections import deque
from pathlib import Path
from datetime import timedelta
import json

from telemetry_sim import (
    simulate_episode,
    WINDOW_LEN,           # 120 s
    window_events,        # unchanged
)

K = 4                                 # windows per sample = 8 min span
NO_OVERLAP = WINDOW_LEN               # stride = Δ  (full shift)

def concat_windows(windows):
    """Return a single UTF‑8 string prompt for the LLM."""
    return "\n---\n".join(json.dumps(w.to_json()) for w in windows)

def generate_kstack_jsonl(out_file: str, n_episodes: int = 100):
    out = Path(out_file).open("w", encoding="utf-8")
    start = datetime.utcnow()

    for _ in range(n_episodes):
        label = random.choice(["benign", "ransomware"])
        length = timedelta(minutes=random.randint(3, 10))
        events = simulate_episode(label, start, length)

        w_start = start
        windows_deque = deque(maxlen=K)
        end_time = start + length

        while w_start + WINDOW_LEN <= end_time:
            w = window_events(events, w_start)
            windows_deque.append(w)

            # only emit once we have K full windows
            if len(windows_deque) == K:
                sample = {
                    "context": concat_windows(windows_deque),
                    "label": windows_deque[-1].label,
                    "stage": windows_deque[-1].stage,
                    "episode_id": windows_deque[-1].episode_id,
                    "window_end": w_start.isoformat()
                }
                out.write(json.dumps(sample) + "\n")

            w_start += NO_OVERLAP      # 120 s advance, no overlap

        start += length + timedelta(minutes=1)

    out.close()
