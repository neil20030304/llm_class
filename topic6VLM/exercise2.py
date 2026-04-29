"""
exercise2.py — Video Surveillance Agent (LLaVA via Ollama)
==========================================================

Walks through a video file, extracts a frame every N seconds (default 2s),
asks LLaVA "is there a person in this scene?" for each frame, and reports
the timestamps at which a person ENTERS and EXITS the scene.

LLaVA cannot consume video directly, so we sample keyframes with OpenCV and
feed them as still images. Frame N's timestamp is `frame_index / fps`.

Usage
-----
    python exercise2.py --video test_video.mp4
    python exercise2.py --video clip.mp4 --interval 2 --max-side 384

    # Extra-credit mode: detect person/cat/dog and shout INTRUDER ALERT for people
    python exercise2.py --video clip.mp4 --pets

Requirements
------------
    pip install opencv-python ollama Pillow
    ollama pull llava
"""

import argparse
import base64
import os
import re
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import cv2
import ollama
from PIL import Image

OLLAMA_MODEL = "llava"

PERSON_PROMPT = (
    "Is there a person (a human being) visible anywhere in this image? "
    "Answer with exactly one word on the first line: YES or NO. "
    "On a second line, give a one-sentence reason."
)

PETS_PROMPT = (
    "Carefully examine this image. Tell me how many of each appear: "
    "people (humans), cats, dogs. "
    "Reply in exactly this format on three lines, with integer counts:\n"
    "PEOPLE: <n>\n"
    "CATS: <n>\n"
    "DOGS: <n>\n"
    "Then a one-sentence reason."
)


@dataclass
class FrameResult:
    index: int          # which sampled frame (0, 1, 2, ...)
    timestamp_s: float  # seconds into the video
    person: bool
    raw_reply: str
    n_people: int = 0
    n_cats: int = 0
    n_dogs: int = 0


# --------------------------------------------------------------------------
# Frame extraction
# --------------------------------------------------------------------------
def extract_frames(video_path: str, interval_s: float = 2.0):
    """Yield (timestamp_seconds, BGR_frame) tuples sampled every interval_s."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = max(1, int(round(fps * interval_s)))

    print(f"[video] {video_path}")
    print(f"[video] fps={fps:.2f}  total_frames={total_frames}  "
          f"duration={total_frames / fps:.1f}s")
    print(f"[video] sampling every {interval_frames} frames "
          f"(~{interval_s}s) → {total_frames // interval_frames + 1} samples")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval_frames == 0:
            yield frame_idx / fps, frame
        frame_idx += 1
    cap.release()


# --------------------------------------------------------------------------
# Frame → base64 (with optional resize for speed)
# --------------------------------------------------------------------------
def frame_to_b64(frame, max_side: int = 512) -> str:
    """BGR ndarray → base64-encoded JPEG string for Ollama's `images` field."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------------------------------------------------------------
# Reply parsing
# --------------------------------------------------------------------------
def parse_person_reply(reply: str) -> bool:
    """First-line YES/NO; if absent, look for 'yes' / 'person' anywhere."""
    first = reply.strip().splitlines()[0].strip().upper() if reply.strip() else ""
    if first.startswith("YES"):
        return True
    if first.startswith("NO"):
        return False
    low = reply.lower()
    if "no person" in low or "there is no" in low:
        return False
    return "person" in low or "human" in low or "yes" in low


def parse_pets_reply(reply: str):
    """Returns (n_people, n_cats, n_dogs) from the PEOPLE:/CATS:/DOGS: format."""
    def grab(label: str) -> int:
        m = re.search(rf"{label}\s*[:=]\s*(\d+)", reply, re.IGNORECASE)
        return int(m.group(1)) if m else 0
    return grab("PEOPLE"), grab("CATS"), grab("DOGS")


# --------------------------------------------------------------------------
# LLaVA call per frame
# --------------------------------------------------------------------------
def analyze_frame(b64_image: str, prompt: str) -> str:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [b64_image]}],
    )
    return response["message"]["content"]


# --------------------------------------------------------------------------
# Transition detection: find ENTER (no→yes) and EXIT (yes→no) moments
# --------------------------------------------------------------------------
def find_transitions(results: List[FrameResult]):
    """Return a list of ('ENTER'|'EXIT', timestamp_s) events."""
    events = []
    prev = False
    for r in results:
        if r.person and not prev:
            events.append(("ENTER", r.timestamp_s))
        elif not r.person and prev:
            events.append(("EXIT", r.timestamp_s))
        prev = r.person
    # If still present at the very end, note it
    if results and results[-1].person:
        events.append(("STILL_PRESENT_AT_END", results[-1].timestamp_s))
    return events


# --------------------------------------------------------------------------
# Optional: text-to-speech for INTRUDER ALERT
# --------------------------------------------------------------------------
def speak_intruder_alert():
    """Best-effort TTS on macOS. Falls back to a printed banner."""
    try:
        import subprocess
        subprocess.Popen(
            ["say", "Intruder alert!"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print("\a")  # terminal bell as last resort


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run(video_path: str, interval_s: float, max_side: int, pets_mode: bool,
        save_frames_dir: Optional[str] = None):
    if not os.path.isfile(video_path):
        sys.exit(f"Video not found: {video_path}")

    prompt = PETS_PROMPT if pets_mode else PERSON_PROMPT
    results: List[FrameResult] = []

    print(f"[mode] {'pets' if pets_mode else 'person-only'}  "
          f"interval={interval_s}s  max_side={max_side}px")
    print()

    t0 = time.time()
    for i, (ts, frame) in enumerate(extract_frames(video_path, interval_s)):
        if save_frames_dir:
            os.makedirs(save_frames_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_frames_dir, f"frame_{i:04d}.jpg"), frame)

        b64 = frame_to_b64(frame, max_side=max_side)
        t_call = time.time()
        reply = analyze_frame(b64, prompt)
        dt = time.time() - t_call

        if pets_mode:
            n_p, n_c, n_d = parse_pets_reply(reply)
            person = n_p > 0
            print(f"[{ts:6.1f}s] frame {i:03d}  "
                  f"people={n_p} cats={n_c} dogs={n_d}  ({dt:.1f}s)")
            if n_p > 0 and n_c == 0 and n_d == 0:
                print("           ⚠  INTRUDER ALERT")
                speak_intruder_alert()
            results.append(FrameResult(i, ts, person, reply, n_p, n_c, n_d))
        else:
            person = parse_person_reply(reply)
            tag = "PERSON" if person else "empty "
            first_line = reply.strip().splitlines()[0] if reply.strip() else ""
            print(f"[{ts:6.1f}s] frame {i:03d}  {tag}  ({dt:.1f}s)  "
                  f"| {first_line[:60]}")
            results.append(FrameResult(i, ts, person, reply))

    total = time.time() - t0
    print()
    print(f"[done] processed {len(results)} frames in {total:.1f}s "
          f"({total / max(1, len(results)):.1f}s/frame)")

    # Report transitions
    events = find_transitions(results)
    print()
    print("=" * 60)
    print("  ENTER / EXIT TIMELINE")
    print("=" * 60)
    if not events:
        print("  No person detected in any sampled frame.")
    else:
        for kind, ts in events:
            mm, ss = divmod(int(ts), 60)
            print(f"  {kind:22s}  at  {mm:02d}:{ss:02d}  ({ts:.1f}s)")
    print("=" * 60)
    return results, events


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="path to .mp4 / .mov / .avi")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between sampled frames (default 2.0)")
    ap.add_argument("--max-side", type=int, default=512,
                    help="resize longest image side before sending to LLaVA "
                         "(smaller = faster, default 512)")
    ap.add_argument("--pets", action="store_true",
                    help="extra-credit mode: count people/cats/dogs and "
                         "INTRUDER ALERT on people only")
    ap.add_argument("--save-frames", default=None,
                    help="optional directory to dump sampled frames as JPEGs")
    args = ap.parse_args()

    run(args.video, args.interval, args.max_side, args.pets, args.save_frames)


if __name__ == "__main__":
    main()
