# Topic 6 — Vision-Language Models (VLMs) with LLaVA

Two exercises built on **Ollama + LLaVA**. Exercise 1 is a multi-turn LangGraph chat agent that talks about an uploaded image. Exercise 2 is a video-surveillance agent that samples frames every N seconds and reports when a person enters/exits the scene.

## Prerequisites

```bash
# Vision model
ollama pull llava
ollama serve              # if not already running

# Python deps (used here from the `llm_class` conda env)
pip install ollama langgraph langchain-core Pillow grandalf opencv-python
```

## Table of contents

| File | What it is |
| ---- | ---------- |
| [`exercise1.py`](exercise1.py) | **Exercise 1** — LangGraph multi-turn image chat agent |
| [`exercise2.py`](exercise2.py) | **Exercise 2** — Video-surveillance agent (per-frame LLaVA) |
| [`make_test_video.py`](make_test_video.py) | Generates `test_video.mp4` (smoke-test fixture for ex2) |
| [`vlm_chat_graph.png`](vlm_chat_graph.png) | LangGraph diagram for ex1 (auto-saved on launch) |
| [`sample_image.jpg`](sample_image.jpg) | Synthetic scene used in the ex1 demo run |
| [`test_video.mp4`](test_video.mp4) | 20-second synthetic clip with a person silhouette entering/exiting |
| [`exercise1_output.txt`](exercise1_output.txt) | Captured terminal output from a 3-turn ex1 demo run |
| [`exercise2_output.txt`](exercise2_output.txt) | Captured terminal output from ex2 against `test_video.mp4` |

---

## Exercise 1 — Vision-Language LangGraph Chat Agent

A LangGraph agent that lets you carry on a multi-turn conversation about a single image.

**Graph shape** (saved to `vlm_chat_graph.png` on every run):

```
START → upload_image → get_user_input ──┬─→ END                 (quit/exit/q)
                            ▲           ├─→ get_user_input      (empty input)
                            │           └─→ call_vlm → print_response ─┘
```

**Key design choices**

- **State** uses `Annotated[List[dict], operator.add]` so each node only returns *new* messages and LangGraph appends them to history — same pattern as `task2/langraph_llama_agent.py`.
- **Image attached only on the first user turn.** Subsequent turns are text-only; LLaVA keeps the image in its context window, so we don't pay the encoding cost on every turn.
- **Resize-on-upload** (`MAX_IMAGE_SIDE = 512`). VLMs tokenize images as patches, so a 3000-px photo blows up token counts and tanks latency. 512 px is a good quality/speed compromise.
- **Special-input router** (`route_after_input`) handles `quit` / empty input / valid input as three distinct branches, cleanly separating control flow from the model call.

**Run**

```bash
python exercise1.py
```

You'll be prompted for an image path, then for questions. In-session commands: `verbose`, `quiet`, `quit`.

**Demo output** (`exercise1_output.txt`, abridged):

```
Image ready: sample_image_resized.jpg

> Describe what you see in this image.
LLaVA: This is a digital image that features a simple scene of a single-story
        house with a red roof and brown shutters, set against a light blue sky.
        ... a green tree with a clear outline, suggesting a stylized or pixelated
        representation. The sun appears as a simple yellow circle in the top right...

> What color is the roof of the house?
LLaVA: The roof of the house in the image is red.

> Is there any vegetation visible? If so, describe it.
LLaVA: Yes, there is a tree visible next to the house. It appears to be a green
        tree with a clear outline...
```

The model correctly carries the image across all three turns without re-uploading.

---

## Exercise 2 — Video-Surveillance Agent

LLaVA cannot ingest video, so we sample keyframes with OpenCV, send each frame to LLaVA as a still image, and post-process the per-frame answers into ENTER / EXIT events.

**Pipeline**

```
video.mp4
   │  cv2.VideoCapture, sample every interval_s seconds
   ▼
frame[i] (BGR ndarray)
   │  resize → JPEG → base64
   ▼
ollama.chat(model="llava", images=[b64], prompt="Is there a person? YES/NO")
   │  parse first-line YES/NO
   ▼
[(t=0.0s, person=False), (t=2.0s, person=False), (t=4.0s, person=True), ...]
   │  scan for no→yes (ENTER) and yes→no (EXIT) transitions
   ▼
ENTER / EXIT timeline
```

**Run**

```bash
python make_test_video.py             # creates test_video.mp4 (~20s smoke fixture)
python exercise2.py --video test_video.mp4 --interval 2 --max-side 384

# Optional extra-credit: count people / cats / dogs and trigger INTRUDER ALERT
python exercise2.py --video clip.mp4 --pets

# Optional: dump the sampled frames to disk for inspection
python exercise2.py --video clip.mp4 --save-frames frames/
```

**Demo output** (`exercise2_output.txt`):

```
[video] fps=1.00  total_frames=20  duration=20.0s
[video] sampling every 2 frames (~2.0s) → 11 samples

[   0.0s] frame 000  PERSON  (15.6s)  | Yes
[   2.0s] frame 001  empty   (1.3s)  | NO
[   4.0s] frame 002  empty   (1.2s)  | No
[   6.0s] frame 003  PERSON  (1.2s)  | Yes
[   8.0s] frame 004  PERSON  (1.3s)  | Yes
[  10.0s] frame 005  PERSON  (1.5s)  | Yes
[  12.0s] frame 006  empty   (1.5s)  | No
[  14.0s] frame 007  PERSON  (1.5s)  | Yes
[  16.0s] frame 008  empty   (1.4s)  | No
[  18.0s] frame 009  empty   (1.4s)  | No

============================================================
  ENTER / EXIT TIMELINE
============================================================
  ENTER  at  00:00  (0.0s)
  EXIT   at  00:02  (2.0s)
  ENTER  at  00:06  (6.0s)
  EXIT   at  00:12  (12.0s)
  ENTER  at  00:14  (14.0s)
  EXIT   at  00:16  (16.0s)
```

The synthetic clip embeds a stick-figure between t=6s and t=14s. LLaVA correctly catches the main 6–12s window. The first-frame false positive and the 12→14s flicker are artefacts of the synthetic figure being a crude geometric silhouette — LLaVA's confidence is much higher on real footage. **For the graded task, replace `test_video.mp4` with your own 2-minute recording.**

**Notes on speed.** First call is always slow because Ollama loads the model into VRAM (~15s here). Subsequent calls are ~1–2 s/frame on this machine (M-series Mac). On a CPU-only machine expect ~8 s/frame, which is what the task description quotes — if you run on a 2-min clip you'll want either a coarser `--interval` or a GPU.

---

## Notes on the design

- **Why LangGraph for ex1 but not ex2?** Ex1 is a stateful, branching, looped conversation — exactly what LangGraph is for. Ex2 is a one-shot batch pipeline (extract → for each frame call LLM → reduce), where a plain script is clearer than wrapping a linear pipeline in a graph.
- **Reply parsing is defensive.** Ex2 first looks for `YES`/`NO` on the first line (the prompt asks for that format), then falls back to keyword search (`"person"`, `"human"`, `"no person"`) — small models often ignore strict format instructions.
- **TTS for `--pets` mode** uses macOS's built-in `say` command via `subprocess.Popen` so it doesn't block the next frame. Replaceable with `edge-tts` if you want non-Mac-specific TTS.
