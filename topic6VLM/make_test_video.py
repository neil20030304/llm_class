"""
make_test_video.py — generate a synthetic test video for exercise2.py
======================================================================

Produces a 20-second 1-fps clip (`test_video.mp4`) of an "empty room" — a
beige wall, a dark-blue floor, and a door frame on the right. A crude human
silhouette (head + body + legs + arms) is drawn into the frame between
t=6s and t=14s; the rest of the time the room is empty.

The output is the input to `exercise2.py`, which samples it every 2 seconds
and asks LLaVA whether a person is present in each sampled frame.
"""
import cv2
import numpy as np

OUT = "test_video.mp4"
W, H = 480, 360
FPS = 1
DURATION_S = 20
PERSON_FROM, PERSON_TO = 6, 14   # seconds inclusive of person presence


def draw_room(img):
    """Draw a simple "empty room" background (floor + wall + door)."""
    img[:] = (210, 220, 225)                                   # wall
    cv2.rectangle(img, (0, 240), (W, H), (90, 70, 55), -1)     # floor
    cv2.rectangle(img, (340, 80), (430, 240), (60, 40, 30), 3) # door frame
    cv2.line(img, (0, 240), (W, 240), (0, 0, 0), 2)


def draw_person(img, cx=240, base_y=235):
    """Draw a stick-figure-ish silhouette."""
    # Head
    cv2.circle(img, (cx, base_y - 130), 22, (40, 30, 25), -1)
    # Body
    cv2.rectangle(img, (cx - 22, base_y - 108), (cx + 22, base_y - 30),
                  (40, 30, 25), -1)
    # Legs
    cv2.rectangle(img, (cx - 20, base_y - 30), (cx - 4, base_y),
                  (30, 30, 60), -1)
    cv2.rectangle(img, (cx + 4, base_y - 30), (cx + 20, base_y),
                  (30, 30, 60), -1)
    # Arms
    cv2.rectangle(img, (cx - 38, base_y - 105), (cx - 22, base_y - 55),
                  (40, 30, 25), -1)
    cv2.rectangle(img, (cx + 22, base_y - 105), (cx + 38, base_y - 55),
                  (40, 30, 25), -1)


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))

    for sec in range(DURATION_S):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_room(frame)
        if PERSON_FROM <= sec <= PERSON_TO:
            draw_person(frame)
        cv2.putText(frame, f"t={sec:02d}s", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()
    print(f"wrote {OUT}  ({DURATION_S}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
