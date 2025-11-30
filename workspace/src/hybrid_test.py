import os
import sys
import pygame
import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# ==========================================================
# Audio init
# ==========================================================
os.environ["SDL_AUDIODRIVER"] = "alsa"

alsa_devs = [
    "plughw:1,0",
    "plughw:0,3",
    "plughw:0,7",
    "plughw:0,8",
    "plughw:0,9",
    "plughw:1,1",
    "plughw:1,3",
    "default",
    "dmix",
]
success = False
errors = []
for dev in alsa_devs:
    os.environ["AUDIODEV"] = dev
    try:
        pygame.mixer.init()
        print(f"[ALSA] Successfully using audio device: {dev}")
        success = True
        break
    except pygame.error as e:
        errors.append(f"{dev}: {e}")

if not success:
    print("[ALSA ERROR] Could not open any audio device.")
    sys.exit(1)

# ==========================================================
# 88-key setup
# ==========================================================
WHITE_NOTES = ["C", "D", "E", "F", "G", "A", "B"]

def generate_88_keys():
    notes = []
    notes += ["A0", "A#0", "B0"]
    for octave in range(1, 8):
        for n in WHITE_NOTES:
            notes.append(f"{n}{octave}")
            if n in ["C", "D", "F", "G", "A"]:
                notes.append(f"{n}#{octave}")
    notes.append("C8")

    order = []
    base_order = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    order += ["A0","A#0","B0"]
    for octave in range(1, 9):
        for n in base_order:
            key = f"{n}{octave}"
            if key in notes:
                order.append(key)
    return order

ALL_88_NOTES = generate_88_keys()
TOTAL_KEYS = len(ALL_88_NOTES)

# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
print("[Path] Working directory set to:", BASE_DIR)

SOUND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "resources", "sounds"))

# ==========================================================
class SoundPlayer:
    def __init__(self, notes, sound_dir):
        self.notes = notes
        self.sounds = {}
        print(f"[SoundPlayer] Loading sounds from: {sound_dir}")

        for note in notes:
            f = os.path.join(sound_dir, f"{note}.mp3")
            if os.path.exists(f):
                try:
                    self.sounds[note] = pygame.mixer.Sound(f)
                except pygame.error as e:
                    print(f"[Sound ERR] {note}: {e}")
            else:
                print(f"[Missing] {note}.mp3")

    def play_note_by_index(self, idx):
        if 0 <= idx < len(self.notes):
            note = self.notes[idx]
            if note in self.sounds:
                try:
                    self.sounds[note].play()
                except:
                    pass

player = SoundPlayer(ALL_88_NOTES, SOUND_DIR)

# ==========================================================
# RealSense camera
# ==========================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
    print("[RealSense] Color stream started.")
except Exception as e:
    print("[RealSense ERROR]", e)
    sys.exit(1)

# ==========================================================
# Mediapipe
# ==========================================================
mp_hands = mp.solutions.hands
mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.9
)

# ==========================================================
# State variables
# ==========================================================
finger_pressed_left = [False] * 5
finger_pressed_right = [False] * 5
prev_left = [None] * 5
prev_right = [None] * 5
key_glow_timer = [0] * TOTAL_KEYS

# octave shift system
octave_shift = 0
octave_up_cnt = 0
octave_dn_cnt = 0
octave_cooldown = 0
gesture_frames_required = 45

ENABLE_HORIZONTAL_CALIBRATION = False

# ==========================================================
def smooth_position(c, p, a=0.3):
    if p is None:
        return c
    return (a * c[0] + (1 - a) * p[0], a * c[1] + (1 - a) * p[1])

def velocity(c, p):
    if p is None:
        return 0
    return c[1] - p[1]

def draw_piano(frame, w, h):
    key_w = max(1, w // TOTAL_KEYS)
    key_h = 80
    y_top = h - key_h

    for i, note in enumerate(ALL_88_NOTES):
        x1 = i * key_w
        x2 = x1 + key_w

        if "#" in note:
            base = (50, 50, 50)
            txt = (255, 255, 255)
        else:
            base = (255, 255, 255)
            txt = (0, 0, 0)

        if key_glow_timer[i] > 0:
            color = (0, 255, 0)
            key_glow_timer[i] -= 1
        else:
            color = base

        cv2.rectangle(frame, (x1, y_top), (x2, h), color, -1)
        cv2.rectangle(frame, (x1, y_top), (x2, h), (0, 0, 0), 1)
        cv2.putText(frame, note, (x1 + 2, y_top + 20),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, txt, 1)

# ==========================================================
# Main loop
# ==========================================================
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = hands.process(rgb)

    if r.multi_hand_landmarks:
        for iHand, handLm in enumerate(r.multi_hand_landmarks):
            hand_label = r.multi_handedness[iHand].classification[0].label
            is_left = (hand_label == "Left")

            finger_ids = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            ]

            palm = handLm.landmark[mp_hands.HandLandmark.WRIST]
            palm_y = palm.y

            fingertips = []
            ys = []

            for fid in finger_ids:
                lm = handLm.landmark[fid]
                x, y = int(lm.x * w), int(lm.y * h)
                fingertips.append((x, y))

                if ENABLE_HORIZONTAL_CALIBRATION:
                    ys.append(lm.y - palm_y)
                else:
                    ys.append(y)

            thumb, idxF, midF, ringF, pinky = ys

            # ==================================================
            # Octave gesture (NO global needed)
            # ==================================================
            if is_left and octave_cooldown == 0:
                up = (thumb < idxF < midF < ringF < pinky)
                dn = (pinky < ringF < midF < idxF < thumb)

                if up:
                    octave_up_cnt += 1
                    octave_dn_cnt = 0
                    if octave_up_cnt >= gesture_frames_required:
                        octave_shift = min(2, octave_shift + 1)
                        octave_up_cnt = 0
                        octave_cooldown = 30
                        print("[Octave] UP →", octave_shift)

                elif dn:
                    octave_dn_cnt += 1
                    octave_up_cnt = 0
                    if octave_dn_cnt >= gesture_frames_required:
                        octave_shift = max(-2, octave_shift - 1)
                        octave_dn_cnt = 0
                        octave_cooldown = 30
                        print("[Octave] DOWN →", octave_shift)

                else:
                    octave_up_cnt = 0
                    octave_dn_cnt = 0

            # ==================================================
            # Finger press / play note
            # ==================================================
            FP = finger_pressed_left if is_left else finger_pressed_right
            PP = prev_left if is_left else prev_right

            for j, pos in enumerate(fingertips):
                sm = smooth_position(pos, PP[j])
                vel = velocity(sm, PP[j])
                PP[j] = sm

                if vel > 5 and not FP[j]:
                    FP[j] = True
                    x_pos = sm[0]

                    key_index = int((x_pos / w) * TOTAL_KEYS)
                    key_index = max(0, min(TOTAL_KEYS - 1, key_index))

                    key_glow_timer[key_index] = 9

                    shifted = key_index + octave_shift * 12
                    shifted = max(0, min(TOTAL_KEYS - 1, shifted))

                    player.play_note_by_index(shifted)

    if octave_cooldown > 0:
        octave_cooldown -= 1

    draw_piano(frame, w, h)

    cv2.imshow("Hybrid Piano (RealSense)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
