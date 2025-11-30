import os
import pygame
os.environ["SDL_AUDIODRIVER"] = "alsa"

# ===============================
#  水平校正設定（你要求新增的功能）
# ===============================
ENABLE_HORIZONTAL_CALIBRATION = False   # 預設 False = 不水平校正，用原本畫面座標

# 嘗試多種 AUDIODEV
alsa_devs = [
    "plughw:1,0","plughw:0,3","plughw:0,7","plughw:0,8","plughw:0,9",
    "plughw:1,1","plughw:1,3","default","dmix"
]
success = False
for dev in alsa_devs:
    os.environ["AUDIODEV"] = dev
    try:
        pygame.mixer.init()
        print(f"[ALSA] Successfully using audio device: {dev}")
        success = True
        break
    except pygame.error:
        pass
if not success:
    print("[ALSA ERROR] No audio device available")
    exit(1)

import cv2
import mediapipe as mp
import numpy as np
from play_notes import SoundPlayer


# ===============================
# 音符設定
# ===============================
LEFT_HAND_NOTES  = ['B3', 'A3', 'G3', 'F3', 'E3']
RIGHT_HAND_NOTES = ['C4', 'D4', 'E4', 'F4', 'G4']
ALL_NOTES = LEFT_HAND_NOTES + RIGHT_HAND_NOTES
sound_dir = os.path.join(os.path.dirname(__file__), '../resources/sounds')
player = SoundPlayer(ALL_NOTES, sound_dir)

# ===============================
# 八度控制
# ===============================
octave_shift = 0
octave_cooldown = 0
gesture_frames_required = 45       # 1.5s
octave_gesture_up = 0
octave_gesture_down = 0

# ===============================
# 綠光計時器
# ===============================
finger_glow_timer_left  = [0] * 5
finger_glow_timer_right = [0] * 5

# ===============================
# 攝影機偵測
# ===============================
def find_available_camera(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                cap.release()
                print(f"[Camera] Using /dev/video{i}")
                return i
        cap.release()
    print("[Camera ERROR] No camera found")
    exit(1)

cam_index = find_available_camera()
cap = cv2.VideoCapture(cam_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.9
)

finger_pressed_left  = [False] * 5
finger_pressed_right = [False] * 5
previous_positions_left  = [None] * 5
previous_positions_right = [None] * 5

desk_edge_y = None

# ===============================
# 工具函式
# ===============================
def smooth_position(cur, pre, alpha=0.3):
    if pre is None:
        return cur
    return tuple(alpha*c + (1-alpha)*p for c,p in zip(cur, pre))

def detect_desk_edge(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(y1 - y2) < 5:
                return int((y1+y2)/2)
    return None

def calculate_velocity(cur, pre):
    if cur is None or pre is None:
        return 0
    return cur[1] - pre[1]

# 依 octave_shift 調整音符
def apply_octave(note, shift):
    name = note[0]
    octave = int(note[1])
    return f"{name}{octave + shift}"

# ===============================
# 主迴圈
# ===============================
while True:

    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if ENABLE_HORIZONTAL_CALIBRATION:
        if desk_edge_y is None:
            desk_edge_y = detect_desk_edge(frame)

    if ENABLE_HORIZONTAL_CALIBRATION:
        if desk_edge_y is None:
            cv2.putText(frame, "Desk Edge Not Detected.", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Virtual Piano", frame)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break
            continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hid, hand_lm in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[hid].classification[0].label
            hand_type = "LEFT" if hand_label == "Left" else "RIGHT"

            hand_notes   = LEFT_HAND_NOTES if hand_type=="LEFT" else RIGHT_HAND_NOTES
            fp           = finger_pressed_left if hand_type=="LEFT" else finger_pressed_right
            prev_pos     = previous_positions_left if hand_type=="LEFT" else previous_positions_right
            glow_timer   = finger_glow_timer_left if hand_type=="LEFT" else finger_glow_timer_right

            fingertips = []
            fingertip_ids = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]

            # palm base (for local coordinate)
            palm = hand_lm.landmark[mp_hands.HandLandmark.WRIST]
            palm_y = palm.y

            ys = []
            for lm_idx in fingertip_ids:
                lm = hand_lm.landmark[lm_idx]

                if ENABLE_HORIZONTAL_CALIBRATION:
                    # 使用相對手掌的 Y → 不受攝影機角度影響
                    y_value = lm.y - palm_y
                else:
                    # 使用影像座標（預設）
                    y_value = lm.y * h

                ys.append(y_value)
                fingertips.append((int(lm.x*w), int(lm.y*h)))

            thumb, idx, mid, ring, pinky = ys

            # ===============================
            # 八度手勢（只看左手）
            # ===============================
            is_up   = (thumb < idx < mid < ring < pinky)
            is_down = (pinky < ring < mid < idx < thumb)

            if hand_type == "LEFT" and octave_cooldown == 0:

                if is_up:
                    octave_gesture_up += 1
                    octave_gesture_down = 0
                    if octave_gesture_up >= gesture_frames_required:
                        octave_shift = min(octave_shift+1, 2)
                        octave_cooldown = 30
                        octave_gesture_up = 0
                        print(f"[Octave] Shift UP → {octave_shift}")

                elif is_down:
                    octave_gesture_down += 1
                    octave_gesture_up = 0
                    if octave_gesture_down >= gesture_frames_required:
                        octave_shift = max(octave_shift-1, -2)
                        octave_cooldown = 30
                        octave_gesture_down = 0
                        print(f"[Octave] Shift DOWN → {octave_shift}")

                else:
                    octave_gesture_up = 0
                    octave_gesture_down = 0

            octave_active = (hand_type == "LEFT" and (octave_gesture_up > 0 or octave_gesture_down > 0))

            # ===============================
            # 指尖處理 + 播放音符 + 顯色
            # ===============================
            for i, pos in enumerate(fingertips):

                smoothed = smooth_position(pos, prev_pos[i])
                vel = calculate_velocity(smoothed, prev_pos[i])
                prev_pos[i] = smoothed

                # 播放音符（加綠光）
                if vel > 5 and not fp[i]:
                    fp[i] = True

                    if hand_type=="LEFT":
                        base_index = LEFT_HAND_NOTES.index(hand_notes[i])
                    else:
                        base_index = 5 + RIGHT_HAND_NOTES.index(hand_notes[i])

                    final_index = base_index + octave_shift*12
                    player.play_note_by_index(final_index)

                    glow_timer[i] = 9

                elif vel < -3:
                    fp[i] = False

                # ===============================
                # 顏色決定
                # ===============================
                if glow_timer[i] > 0:
                    color = (0,255,0)     # 綠
                    glow_timer[i] -= 1
                elif octave_active:
                    color = (255,0,0)     # 藍（僅左手）
                else:
                    color = (0,0,255)     # 紅（預設）

                shown_note = apply_octave(hand_notes[i], octave_shift)

                cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 10, color, -1)
                cv2.putText(frame, shown_note,
                    (int(smoothed[0]) + 10, int(smoothed[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ===============================
    #  UI 顯示
    # ===============================
    if ENABLE_HORIZONTAL_CALIBRATION and desk_edge_y is not None:
        cv2.line(frame, (0, desk_edge_y), (w, desk_edge_y), (255,255,0), 2)

    cv2.putText(frame, f"Octave Shift: {octave_shift}", (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.putText(frame, "Deep Learning & Application (Final Project)", (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    cv2.imshow("Virtual Piano", frame)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break

    if octave_cooldown > 0:
        octave_cooldown -= 1

cap.release()
cv2.destroyAllWindows()
