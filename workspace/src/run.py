import os
import pygame

# ===============================
#  水平校正設定（你要求新增的功能）
# ===============================
ENABLE_HORIZONTAL_CALIBRATION = False   # 預設 False = 不水平校正，用原本畫面座標

# 音訊初始化 - Docker container 環境配置
audio_initialized = False

# 設定 PulseAudio 環境變數 (連接到 host 的 PulseAudio/PipeWire)
os.environ["PULSE_SERVER"] = "unix:/run/user/1000/pulse/native"
os.environ["PULSE_COOKIE"] = "/run/user/1000/pulse/cookie"

# 優先嘗試 PulseAudio (透過 socket 連接到 host)
try:
    os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    print(f"[Audio] Using PulseAudio via socket (host audio)")
    audio_initialized = True
except pygame.error as e:
    print(f"[Audio] PulseAudio via socket failed: {e}")

# 如果 PulseAudio 失敗，直接嘗試 ALSA 硬體裝置
if not audio_initialized:
    os.environ["SDL_AUDIODRIVER"] = "alsa"
    # hw:1,0 = 主機板 Line Out (耳機/喇叭孔)
    # hw:0,3 = NVIDIA HDMI 音訊
    alsa_devices = ["hw:1,0", "plughw:1,0", "hw:0,3", "plughw:0,3"]
    for dev in alsa_devices:
        os.environ["AUDIODEV"] = dev
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            print(f"[Audio] Using ALSA device: {dev}")
            audio_initialized = True
            break
        except pygame.error:
            continue

if not audio_initialized:
    print("[Audio ERROR] No audio device available")
    print("[Hint] Make sure the container has access to /dev/snd and /run/user/1000/pulse")
    exit(1)

import cv2
import mediapipe as mp
import numpy as np
from play_notes import SoundPlayer
import wave
import struct
import time
from datetime import datetime


# ===============================
# 音符設定
# ===============================
LEFT_HAND_NOTES  = ['B3', 'A3', 'G3', 'F3', 'E3']
RIGHT_HAND_NOTES = ['C4', 'D4', 'E4', 'F4', 'G4']
# 增加所有八度的音符以支援八度切換和半音調整 (涵蓋 A0~C8 及半音)
ADDITIONAL_NOTES = [
    # 第0八度
    'A0', 'Bb0', 'B0',
    # 第1八度
    'C1', 'Db1', 'D1', 'Eb1', 'E1', 'F1', 'Gb1', 'G1', 'Ab1', 'A1', 'Bb1', 'B1',
    # 第2八度
    'C2', 'Db2', 'D2', 'Eb2', 'E2', 'F2', 'Gb2', 'G2', 'Ab2', 'A2', 'Bb2', 'B2',
    # 第3八度
    'C3', 'Db3', 'D3', 'Eb3',
    # 第4八度 (部分)
    'A4', 'Bb4', 'B4',
    # 第5八度
    'C5', 'Db5', 'D5', 'Eb5', 'E5', 'F5', 'Gb5', 'G5', 'Ab5', 'A5', 'Bb5', 'B5',
    # 第6八度
    'C6', 'Db6', 'D6', 'Eb6', 'E6', 'F6', 'Gb6', 'G6', 'Ab6', 'A6', 'Bb6', 'B6',
    # 第7八度
    'C7', 'Db7', 'D7', 'Eb7', 'E7', 'F7', 'Gb7', 'G7', 'Ab7', 'A7', 'Bb7', 'B7',
    # 第8八度
    'C8', 'Db8'
]
ALL_NOTES = LEFT_HAND_NOTES + RIGHT_HAND_NOTES + ADDITIONAL_NOTES
sound_dir = os.path.join(os.path.dirname(__file__), '../resources/sounds')
player = SoundPlayer(ALL_NOTES, sound_dir)

# ===============================
# 錄影設定
# ===============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.path.dirname(__file__), '../output')
os.makedirs(output_dir, exist_ok=True)

video_temp_path = os.path.join(output_dir, f'temp_video_{timestamp}.avi')
audio_temp_path = os.path.join(output_dir, f'temp_audio_{timestamp}.wav')
final_video_path = os.path.join(output_dir, f'piano_recording_{timestamp}.mp4')

print(f"[Recording] Video will be saved to: {final_video_path}")

# 音訊錄製緩衝區
audio_buffer = []
audio_sample_rate = 44100
start_time = time.time()

# ===============================
# 八度控制（左手）
# ===============================
octave_shift = 0
octave_cooldown = 0
gesture_frames_required = 45       # 1.5s
octave_gesture_up = 0
octave_gesture_down = 0

# ===============================
# 半音調整（右手）
# ===============================
semitone_shift = 0  # +1 = 升半階， -1 = 降半階
semitone_cooldown = 0
semitone_gesture_up = 0
semitone_gesture_down = 0

# ===============================
# 綠光計時器
# ===============================
finger_glow_timer_left  = [0] * 5
finger_glow_timer_right = [0] * 5

# ===============================
# 攝影機偵測
# ===============================
def find_available_camera(max_index=10):
    # 測試每個攝影機，讀取影像並檢查是否為彩色
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            
            # 嘗試設定為 YUYV 格式
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            ok, frame = cap.read()
            cap.release()
            
            if not ok or frame is None:
                continue
            
            # 檢查是否為彩色影像（3通道且有色彩變化）
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # 檢查是否為真正的彩色（不是灰階轉成的BGR）
                # 計算RGB通道的標準差，如果太小表示是灰階
                b, g, r = cv2.split(frame)
                if np.std(b - g) > 1 or np.std(g - r) > 1 or np.std(r - b) > 1:
                    print(f"[Camera] Using /dev/video{i} (color)")
                    return i
                else:
                    print(f"[Camera] Skipping /dev/video{i} (grayscale)")
        except Exception as e:
            print(f"[Camera] Error testing /dev/video{i}: {e}")
            continue
    
    print("[Camera ERROR] No color camera found")
    exit(1)

cam_index = find_available_camera()
cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# 使用 YUYV 格式而非 MJPG
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

# 設定影片錄製
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30.0
video_writer = cv2.VideoWriter(video_temp_path, fourcc, fps, (640, 480))
print(f"[Recording] Started recording at {fps} FPS")

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
    # 分離音符名稱和八度數字
    # 處理可能的格式：C4, Ab4, C#4 等
    import re
    match = re.match(r'([A-G][b#]?)(\d+)', note)
    if match:
        name = match.group(1)
        octave = int(match.group(2))
        new_octave = octave + shift
        # 確保八度在合理範圍內 (0-8)
        new_octave = max(0, min(8, new_octave))
        return f"{name}{new_octave}"
    return note

# 依 semitone_shift 調整半音（升或降）
def apply_semitone(note, shift):
    # 半音序列
    chromatic_scale = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    
    import re
    match = re.match(r'([A-G][b#]?)(\d+)', note)
    if not match:
        return note
    
    note_name = match.group(1)
    octave = int(match.group(2))
    
    # 轉換 # 為 b 記號
    if '#' in note_name:
        note_name = note_name.replace('C#', 'Db').replace('D#', 'Eb')\
                             .replace('F#', 'Gb').replace('G#', 'Ab').replace('A#', 'Bb')
    
    # 找到當前音符的索引
    if note_name not in chromatic_scale:
        return note
    
    current_index = chromatic_scale.index(note_name)
    new_index = current_index + shift
    
    # 處理八度跟換
    while new_index < 0:
        new_index += 12
        octave -= 1
    while new_index >= 12:
        new_index -= 12
        octave += 1
    
    # 確保八度在合理範圍
    if octave < 0:
        return note
    if octave > 8:
        return note
    
    return f"{chromatic_scale[new_index]}{octave}"

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
            # 八度手勢（左手）
            # ===============================
            # 計算拇指與其他手指的平均距離
            avg_distance_from_thumb = (abs(thumb - idx) + abs(thumb - mid) + 
                                      abs(thumb - ring) + abs(thumb - pinky)) / 4
            
            # 必須大於閘值才認定為手勢（避免誤觸發）
            GESTURE_DISTANCE_THRESHOLD = 100  # 像素距離閘值
            thumb_is_separated = avg_distance_from_thumb > GESTURE_DISTANCE_THRESHOLD
            
            is_up   = (thumb < idx < mid < ring < pinky) and thumb_is_separated
            is_down = (pinky < ring < mid < idx < thumb) and thumb_is_separated

            if hand_type == "LEFT" and octave_cooldown == 0:

                if is_up:
                    octave_gesture_up += 1
                    octave_gesture_down = 0
                    if octave_gesture_up >= gesture_frames_required:
                        octave_shift = min(octave_shift+1, 3)
                        octave_cooldown = 30
                        octave_gesture_up = 0
                        print(f"[Octave] Shift UP → {octave_shift}")

                elif is_down:
                    octave_gesture_down += 1
                    octave_gesture_up = 0
                    if octave_gesture_down >= gesture_frames_required:
                        octave_shift = max(octave_shift-1, -3)
                        octave_cooldown = 30
                        octave_gesture_down = 0
                        print(f"[Octave] Shift DOWN → {octave_shift}")

                else:
                    octave_gesture_up = 0
                    octave_gesture_down = 0

            # ===============================
            # 半音手勢（右手）
            # ===============================
            if hand_type == "RIGHT" and semitone_cooldown == 0:

                if is_up:
                    semitone_gesture_up += 1
                    semitone_gesture_down = 0
                    if semitone_gesture_up >= gesture_frames_required:
                        semitone_shift = min(semitone_shift+1, 1)  # 最多+1（升半階）
                        semitone_cooldown = 30
                        semitone_gesture_up = 0
                        print(f"[Semitone] Shift UP → {semitone_shift}")

                elif is_down:
                    semitone_gesture_down += 1
                    semitone_gesture_up = 0
                    if semitone_gesture_down >= gesture_frames_required:
                        semitone_shift = max(semitone_shift-1, -1)  # 最多-1（降半階）
                        semitone_cooldown = 30
                        semitone_gesture_down = 0
                        print(f"[Semitone] Shift DOWN → {semitone_shift}")

                else:
                    semitone_gesture_up = 0
                    semitone_gesture_down = 0

            octave_active = (hand_type == "LEFT" and (octave_gesture_up > 0 or octave_gesture_down > 0))
            semitone_active = (hand_type == "RIGHT" and (semitone_gesture_up > 0 or semitone_gesture_down > 0))

            # ===============================
            # 指尖處理 + 播放音符 + 顯色
            # ===============================
            for i, pos in enumerate(fingertips):

                smoothed = smooth_position(pos, prev_pos[i])
                vel = calculate_velocity(smoothed, prev_pos[i])
                prev_pos[i] = smoothed

                # 播放音符（加線光）
                # 提高速度閘值，必須更快按下才會觸發
                if vel > 5 and not fp[i]:
                    fp[i] = True

                    # 根據 octave_shift 和 semitone_shift 計算實際要播放的音符
                    actual_note = apply_octave(hand_notes[i], octave_shift)
                    actual_note = apply_semitone(actual_note, semitone_shift)
                    
                    # 在 ALL_NOTES 中找到對應的音符並播放
                    if actual_note in player.sounds:
                        try:
                            # 播放音符
                            sound = player.sounds[actual_note]
                            sound.play()
                            
                            # 記錄音訊數據用於錄製
                            current_time = time.time() - start_time
                            audio_buffer.append((current_time, actual_note, sound))
                            
                            print(f"[Play] {actual_note}")
                        except pygame.error as e:
                            print(f"[Play Error] {actual_note}: {e}")
                    else:
                        print(f"[Not Found] {actual_note}")

                    glow_timer[i] = 9

                elif vel < -3:
                    fp[i] = False

                # ===============================
                # 顏色決定
                # ===============================
                if glow_timer[i] > 0:
                    color = (0,255,0)     # 線
                    glow_timer[i] -= 1
                elif semitone_active:
                    color = (255,0,255)   # 紫色（右手半音模式）
                elif octave_active:
                    color = (255,0,0)     # 藍色（左手八度模式）
                else:
                    color = (0,0,255)     # 紅色（預設）

                shown_note = apply_octave(hand_notes[i], octave_shift)
                shown_note = apply_semitone(shown_note, semitone_shift)

                cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 10, color, -1)
                cv2.putText(frame, shown_note,
                    (int(smoothed[0]) + 10, int(smoothed[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ===============================
    #  UI 顯示
    # ===============================
    if ENABLE_HORIZONTAL_CALIBRATION and desk_edge_y is not None:
        cv2.line(frame, (0, desk_edge_y), (w, desk_edge_y), (255,255,0), 2)

    cv2.putText(frame, f"Octave Shift: {octave_shift} | Semitone: {semitone_shift:+d}", (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.putText(frame, "Deep Learning & Application (Final Project)", (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    # 寫入影片
    video_writer.write(frame)
    
    cv2.imshow("Virtual Piano", frame)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break

    if octave_cooldown > 0:
        octave_cooldown -= 1
    if semitone_cooldown > 0:
        semitone_cooldown -= 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("[Recording] Processing audio and video...")

# 生成音訊檔案
try:
    # 創建靜音音訊作為基底
    duration = time.time() - start_time
    num_samples = int(duration * audio_sample_rate)
    audio_data = [0] * num_samples  # 靜音
    
    # 將播放的音符混合到音訊中
    for play_time, note_name, sound in audio_buffer:
        if note_name in player.sounds:
            # 獲取音符的原始數據
            sound_array = pygame.sndarray.array(sound)
            start_sample = int(play_time * audio_sample_rate)
            
            # 混合音訊
            for i, sample in enumerate(sound_array):
                if start_sample + i < num_samples:
                    if len(sample.shape) > 0:
                        audio_data[start_sample + i] += int(sample[0] / 10)  # 降低音量避免爆音
    
    # 寫入 WAV 檔案
    with wave.open(audio_temp_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # 單聲道
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(audio_sample_rate)
        for sample in audio_data:
            wav_file.writeframes(struct.pack('<h', max(-32768, min(32767, sample))))
    
    print(f"[Recording] Audio saved to: {audio_temp_path}")
    
    # 使用 ffmpeg 合併影片和音訊，輸出為 MP3 音訊格式
    import subprocess
    
    # 先生成 MP3 音訊檔
    audio_mp3_path = os.path.join(output_dir, f'piano_audio_{timestamp}.mp3')
    mp3_cmd = [
        'ffmpeg', '-y',
        '-i', audio_temp_path,
        '-codec:a', 'libmp3lame',
        '-b:a', '192k',
        audio_mp3_path
    ]
    
    subprocess.run(mp3_cmd, capture_output=True)
    print(f"[Recording] Audio MP3 saved to: {audio_mp3_path}")
    
    # 合併影片和音訊
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', video_temp_path,
        '-i', audio_temp_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-shortest',
        final_video_path
    ]
    
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"[Recording] ✓ Final video saved to: {final_video_path}")
        # 刪除臨時檔案
        os.remove(video_temp_path)
        os.remove(audio_temp_path)
        print("[Recording] Temporary files cleaned up")
    else:
        print(f"[Recording] Error merging video and audio: {result.stderr}")
        print(f"[Recording] Video saved to: {video_temp_path}")
        print(f"[Recording] Audio saved to: {audio_temp_path}")
        
except Exception as e:
    print(f"[Recording] Error processing recording: {e}")
    print(f"[Recording] Video saved to: {video_temp_path}")
