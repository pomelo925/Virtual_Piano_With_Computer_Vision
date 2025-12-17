import pygame
import os

class SoundPlayer:
    def __init__(self, notes, sound_dir):
        # IMPORTANT: 不要在這裡初始化 mixer！
        # pygame.mixer.init() 必須交由 hybrid_test.py 控制
        
        self.notes = notes
        self.sounds = {}
        self.sound_dir = sound_dir

        print(f"[SoundPlayer] Loading sounds from: {self.sound_dir}")

        for note in self.notes:
            path = os.path.join(self.sound_dir, f"{note}.mp3")
            if os.path.exists(path):
                try:
                    sound = pygame.mixer.Sound(path)
                    sound.set_volume(1.0)  # 設定音量為最大 (0.0 ~ 1.0)
                    self.sounds[note] = sound
                    # print(f"[Loaded] {note} -> {path}")
                except pygame.error as e:
                    print(f"[Error loading] {path}: {e}")
            else:
                print(f"[Missing] {note}.mp3 ({path})")

    def play_note_by_index(self, i):
        if 0 <= i < len(self.notes):
            note = self.notes[i]
            if note in self.sounds:
                try:
                    self.sounds[note].play()
                except pygame.error as e:
                    print(f"[Play Error] {note}: {e}")
