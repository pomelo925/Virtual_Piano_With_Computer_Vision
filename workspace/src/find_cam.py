import cv2

def find_available_camera(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found available camera at /dev/video{i}")
            cap.release()
            return i
        cap.release()
    print("No available camera found.")
    return None

if __name__ == "__main__":
    find_available_camera()