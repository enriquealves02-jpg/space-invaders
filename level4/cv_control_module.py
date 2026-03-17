import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import asyncio
import websockets
import threading
import time
import os
import urllib.request

# WebSocket server address
URI = "ws://localhost:8765"

# Model path
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Shared variable between camera thread and websocket
current_command = None
command_lock = threading.Lock()


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~25MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def detect_gesture(landmarks):
    """
    Detect gesture from hand landmarks (new Tasks API).
    - Hand on left third of screen  -> LEFT
    - Hand on right third of screen -> RIGHT
    - Index finger raised           -> FIRE
    """
    wrist_x = landmarks[0].x
    index_tip_y = landmarks[8].y
    index_base_y = landmarks[5].y

    finger_raised = index_tip_y < index_base_y - 0.05

    if finger_raised:
        return "FIRE"
    elif wrist_x < 0.4:
        return "LEFT"
    elif wrist_x > 0.6:
        return "RIGHT"
    else:
        return None


def camera_loop():
    global current_command

    download_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print("Camera started. Show your hand to control the game.")
    print("  Hand left  -> LEFT")
    print("  Hand right -> RIGHT")
    print("  Index up   -> FIRE")
    print("Press 'q' in the camera window to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        gesture = None
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            gesture = detect_gesture(landmarks)

            # Draw landmarks
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        with command_lock:
            current_command = gesture

        label = gesture if gesture else "No gesture"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.line(frame, (int(w * 0.4), 0), (int(w * 0.4), h), (255, 0, 0), 1)
        cv2.line(frame, (int(w * 0.6), 0), (int(w * 0.6), h), (255, 0, 0), 1)

        cv2.imshow("Space Invaders - CV Control", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            with command_lock:
                current_command = "ENTER"

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


async def send_commands():
    global current_command

    print("Connecting to WebSocket server...")
    async with websockets.connect(URI) as websocket:
        print("Connected! Game control active.")
        last_command = None
        last_sent = 0

        while True:
            with command_lock:
                cmd = current_command

            now = time.time()
            if cmd and (cmd != last_command or now - last_sent > 0.3):
                await websocket.send(cmd)
                print(f"Sent: {cmd}")
                last_command = cmd
                last_sent = now

            await asyncio.sleep(0.05)


def main():
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()
    asyncio.run(send_commands())


if __name__ == "__main__":
    main()
