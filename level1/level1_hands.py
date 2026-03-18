# /// script
# dependencies = ["websockets", "opencv-python", "mediapipe", "numpy"]
# ///

import asyncio
import cv2
import numpy as np
import websockets
import time
import mediapipe as mp

WS_URI = "ws://localhost:8765"

def draw_hud(frame, text_lines, color=(0, 255, 200)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 20 + 22 * len(text_lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (8, 18 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def draw_command_banner(frame, command):
    if command:
        colors = {"LEFT": (255, 100, 0), "RIGHT": (0, 100, 255),
                  "FIRE": (0, 255, 80), "ENTER": (255, 255, 0)}
        col = colors.get(command, (255, 255, 255))
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f">>> {command}", (10, h - 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 2, cv2.LINE_AA)

async def level1_hands():
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'accéder à la webcam.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def finger_extended(lm, tip_id, pip_id, mcp_id):
        wrist = np.array([lm[0].x, lm[0].y])
        tip   = np.array([lm[tip_id].x, lm[tip_id].y])
        pip   = np.array([lm[pip_id].x, lm[pip_id].y])
        return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist)

    def count_extended_fingers(lm):
        pairs = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]
        return sum(finger_extended(lm, t, p, m) for t, p, m in pairs)

    def is_fist(lm):
        return count_extended_fingers(lm) <= 1

    def is_index_only_up(lm):
        index_up  = finger_extended(lm, 8,  6,  5)
        middle_up = finger_extended(lm, 12, 10, 9)
        ring_up   = finger_extended(lm, 16, 14, 13)
        pinky_up  = finger_extended(lm, 20, 18, 17)
        return index_up and not middle_up and not ring_up and not pinky_up

    last_cmd      = None
    last_cmd_time = 0
    cooldown      = 0.20

    async with websockets.connect(WS_URI, ping_interval=None, ping_timeout=None) as ws:
        print("✅ Niveau 1 connecté.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            command   = None
            hand_info = []

            if res.multi_hand_landmarks:
                fist_count  = 0
                index_count = 0
                positions_x = []

                for hlm in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
                    lm = hlm.landmark

                    wrist_x = lm[0].x
                    n_ext   = count_extended_fingers(lm)
                    fist    = is_fist(lm)
                    idx     = is_index_only_up(lm)

                    if fist: fist_count += 1
                    if idx: index_count += 1

                    positions_x.append(wrist_x)

                    px = int(wrist_x * frame_w)
                    py = int(lm[0].y * frame_h) - 15
                    geste = "POING" if fist else ("INDEX" if idx else f"{n_ext}doigts")
                    cv2.putText(frame, geste, (px - 20, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
                    hand_info.append(f"x={wrist_x:.2f} | {geste}")

                if index_count >= 1:
                    command = "FIRE"
                elif fist_count >= 1:
                    command = "ENTER"
                elif len(positions_x) == 1:
                    command = "LEFT" if positions_x[0] < 0.45 else "RIGHT"
                elif len(positions_x) >= 2:
                    avg = sum(positions_x) / len(positions_x)
                    command = "LEFT" if avg < 0.45 else "RIGHT"

                mid_x = int(frame_w * 0.45)
                cv2.line(frame, (mid_x, 0), (mid_x, frame_h), (100, 100, 255), 1)

            now = time.time()
            if command:
                if command != last_cmd or (now - last_cmd_time) >= cooldown:
                    try:
                        await ws.send(command)
                        print(f"Envoyé : {command}")
                        last_cmd      = command
                        last_cmd_time = now
                    except Exception as e:
                        print(f"Erreur envoi : {e}")
                        break
            else:
                last_cmd = None

            draw_hud(frame, ["NIVEAU 1 — GESTES MAINS",
                              "Gauche=LEFT | Droite=RIGHT",
                              "Poing=ENTER | Index=FIRE"] + hand_info)
            draw_command_banner(frame, command)
            cv2.imshow("CV Controller — Niveau 1", frame)

            await asyncio.sleep(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🚀 Lancement du Niveau 1 (Mains)... Appuyez sur 'q' pour quitter.")
    asyncio.run(level1_hands())