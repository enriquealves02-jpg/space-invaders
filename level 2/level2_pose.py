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

async def level2_pose():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose    = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0)

    last_cmd      = None
    last_cmd_time = 0
    cooldown      = 0.18

    tilt_history  = []
    HISTORY_SIZE  = 6
    TILT_THRESH   = 0.10

    async with websockets.connect(WS_URI, ping_interval=None, ping_timeout=None) as ws:
        print("✅ Niveau 2 connecté.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = pose.process(rgb)

            command = None
            info    = []

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                shoulder_tilt = ls.y - rs.y
                tilt_history.append(shoulder_tilt)
                if len(tilt_history) > HISTORY_SIZE:
                    tilt_history.pop(0)
                smoothed_tilt = sum(tilt_history) / len(tilt_history)

                arm_raised = (lw.y < ls.y - 0.05) or (rw.y < rs.y - 0.05)
                lw_hidden    = lw.visibility < 0.3
                rw_hidden    = rw.visibility < 0.3
                hands_hidden = lw_hidden and rw_hidden

                info = [f"Tilt brut={shoulder_tilt:+.2f}  lissé={smoothed_tilt:+.2f}",
                        f"Seuil=±{TILT_THRESH}",
                        f"Bras levé={arm_raised}",
                        f"Mains cachées={hands_hidden} (L={lw.visibility:.2f} R={rw.visibility:.2f})"]

                h, w = frame.shape[:2]
                bar_cx, bar_y = w // 2, h - 55
                cv2.line(frame, (bar_cx - 80, bar_y), (bar_cx + 80, bar_y), (80, 80, 80), 2)
                neutral_px = int(TILT_THRESH * 400)
                cv2.rectangle(frame, (bar_cx - neutral_px, bar_y - 5),
                              (bar_cx + neutral_px, bar_y + 5), (60, 60, 60), -1)
                cursor_x = max(bar_cx - 80, min(bar_cx + 80, bar_cx + int(smoothed_tilt * 400)))
                col = (0, 200, 255) if abs(smoothed_tilt) < TILT_THRESH else (0, 80, 255)
                cv2.circle(frame, (cursor_x, bar_y), 8, col, -1)
                cv2.line(frame, (bar_cx, bar_y - 8), (bar_cx, bar_y + 8), (200, 200, 200), 1)

                if hands_hidden: command = "ENTER"
                elif arm_raised: command = "FIRE"
                elif smoothed_tilt > TILT_THRESH: command = "RIGHT"
                elif smoothed_tilt < -TILT_THRESH: command = "LEFT"

            now = time.time()
            if command and (command != last_cmd or now - last_cmd_time > cooldown):
                await ws.send(command)
                last_cmd = command
                last_cmd_time = now

            draw_hud(frame, ["NIVEAU 2 — POSE CORPORELLE",
                              "Inclinaison=LEFT/RIGHT",
                              "Bras levé=FIRE",
                              "Mains cachées=ENTER"] + info)
            draw_command_banner(frame, command)
            cv2.imshow("CV Controller — Niveau 2", frame)

            await asyncio.sleep(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🚀 Lancement du Niveau 2 (Pose)... Appuyez sur 'q' pour quitter.")
    asyncio.run(level2_pose())