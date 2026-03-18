# /// script
# dependencies = ["websockets", "opencv-python", "numpy"]
# ///

import asyncio
import cv2
import numpy as np
import websockets
import time

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

async def level3_color():
    cap = cv2.VideoCapture(0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lower_hsv = np.array([0, 120, 80])
    upper_hsv = np.array([15, 255, 255])
    color_name = "Orange (défaut)"

    last_cmd      = None
    last_cmd_time = 0
    cooldown      = 0.15

    async with websockets.connect(WS_URI, ping_interval=None, ping_timeout=None) as ws:
        print("✅ Niveau 3 connecté.")
        print("   Appuyez sur 'c' pour calibrer une nouvelle couleur.")

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                cx, cy = frame_w // 2, frame_h // 2
                roi     = frame[cy-20:cy+20, cx-20:cx+20]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mean_h  = int(np.median(hsv_roi[:, :, 0]))
                margin  = 20
                lower_hsv = np.array([max(0, mean_h - margin), 60, 60])
                upper_hsv = np.array([min(179, mean_h + margin), 255, 255])
                color_name = f"H={mean_h} ±{margin}"
                print(f"   Couleur calibrée : {color_name}")
                cv2.rectangle(frame, (cx-25, cy-25), (cx+25, cy+25), (0, 255, 0), 3)

            if key == ord('q'): break

            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask  = cv2.inRange(hsv, lower_hsv, upper_hsv)
            mask  = cv2.erode(mask,  None, iterations=2)
            mask  = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            command  = None
            info     = []

            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 300:
                    M  = cv2.moments(c)
                    if M["m00"] > 0:
                        cx_ = int(M["m10"] / M["m00"])
                        cy_ = int(M["m01"] / M["m00"])
                        centroid = (cx_, cy_)
                        cv2.drawContours(frame, [c], -1, (0, 255, 100), 2)
                        cv2.circle(frame, centroid, 8, (0, 255, 0), -1)

                        norm_x = cx_ / frame_w
                        norm_y = cy_ / frame_h
                        info   = [f"Centroïde=({cx_},{cy_})",
                                  f"normX={norm_x:.2f}  normY={norm_y:.2f}"]

                        if norm_y < 0.35: command = "FIRE"
                        elif norm_y > 0.70: command = "ENTER"
                        elif norm_x < 0.40: command = "LEFT"
                        else: command = "RIGHT"

            cv2.line(frame, (frame_w//2-15, frame_h//2), (frame_w//2+15, frame_h//2), (200, 200, 200), 1)
            cv2.line(frame, (frame_w//2, frame_h//2-15), (frame_w//2, frame_h//2+15), (200, 200, 200), 1)

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

            draw_hud(frame, [f"NIVEAU 3 — COULEUR : {color_name}",
                              "Gauche=LEFT  Droite=RIGHT",
                              "Haut=FIRE    Bas=ENTER",
                              "'c' pour calibrer"] + info)
            draw_command_banner(frame, command)
            cv2.imshow("CV Controller — Niveau 3", frame)

            await asyncio.sleep(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🚀 Lancement du Niveau 3 (Couleur)... Appuyez sur 'q' pour quitter.")
    asyncio.run(level3_color())