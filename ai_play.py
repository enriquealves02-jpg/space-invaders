"""
Space Invaders AI Player
Loads a trained DQN model and plays the game autonomously.
No training, just playing.
"""

import torch
import numpy as np
import cv2
import asyncio
import websockets
import json
import time
from collections import deque
import mss

# Reuse model and config from ai_agent
from ai_agent import DQN, DEVICE, NUM_ACTIONS, ACTIONS, FRAME_STACK, IMG_SIZE

URI = "ws://localhost:8765"


def find_game_window():
    """Let user select the game region by drawing a rectangle."""
    print("A window will open: draw a rectangle around the game, then press ENTER.")

    with mss.mss() as sct:
        screenshot = np.array(sct.grab(sct.monitors[1]))
        img = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        scale = 1.0
        if w > 1920:
            scale = 1920 / w
            img_display = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            img_display = img.copy()

        roi = cv2.selectROI("Select the game area, then press ENTER", img_display, fromCenter=False)
        cv2.destroyAllWindows()

        x, y, rw, rh = roi
        x = int(x / scale)
        y = int(y / scale)
        rw = int(rw / scale)
        rh = int(rh / scale)

        print(f"Game region: x={x}, y={y}, w={rw}, h={rh}")
        return {"left": x, "top": y, "width": rw, "height": rh}


def capture_frame(sct, region):
    screenshot = np.array(sct.grab(region))
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return resized.astype(np.float32) / 255.0


async def play():
    print("=" * 60)
    print("  SPACE INVADERS AI PLAYER (no training)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Load trained model
    model = DQN(NUM_ACTIONS).to(DEVICE)
    checkpoint = torch.load("dqn_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["policy_net"])
    model.eval()
    print(f"Model loaded (trained for {checkpoint['episode']} episodes, {checkpoint['steps']} steps)")

    game_region = find_game_window()

    print("\nConnecting to WebSocket server...")
    async with websockets.connect(URI, ping_interval=None, close_timeout=5) as websocket:
        print("Connected! AI is playing...\n")

        await websocket.send("ENTER")
        await asyncio.sleep(2)
        await websocket.send("ENTER")
        await asyncio.sleep(1)

        sct = mss.mss()
        frame_buffer = deque(maxlen=FRAME_STACK)
        game_state = None
        games_played = 0
        wins = 0

        for _ in range(FRAME_STACK):
            frame = capture_frame(sct, game_region)
            frame_buffer.append(frame)

        print("Watching the AI play! Press Ctrl+C to stop.\n")

        try:
            while True:
                # Read game state
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    data = json.loads(msg)
                    if data.get("type") == "game_state":
                        game_state = data
                except asyncio.TimeoutError:
                    pass

                # Capture and decide
                frame = capture_frame(sct, game_region)
                frame_buffer.append(frame)
                stacked = np.array(list(frame_buffer))
                state_tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    q_values = model(state_tensor.to(DEVICE))
                    action_idx = q_values.argmax(dim=1).item()

                action = ACTIONS[action_idx]
                # Always fire first (spam fire like training)
                await websocket.send("FIRE")
                if action is not None and action != "FIRE":
                    await websocket.send(action)
                await websocket.send("FIRE")

                # Game over detection
                if game_state and (game_state.get("win") or game_state.get("lost")):
                    games_played += 1
                    if game_state.get("win"):
                        wins += 1
                        print(f"Game {games_played}: WIN!  (Win rate: {wins}/{games_played} = {100*wins/games_played:.0f}%)")
                    else:
                        print(f"Game {games_played}: LOST  (Win rate: {wins}/{games_played} = {100*wins/games_played:.0f}%)")

                    game_state = None
                    await asyncio.sleep(3)
                    await websocket.send("ENTER")
                    await asyncio.sleep(2)

                    # Drain stale messages and wait for game to restart
                    for _ in range(50):
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            data = json.loads(msg)
                            if data.get("type") == "game_state":
                                if data.get("started") and not data.get("win") and not data.get("lost"):
                                    game_state = None
                                    break
                        except asyncio.TimeoutError:
                            pass

                    for _ in range(FRAME_STACK):
                        f = capture_frame(sct, game_region)
                        frame_buffer.append(f)

                await asyncio.sleep(0.05)

        except KeyboardInterrupt:
            print(f"\n\nDone! Played {games_played} games, won {wins}.")


if __name__ == "__main__":
    asyncio.run(play())
