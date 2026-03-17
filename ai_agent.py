"""
Space Invaders AI Agent - Deep Q-Network (DQN)
Uses screen capture (CNN) + reinforcement learning to learn to play.
GPU accelerated with PyTorch CUDA.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import asyncio
import websockets
import json
import time
import random
from collections import deque
import mss

# ============================================================
# CONFIGURATION
# ============================================================
URI = "ws://localhost:8765"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 5000
LEARNING_RATE = 1e-3
MEMORY_SIZE = 50000
TARGET_UPDATE = 300
FRAME_STACK = 4
IMG_SIZE = 84

# Actions: 0=NOTHING, 1=LEFT, 2=RIGHT, 3=FIRE (same as original model)
ACTIONS = [None, "LEFT", "RIGHT", "FIRE"]
NUM_ACTIONS = len(ACTIONS)


# ============================================================
# CNN MODEL (Deep Q-Network)
# ============================================================
class DQN(nn.Module):
    """
    CNN architecture inspired by the original DQN paper (Atari).
    3 convolutional layers extract visual features from the game screen,
    then 2 fully connected layers map features to Q-values per action.
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # Convolution layers (feature extraction from pixels)
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 9x9 -> 7x7
            nn.ReLU()
        )
        conv_out = self._get_conv_out((FRAME_STACK, IMG_SIZE, IMG_SIZE))
        # Fully connected layers (decision making)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# REPLAY BUFFER (Experience Replay)
# ============================================================
class ReplayBuffer:
    """
    Stores past transitions for training stability.
    The agent samples random mini-batches instead of learning
    from consecutive correlated samples.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states).to(DEVICE),
            torch.tensor(actions, dtype=torch.long).to(DEVICE),
            torch.tensor(rewards, dtype=torch.float32).to(DEVICE),
            torch.cat(next_states).to(DEVICE),
            torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# SCREEN CAPTURE
# ============================================================
def find_game_window():
    """Let user select the game region by drawing a rectangle on a screenshot."""
    print("Taking a screenshot of your screen...")
    print("A window will open: draw a rectangle around the game, then press ENTER.")

    with mss.mss() as sct:
        screenshot = np.array(sct.grab(sct.monitors[1]))
        img = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Resize for display if screen is too large
        h, w = img.shape[:2]
        scale = 1.0
        if w > 1920:
            scale = 1920 / w
            img_display = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            img_display = img.copy()

        # Let user draw rectangle
        roi = cv2.selectROI("Select the game area, then press ENTER", img_display, fromCenter=False)
        cv2.destroyAllWindows()

        x, y, rw, rh = roi
        # Scale back to original resolution
        x = int(x / scale)
        y = int(y / scale)
        rw = int(rw / scale)
        rh = int(rh / scale)

        print(f"Game region selected: x={x}, y={y}, w={rw}, h={rh}")
        return {"left": x, "top": y, "width": rw, "height": rh}


def capture_frame(sct, region):
    """Capture a game frame, convert to grayscale 84x84, normalize to [0,1]."""
    screenshot = np.array(sct.grab(region))
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return normalized


# ============================================================
# REWARD FUNCTION
# ============================================================
def compute_reward(prev_state, curr_state):
    """
    Reward shaping based on game state changes:
    - Killed invader: +10 per invader
    - Lost a life: -50
    - Won the game: +100
    - Lost the game: -100
    - Each step: -0.1 (encourages faster play)
    """
    if curr_state is None or prev_state is None:
        return 0.0

    reward = 0.0

    # Invaders killed - very high reward
    prev_inv = prev_state.get("invaders", 60)
    curr_inv = curr_state.get("invaders", 60)
    if curr_inv < prev_inv:
        killed = prev_inv - curr_inv
        reward += killed * 100.0

    # Lives lost - penalty PER life lost
    prev_lives = prev_state.get("lives", 3)
    curr_lives = curr_state.get("lives", 3)
    if curr_lives < prev_lives:
        lives_lost = prev_lives - curr_lives
        reward -= lives_lost * 1000.0

    # Win - massive reward
    if curr_state.get("win", False):
        reward += 5000.0

    # Lose - penalty based on how many invaders are LEFT (not killing = bad)
    if curr_state.get("lost", False):
        reward -= curr_inv * 20.0  # More remaining = worse penalty

    return reward


# ============================================================
# DQN AGENT
# ============================================================
class Agent:
    def __init__(self):
        # Policy network (the one we train)
        self.policy_net = DQN(NUM_ACTIONS).to(DEVICE)
        # Target network (stable copy for computing targets)
        self.target_net = DQN(NUM_ACTIONS).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps = 0
        self.episode = 0
        self.total_reward = 0
        self.episode_rewards = []

    def select_action(self, state):
        """Epsilon-greedy: explore randomly or exploit learned Q-values."""
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  np.exp(-self.steps / EPSILON_DECAY)

        if random.random() < epsilon:
            return random.randrange(NUM_ACTIONS)  # Random action
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.to(DEVICE))
                return q_values.argmax(dim=1).item()  # Best action

    def train_step(self):
        """One gradient descent step on a random batch from replay buffer."""
        if len(self.memory) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Q(s, a) from policy network
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max Q(s', a') from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        # Huber loss (smooth L1) for stability
        loss = nn.SmoothL1Loss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Sync target network periodically
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path="dqn_model.pth"):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "episode": self.episode,
            "episode_rewards": self.episode_rewards
        }, path)
        print(f"Model saved to {path}")

    def load(self, path="dqn_model.pth"):
        import os
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["policy_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.steps = checkpoint["steps"]
            self.episode = checkpoint["episode"]
            self.episode_rewards = checkpoint.get("episode_rewards", [])
            print(f"Model loaded from {path} (episode {self.episode}, steps {self.steps})")
            return True
        return False


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
async def train():
    print("=" * 60)
    print("  SPACE INVADERS AI AGENT - DQN Training")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    agent = Agent()
    agent.load()  # Resume if model file exists

    # Find game on screen
    game_region = find_game_window()

    print("\nConnecting to WebSocket server...")
    async with websockets.connect(URI, ping_interval=None, close_timeout=5) as websocket:
        print("Connected! Starting training...\n")

        # Press ENTER to start the game
        await websocket.send("ENTER")
        await asyncio.sleep(2)
        await websocket.send("ENTER")
        await asyncio.sleep(1)

        sct = mss.mss()
        frame_buffer = deque(maxlen=FRAME_STACK)
        game_state = None
        prev_game_state = None
        episode_reward = 0
        game_active = True  # False = waiting for restart, ignore done states
        last_action_time = 0
        ACTION_INTERVAL = 0.05  # Act every 50ms (faster reactions)

        # Fill frame buffer
        for _ in range(FRAME_STACK):
            frame = capture_frame(sct, game_region)
            frame_buffer.append(frame)

        print("Training started! The agent is playing...")
        print("Press Ctrl+C to stop and save.\n")

        try:
            while True:
                now = time.time()

                # Read ALL available game state messages (drain buffer)
                while True:
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=0.005)
                        data = json.loads(msg)
                        if data.get("type") == "game_state":
                            if not game_active:
                                if data.get("started") and not data.get("win") and not data.get("lost"):
                                    game_active = True
                                    game_state = data
                                    prev_game_state = None
                            else:
                                prev_game_state = game_state
                                game_state = data
                    except (asyncio.TimeoutError, json.JSONDecodeError):
                        break

                # Skip playing if game hasn't restarted yet
                if not game_active:
                    await asyncio.sleep(0.05)
                    continue

                # Wait for action interval
                if now - last_action_time < ACTION_INTERVAL:
                    await asyncio.sleep(0.01)
                    continue

                last_action_time = now

                # Capture current state
                frame = capture_frame(sct, game_region)
                frame_buffer.append(frame)
                stacked = np.array(list(frame_buffer))
                state_tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)

                # Choose action
                action_idx = agent.select_action(state_tensor)
                action = ACTIONS[action_idx]

                # Send action to game + spam fire
                await websocket.send("FIRE")
                if action is not None and action != "FIRE":
                    await websocket.send(action)
                await websocket.send("FIRE")

                # Compute reward from game state
                reward = compute_reward(prev_game_state, game_state)
                episode_reward += reward

                # Check if episode ended (only if game was started)
                done = False
                if game_state and game_state.get("started") == False and \
                   (game_state.get("win") or game_state.get("lost")):
                    done = True

                # Capture next state
                next_frame = capture_frame(sct, game_region)
                frame_buffer.append(next_frame)
                next_stacked = np.array(list(frame_buffer))
                next_state_tensor = torch.tensor(next_stacked, dtype=torch.float32).unsqueeze(0)

                # Store in replay buffer
                agent.memory.push(state_tensor, action_idx, reward, next_state_tensor, done)

                # Train the CNN
                loss = agent.train_step()
                agent.steps += 1

                # Episode ended (game over or win)
                if done:
                    agent.episode += 1
                    agent.episode_rewards.append(episode_reward)
                    avg_reward = np.mean(agent.episode_rewards[-20:])
                    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                              np.exp(-agent.steps / EPSILON_DECAY)

                    result = "WIN" if game_state.get("win") else "LOST"
                    print(f"Episode {agent.episode:4d} | {result} | "
                          f"Reward: {episode_reward:7.1f} | "
                          f"Avg(20): {avg_reward:7.1f} | "
                          f"Epsilon: {epsilon:.3f} | "
                          f"Steps: {agent.steps} | "
                          f"Memory: {len(agent.memory)}")

                    episode_reward = 0
                    game_state = None
                    prev_game_state = None
                    game_active = False

                    # Save every 50 episodes
                    if agent.episode % 50 == 0:
                        agent.save()

                    # Restart game quickly
                    await asyncio.sleep(0.5)
                    await websocket.send("ENTER")
                    await asyncio.sleep(0.5)
                    await websocket.send("ENTER")

                    # Reset frame buffer
                    for _ in range(FRAME_STACK):
                        f = capture_frame(sct, game_region)
                        frame_buffer.append(f)

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            agent.save()
            print(f"Total episodes: {agent.episode}")
            print(f"Total steps: {agent.steps}")
            if agent.episode_rewards:
                print(f"Best reward: {max(agent.episode_rewards):.1f}")
                print(f"Avg last 20: {np.mean(agent.episode_rewards[-20:]):.1f}")


if __name__ == "__main__":
    asyncio.run(train())
