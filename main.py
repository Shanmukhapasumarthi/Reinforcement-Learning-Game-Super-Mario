import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
import pygame

from agent import Agent
from wrappers import apply_wrappers
from utils import get_current_date_time_string

# ----------------------------
# 🔧 CONFIGURATIONS
# ----------------------------
ENV_NAME = 'SuperMarioBros-1-1-v0'
NUM_OF_EPISODES = 25                 # extended for demo
VIDEO_INTERVAL = 5
CKPT_SAVE_INTERVAL = 10

# Fast-forward settings
SKIP_FAST_EPISODES = 15              # fast episodes 1 → 15
FAST_FPS = 200
NORMAL_FPS = 60

# ----------------------------
# ⚙️ DEVICE CHECK
# ----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple MPS backend for acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚙️ Using CPU")

# ----------------------------
# 💾 MODEL PATH SETUP
# ----------------------------
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)
writer = SummaryWriter("runs/mario_training")

# ----------------------------
# 🧠 SELECT MODE
# ----------------------------
mode = input("Enter mode: [auto/manual] → ").strip().lower()

# ----------------------------
# 🕹️ ENVIRONMENT SETUP
# ----------------------------
env = gym_super_mario_bros.make(
    ENV_NAME,
    render_mode='human',
    apply_api_compatibility=True
)
if mode == "manual":
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
else:
    env = JoypadSpace(env, RIGHT_ONLY)

env = apply_wrappers(env)
env = RecordVideo(env, video_folder="videos",
                  episode_trigger=lambda ep: ep % VIDEO_INTERVAL == 0)
print(f"🎥 Recording gameplay every {VIDEO_INTERVAL} episodes")

# ----------------------------
# 🎮 MANUAL MODE
# ----------------------------
if mode == "manual":
    print("\n🕹️ MANUAL MODE ACTIVE")
    print("Hold Arrow Keys + A/S to move or jump. Press ESC or Q to quit.\n")

    pygame.init()
    manual_clock = pygame.time.Clock()

    state, _ = env.reset()
    done = False

    key_to_action = {
        "none": 0,
        "right": 1,
        "right_jump": 2,
        "right_run": 3,
        "right_run_jump": 4,
        "jump": 5,
        "left": 6
    }

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

        pressed = pygame.key.get_pressed()
        action = 0

        if pressed[pygame.K_q]:
            done = True
        elif pressed[pygame.K_RIGHT] and pressed[pygame.K_a] and pressed[pygame.K_s]:
            action = 4
        elif pressed[pygame.K_RIGHT] and pressed[pygame.K_a]:
            action = 2
        elif pressed[pygame.K_RIGHT] and pressed[pygame.K_s]:
            action = 3
        elif pressed[pygame.K_RIGHT]:
            action = 1
        elif pressed[pygame.K_LEFT]:
            action = 6
        elif pressed[pygame.K_a]:
            action = 5

        _, _, done, truncated, info = env.step(action)
        if truncated:
            done = True

        env.render()
        manual_clock.tick(60)

    env.close()
    pygame.quit()
    exit()

# ----------------------------
# 🤖 AGENT SETUP (Auto Mode)
# ----------------------------
agent = Agent(input_dims=env.observation_space.shape,
              num_actions=env.action_space.n)
agent.lr = 0.0002
agent.epsilon_decay = 0.995

# ----------------------------
# 📊 LIVE STATS WINDOW
# ----------------------------
pygame.init()
stats_window = pygame.display.set_mode((420, 300))
pygame.display.set_caption("Live Training Stats")
font = pygame.font.SysFont("Times New Roman", 22)
clock = pygame.time.Clock()

# ----------------------------
# 📈 TRAINING LOOP
# ----------------------------
rewards = []

for episode in range(NUM_OF_EPISODES):
    print(f"\n🚀 Episode {episode + 1}/{NUM_OF_EPISODES}")

    state, _ = env.reset()
    done = False
    total_reward = 0

    # FAST-FORWARD EPISODES
    if (episode + 1) <= SKIP_FAST_EPISODES:
        fps = FAST_FPS
    else:
        fps = NORMAL_FPS

    while not done:

        # Allow safe closing of stats window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                exit()

        action = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)

        # Reward shaping
        reward += info.get('x_pos', 0) * 0.01
        if done and info.get('life', 1) < 3:
            reward -= 50

        total_reward += reward

        agent.store_in_memory(state, action, reward, new_state, done)
        agent.learn()
        state = new_state

        if truncated:
            done = True

        # LIVE STATS WINDOW UPDATE
        stats_window.fill((20, 20, 20))
        lines = [
            f"Episode: {episode + 1}/{NUM_OF_EPISODES}",
            f"Step Count: {agent.learn_step_counter}",
            f"Reward (Step): {reward:.2f}",
            f"Total Reward: {total_reward:.2f}",
            f"Epsilon: {agent.epsilon:.3f}",
            f"Distance (x_pos): {info.get('x_pos', 0)}",
            f"Lives Left: {info.get('life', 3)}"
        ]

        y = 20
        for line in lines:
            txt = font.render(line, True, (255, 255, 255))
            stats_window.blit(txt, (20, y))
            y += 30

        pygame.display.update()
        clock.tick(fps)   # FAST or NORMAL FPS

    rewards.append(total_reward)
    writer.add_scalar("Reward/Total", total_reward, episode)

    print(f"🏆 Episode Reward: {total_reward:.2f}")

    if (episode + 1) % CKPT_SAVE_INTERVAL == 0:
        ckpt = os.path.join(model_path, f"model_{episode + 1}.pt")
        agent.save_model(ckpt)
        print(f"💾 Model saved: {ckpt}")

# ----------------------------
# 📉 FINAL REWARD GRAPH
# ----------------------------
plt.figure(figsize=(10, 5))
plt.title("Final Reward Trend")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.plot(rewards, color='green')
plt.grid(True)
plt.show()

env.close()
pygame.quit()
writer.close()

print("✅ Training finished! Models saved in:", model_path)
