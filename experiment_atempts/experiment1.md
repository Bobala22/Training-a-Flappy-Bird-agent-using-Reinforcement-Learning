# Experiment 1

## Observations:
- not taking in consideration last 2 states, we were only working on the current state of the game
- lr = 0.001
- the reward when the bird succesfully passed through a pipe would be 20
- too big convolutions

## Hyperparameters:
- learning rate = 0,005
- batch_size = 64
- replay buffer capacity = 50000
- target network update every 30 eps.
- epsilon decay = 0,995

## Code:
```python
import flappy_bird_gymnasium
import gymnasium as gym
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import cv2
from datetime import datetime
import random
from collections import deque 
import matplotlib.pyplot as plt

##########################################################################
# 1) Utility Function: Max Pooling
##########################################################################
def max_pooling(image, pool_size=2, stride=2):
    """Apply max pooling to reduce image dimensions."""
    height, width = image.shape
    pool_height = (height - pool_size) // stride + 1
    pool_width = (width - pool_size) // stride + 1
    
    pooled = np.zeros((pool_height, pool_width))
    
    for i in range(pool_height):
        for j in range(pool_width):
            i_start = i * stride
            i_end = i_start + pool_size
            j_start = j * stride
            j_end = j_start + pool_size
            pool_region = image[i_start:i_end, j_start:j_end]
            pooled[i, j] = np.max(pool_region)
    
    return pooled

##########################################################################
# 2) Neural Network Definition
##########################################################################
class FlappyBirdCNN(nn.Module):
    def __init__(self):
        super(FlappyBirdCNN, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # NOTE: The next line assumes the final output of the conv layers is 64 x 7 x 7.
        # If your actual input image size after cropping/pooling is different, 
        # this dimension may need to be adjusted. 
        self.fc_layers = nn.Sequential(
            nn.Linear(1728, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # 2 actions: flap or not flap
        )
    
    def forward(self, x):
        # x should be [batch_size, 1, height, width]
        x = self.conv_layers(x)               # => [batch_size, 64, H', W']
        x = x.view(x.size(0), -1)             # => [batch_size, 64*H'*W']
        return self.fc_layers(x)              # => [batch_size, 2]

##########################################################################
# 3) Frame Preprocessing (Sobel + Thresholding + Pooling)
##########################################################################
count = 0
def preprocess_frame(frame):
    global count
    # Convert to grayscale and normalize
    rbg_image = frame[:-106, :, :]  # Crop the bottom part of the image
    rbg_image = rbg_image[:, 60:, :] # Further crop from the left
    grayscale_image = np.dot(rbg_image[...,:3], [0.2989, 0.5870, 0.1140])

    # Sobel edge detection
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    edges_x = convolve2d(grayscale_image, sobel_x, mode='same', boundary='wrap')
    edges_y = convolve2d(grayscale_image, sobel_y, mode='same', boundary='wrap')
    
    edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    edge_magnitude = (edge_magnitude / edge_magnitude.max() * 255).astype(np.uint8)
    
    # Apply max pooling
    pooled_image = max_pooling(edge_magnitude, pool_size=4, stride=4)
    
    # Normalize and threshold
    normalized_image = (pooled_image - pooled_image.min()) / (pooled_image.max() - pooled_image.min())
    threshold = 0.45
    binary_image = (normalized_image > threshold).astype(np.uint8)
    display_image = (binary_image * 255).astype(np.uint8)

    # Attempt to make the bird body fully white
    bird_body_kernel = np.array([[0.00, 0.47, 0.47, 0.47, 0.00],
                                 [0.47, 0.47, 0.47, 0.47, 0.47],
                                 [0.47, 0.47, 0.47, 0.47, 0.47],
                                 [0.47, 0.47, 0.47, 0.47, 0.47],
                                 [0.00, 0.47, 0.47, 0.47, 0.00]])
    
    display_image = convolve2d(display_image, bird_body_kernel, mode='same', boundary='wrap')
    display_image = (display_image * 255).astype(np.uint8)
    display_image[display_image > 0] = 255

    # Convert to PyTorch tensor as [1, 1, H, W] (batch=1, channels=1)
    tensor_image = torch.FloatTensor(display_image).unsqueeze(0).unsqueeze(0)  # => [1, 1, height, width]

    return tensor_image

##########################################################################
# 4) Saving and Loading
##########################################################################
def save_model(model, target_model, optimizer, epoch, epsilon, path='models'):
    """Save model, target_model, optimizer state, plus epoch and epsilon."""
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save({
        'epoch': epoch,
        'epsilon': epsilon,
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # If you want to store replay buffer, do so here (be mindful of memory size)
        # 'replay_buffer': replay_buffer
    }, f'{path}/flappy_bird_model_epoch_{epoch}.pth')

def load_model(model, target_model, optimizer, path_to_model):
    """Load model, target_model, optimizer state, plus epoch and epsilon."""
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_model.load_state_dict(checkpoint['target_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    epsilon = checkpoint['epsilon']
    # If you had a replay buffer saved:
    # replay_buffer = checkpoint['replay_buffer']
    # return model, target_model, optimizer, start_epoch, epsilon, replay_buffer
    
    return model, target_model, optimizer, start_epoch, epsilon

def calculate_iterations_per_minute(start_time, iteration_count):
    """Calculate the number of iterations per minute."""
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        iterations_per_minute = (iteration_count * 60) / elapsed_time
        return iterations_per_minute
    return 0

##########################################################################
# 5) Replay Buffer
##########################################################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Each state is shape [1, 1, H, W], so torch.cat(...) => [batch_size, 1, H, W]
        return (
            torch.cat(states, dim=0),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.cat(next_states, dim=0),
            torch.tensor(dones, dtype=torch.bool),
        )
    
    def empty(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

##########################################################################
# 6) Main Training Function
##########################################################################
def train_model(env, start_epoch=0, path_to_model=None):
    """
    Train the Flappy Bird model. If path_to_model is provided, 
    it will attempt to load from that checkpoint and resume training.
    """
    num_epochs = 50000
    save_interval = 20
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = FlappyBirdCNN().to(device)
    target_model = FlappyBirdCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Hyperparameters
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    gamma = 0.99
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=50000)

    # If a model path is provided, load it
    if path_to_model is not None and os.path.exists(path_to_model):
        print(f"Loading model from {path_to_model}")
        model, target_model, optimizer, loaded_epoch, epsilon = load_model(
            model, target_model, optimizer, path_to_model
        )
        start_epoch = loaded_epoch  # Resume from that epoch
        print(f"Resuming training from epoch {start_epoch} with epsilon={epsilon:.4f}")

    # Sync target model with main model
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    criterion = nn.MSELoss()

    for epoch in range(start_epoch, num_epochs):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        # Initialize frame collection for potential recording
        frames = []
        should_record = (epoch % save_interval == 0)

        # Get initial state
        state = preprocess_frame(env.render()).to(device)

        while not done:
            # If we're recording this epoch, collect frames
            if should_record:
                rgb_frame = env.render()
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                frames.append(bgr_frame)

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # model(state) => shape [1, 2]; take argmax
                    action = torch.argmax(model(state)).item()

            # Step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            if reward == 1:
                print("The bird got through a pipe")
                reward = reward * 20

            done = terminated or truncated
            total_reward += reward

            # Preprocess the next state
            next_state = preprocess_frame(env.render()).to(device)

            # Store the transition in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Training step
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = states.to(device)        # [batch_size, 1, H, W]
                actions = actions.to(device)      # [batch_size]
                rewards = rewards.to(device)      # [batch_size]
                next_states = next_states.to(device)
                dones = dones.to(device)

                # Q(s', a') for all a'
                with torch.no_grad():
                    max_next_q_values = target_model(next_states).max(dim=1)[0]
                    gamma_t = torch.tensor(gamma).to(device)
                    targets = rewards + gamma_t * max_next_q_values * (~dones)

                # Q(s, a) for the actions we took
                current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()


                loss = criterion(current_q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # lets clear the replay buffer after this
                replay_buffer.empty()

        # Save a video if this epoch was recorded
        if should_record and frames:
            # Create videos directory if it doesn't exist
            os.makedirs('./videos', exist_ok=True)
            
            video_path = f'./videos/flappy_bird_epoch_{epoch}.mp4'
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Saved video for epoch {epoch} to {video_path}")

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Periodically update target network
        if (epoch + 1) % 30 == 0:
            target_model.load_state_dict(model.state_dict())

        # Periodically save model
        if (epoch + 1) % save_interval == 0:
            save_model(
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                epoch=epoch + 1,  # next epoch
                epsilon=epsilon
            )
            print(f"Epoch {epoch + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    return model

##########################################################################
# 7) Main Entry Point
##########################################################################
def main():
    # Make sure you have the correct version of flappy_bird_gymnasium
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    # Provide path_to_model to resume training from a saved checkpoint
    path_to_model = None
    train_model(env, start_epoch=0, path_to_model=path_to_model)
    env.close()

if __name__ == "__main__":
    main()
```

## Results:
```txt
Saved video for epoch 0 to ./videos/flappy_bird_epoch_0.mp4
Epoch 20, Total Reward: -6.299999999999999, Epsilon: 0.9046
Saved video for epoch 20 to ./videos/flappy_bird_epoch_20.mp4
Epoch 40, Total Reward: -8.7, Epsilon: 0.8183
Saved video for epoch 40 to ./videos/flappy_bird_epoch_40.mp4
Epoch 60, Total Reward: -6.899999999999999, Epsilon: 0.7403
Saved video for epoch 60 to ./videos/flappy_bird_epoch_60.mp4
Epoch 80, Total Reward: -6.299999999999999, Epsilon: 0.6696
Saved video for epoch 80 to ./videos/flappy_bird_epoch_80.mp4
Epoch 100, Total Reward: -6.899999999999999, Epsilon: 0.6058
Saved video for epoch 100 to ./videos/flappy_bird_epoch_100.mp4
Epoch 120, Total Reward: -8.7, Epsilon: 0.5480
Saved video for epoch 120 to ./videos/flappy_bird_epoch_120.mp4
Epoch 140, Total Reward: -7.499999999999998, Epsilon: 0.4957
Saved video for epoch 140 to ./videos/flappy_bird_epoch_140.mp4
Epoch 160, Total Reward: -4.499999999999998, Epsilon: 0.4484
Saved video for epoch 160 to ./videos/flappy_bird_epoch_160.mp4
Epoch 180, Total Reward: -8.7, Epsilon: 0.4057
Saved video for epoch 180 to ./videos/flappy_bird_epoch_180.mp4
Epoch 200, Total Reward: -8.7, Epsilon: 0.3670
Saved video for epoch 200 to ./videos/flappy_bird_epoch_200.mp4
Epoch 220, Total Reward: -9.299999999999999, Epsilon: 0.3320
Saved video for epoch 220 to ./videos/flappy_bird_epoch_220.mp4
Epoch 240, Total Reward: -0.8999999999999986, Epsilon: 0.3003
Saved video for epoch 240 to ./videos/flappy_bird_epoch_240.mp4
Epoch 260, Total Reward: -0.8999999999999986, Epsilon: 0.2716
Saved video for epoch 260 to ./videos/flappy_bird_epoch_260.mp4
Epoch 280, Total Reward: -3.299999999999998, Epsilon: 0.2457
Saved video for epoch 280 to ./videos/flappy_bird_epoch_280.mp4
Epoch 300, Total Reward: -1.4999999999999982, Epsilon: 0.2223
Saved video for epoch 300 to ./videos/flappy_bird_epoch_300.mp4
Epoch 320, Total Reward: -3.899999999999998, Epsilon: 0.2011
Saved video for epoch 320 to ./videos/flappy_bird_epoch_320.mp4
Epoch 340, Total Reward: -6.299999999999999, Epsilon: 0.1819
Saved video for epoch 340 to ./videos/flappy_bird_epoch_340.mp4
Epoch 360, Total Reward: -0.09999999999999787, Epsilon: 0.1646
Saved video for epoch 360 to ./videos/flappy_bird_epoch_360.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 380, Total Reward: -0.8999999999999986, Epsilon: 0.1489
Saved video for epoch 380 to ./videos/flappy_bird_epoch_380.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 400, Total Reward: -0.8999999999999986, Epsilon: 0.1347
Saved video for epoch 400 to ./videos/flappy_bird_epoch_400.mp4
Epoch 420, Total Reward: -8.099999999999998, Epsilon: 0.1218
Saved video for epoch 420 to ./videos/flappy_bird_epoch_420.mp4
The bird got through a pipe
Epoch 440, Total Reward: -1.8999999999999986, Epsilon: 0.1102
Saved video for epoch 440 to ./videos/flappy_bird_epoch_440.mp4
The bird got through a pipe
Epoch 460, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 460 to ./videos/flappy_bird_epoch_460.mp4
Epoch 480, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 480 to ./videos/flappy_bird_epoch_480.mp4
The bird got through a pipe
Epoch 500, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 500 to ./videos/flappy_bird_epoch_500.mp4
Epoch 520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 520 to ./videos/flappy_bird_epoch_520.mp4
Epoch 540, Total Reward: 0.5000000000000018, Epsilon: 0.1000
Saved video for epoch 540 to ./videos/flappy_bird_epoch_540.mp4
The bird got through a pipe
Epoch 560, Total Reward: 1.0000000000000018, Epsilon: 0.1000
Saved video for epoch 560 to ./videos/flappy_bird_epoch_560.mp4
The bird got through a pipe
Epoch 580, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 580 to ./videos/flappy_bird_epoch_580.mp4
The bird got through a pipe
Epoch 600, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 600 to ./videos/flappy_bird_epoch_600.mp4
The bird got through a pipe
Epoch 620, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 620 to ./videos/flappy_bird_epoch_620.mp4
Epoch 640, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 640 to ./videos/flappy_bird_epoch_640.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 660, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 660 to ./videos/flappy_bird_epoch_660.mp4
The bird got through a pipe
Epoch 680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 680 to ./videos/flappy_bird_epoch_680.mp4
The bird got through a pipe
Epoch 700, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 700 to ./videos/flappy_bird_epoch_700.mp4
Epoch 720, Total Reward: 2.1000000000000014, Epsilon: 0.1000
Saved video for epoch 720 to ./videos/flappy_bird_epoch_720.mp4
The bird got through a pipe
Epoch 740, Total Reward: -1.9999999999999987, Epsilon: 0.1000
Saved video for epoch 740 to ./videos/flappy_bird_epoch_740.mp4
Epoch 760, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 760 to ./videos/flappy_bird_epoch_760.mp4
The bird got through a pipe
Epoch 780, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 780 to ./videos/flappy_bird_epoch_780.mp4
Epoch 800, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 800 to ./videos/flappy_bird_epoch_800.mp4
The bird got through a pipe
Epoch 820, Total Reward: 1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 820 to ./videos/flappy_bird_epoch_820.mp4
Epoch 840, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 840 to ./videos/flappy_bird_epoch_840.mp4
Epoch 860, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 860 to ./videos/flappy_bird_epoch_860.mp4
The bird got through a pipe
Epoch 880, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 880 to ./videos/flappy_bird_epoch_880.mp4
Epoch 900, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 900 to ./videos/flappy_bird_epoch_900.mp4
Epoch 920, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 920 to ./videos/flappy_bird_epoch_920.mp4
Epoch 940, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 940 to ./videos/flappy_bird_epoch_940.mp4
Epoch 960, Total Reward: -0.3999999999999977, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 960 to ./videos/flappy_bird_epoch_960.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 980 to ./videos/flappy_bird_epoch_980.mp4
Epoch 1000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1000 to ./videos/flappy_bird_epoch_1000.mp4
The bird got through a pipe
Epoch 1020, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 1020 to ./videos/flappy_bird_epoch_1020.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 1040, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 1040 to ./videos/flappy_bird_epoch_1040.mp4
The bird got through a pipe
Epoch 1060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1060 to ./videos/flappy_bird_epoch_1060.mp4
Epoch 1080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1080 to ./videos/flappy_bird_epoch_1080.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 1100, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1100 to ./videos/flappy_bird_epoch_1100.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 1120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1120 to ./videos/flappy_bird_epoch_1120.mp4
The bird got through a pipe
Epoch 1140, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 1140 to ./videos/flappy_bird_epoch_1140.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 1160, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 1160 to ./videos/flappy_bird_epoch_1160.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 1180, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1180 to ./videos/flappy_bird_epoch_1180.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 1200, Total Reward: 23.700000000000006, Epsilon: 0.1000
Saved video for epoch 1200 to ./videos/flappy_bird_epoch_1200.mp4
The bird got through a pipe
Epoch 1220, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 1220 to ./videos/flappy_bird_epoch_1220.mp4
The bird got through a pipe
Epoch 1240, Total Reward: -3.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1240 to ./videos/flappy_bird_epoch_1240.mp4
The bird got through a pipe
Epoch 1260, Total Reward: 20.799999999999997, Epsilon: 0.1000
Saved video for epoch 1260 to ./videos/flappy_bird_epoch_1260.mp4
Epoch 1280, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 1280 to ./videos/flappy_bird_epoch_1280.mp4
Epoch 1300, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 1300 to ./videos/flappy_bird_epoch_1300.mp4
Epoch 1320, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 1320 to ./videos/flappy_bird_epoch_1320.mp4
Epoch 1340, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1340 to ./videos/flappy_bird_epoch_1340.mp4
The bird got through a pipe
Epoch 1360, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1360 to ./videos/flappy_bird_epoch_1360.mp4
The bird got through a pipe
Epoch 1380, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 1380 to ./videos/flappy_bird_epoch_1380.mp4
Epoch 1400, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 1400 to ./videos/flappy_bird_epoch_1400.mp4
Epoch 1420, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 1420 to ./videos/flappy_bird_epoch_1420.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 1440, Total Reward: 17.600000000000005, Epsilon: 0.1000
Saved video for epoch 1440 to ./videos/flappy_bird_epoch_1440.mp4
Epoch 1460, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1460 to ./videos/flappy_bird_epoch_1460.mp4
Epoch 1480, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 1480 to ./videos/flappy_bird_epoch_1480.mp4
Epoch 1500, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 1500 to ./videos/flappy_bird_epoch_1500.mp4
The bird got through a pipe
Epoch 1520, Total Reward: 20.5, Epsilon: 0.1000
Saved video for epoch 1520 to ./videos/flappy_bird_epoch_1520.mp4
Epoch 1540, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 1540 to ./videos/flappy_bird_epoch_1540.mp4
Epoch 1560, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 1560 to ./videos/flappy_bird_epoch_1560.mp4
Epoch 1580, Total Reward: 2.799999999999997, Epsilon: 0.1000
Saved video for epoch 1580 to ./videos/flappy_bird_epoch_1580.mp4
Epoch 1600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1600 to ./videos/flappy_bird_epoch_1600.mp4
The bird got through a pipe
Epoch 1620, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 1620 to ./videos/flappy_bird_epoch_1620.mp4
Epoch 1640, Total Reward: 1.7763568394002505e-15, Epsilon: 0.1000
Saved video for epoch 1640 to ./videos/flappy_bird_epoch_1640.mp4
Epoch 1660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1660 to ./videos/flappy_bird_epoch_1660.mp4
Epoch 1680, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 1680 to ./videos/flappy_bird_epoch_1680.mp4
The bird got through a pipe
Epoch 1700, Total Reward: 22.500000000000007, Epsilon: 0.1000
Saved video for epoch 1700 to ./videos/flappy_bird_epoch_1700.mp4
Epoch 1720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1720 to ./videos/flappy_bird_epoch_1720.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 1740, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1740 to ./videos/flappy_bird_epoch_1740.mp4
The bird got through a pipe
Epoch 1760, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 1760 to ./videos/flappy_bird_epoch_1760.mp4
The bird got through a pipe
Epoch 1780, Total Reward: -5.099999999999999, Epsilon: 0.1000
Saved video for epoch 1780 to ./videos/flappy_bird_epoch_1780.mp4
Epoch 1800, Total Reward: -5.699999999999999, Epsilon: 0.1000
Saved video for epoch 1800 to ./videos/flappy_bird_epoch_1800.mp4
Epoch 1820, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 1820 to ./videos/flappy_bird_epoch_1820.mp4
Epoch 1840, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 1840 to ./videos/flappy_bird_epoch_1840.mp4
Epoch 1860, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 1860 to ./videos/flappy_bird_epoch_1860.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 1880, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 1880 to ./videos/flappy_bird_epoch_1880.mp4
Epoch 1900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1900 to ./videos/flappy_bird_epoch_1900.mp4
Epoch 1920, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1920 to ./videos/flappy_bird_epoch_1920.mp4
The bird got through a pipe
Epoch 1940, Total Reward: -0.9999999999999987, Epsilon: 0.1000
Saved video for epoch 1940 to ./videos/flappy_bird_epoch_1940.mp4
The bird got through a pipe
Epoch 1960, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 1960 to ./videos/flappy_bird_epoch_1960.mp4
The bird got through a pipe
Epoch 1980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 1980 to ./videos/flappy_bird_epoch_1980.mp4
Epoch 2000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2000 to ./videos/flappy_bird_epoch_2000.mp4
The bird got through a pipe
Epoch 2020, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2020 to ./videos/flappy_bird_epoch_2020.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 2040, Total Reward: 22.900000000000006, Epsilon: 0.1000
Saved video for epoch 2040 to ./videos/flappy_bird_epoch_2040.mp4
Epoch 2060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2060 to ./videos/flappy_bird_epoch_2060.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 2080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2080 to ./videos/flappy_bird_epoch_2080.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 2100, Total Reward: -1.3999999999999986, Epsilon: 0.1000
Saved video for epoch 2100 to ./videos/flappy_bird_epoch_2100.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 2120, Total Reward: -0.9999999999999987, Epsilon: 0.1000
Saved video for epoch 2120 to ./videos/flappy_bird_epoch_2120.mp4
The bird got through a pipe
Epoch 2140, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 2140 to ./videos/flappy_bird_epoch_2140.mp4
Epoch 2160, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 2160 to ./videos/flappy_bird_epoch_2160.mp4
Epoch 2180, Total Reward: 1.1000000000000019, Epsilon: 0.1000
Saved video for epoch 2180 to ./videos/flappy_bird_epoch_2180.mp4
The bird got through a pipe
Epoch 2200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2200 to ./videos/flappy_bird_epoch_2200.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 2220, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 2220 to ./videos/flappy_bird_epoch_2220.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 2240, Total Reward: 3.599999999999996, Epsilon: 0.1000
Saved video for epoch 2240 to ./videos/flappy_bird_epoch_2240.mp4
Epoch 2260, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 2260 to ./videos/flappy_bird_epoch_2260.mp4
Epoch 2280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2280 to ./videos/flappy_bird_epoch_2280.mp4
The bird got through a pipe
Epoch 2300, Total Reward: -0.4999999999999982, Epsilon: 0.1000
Saved video for epoch 2300 to ./videos/flappy_bird_epoch_2300.mp4
Epoch 2320, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 2320 to ./videos/flappy_bird_epoch_2320.mp4
Epoch 2340, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 2340 to ./videos/flappy_bird_epoch_2340.mp4
The bird got through a pipe
Epoch 2360, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2360 to ./videos/flappy_bird_epoch_2360.mp4
Epoch 2380, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 2380 to ./videos/flappy_bird_epoch_2380.mp4
Epoch 2400, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 2400 to ./videos/flappy_bird_epoch_2400.mp4
Epoch 2420, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 2420 to ./videos/flappy_bird_epoch_2420.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 2440, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2440 to ./videos/flappy_bird_epoch_2440.mp4
Epoch 2460, Total Reward: 1.7763568394002505e-15, Epsilon: 0.1000
Saved video for epoch 2460 to ./videos/flappy_bird_epoch_2460.mp4
The bird got through a pipe
Epoch 2480, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2480 to ./videos/flappy_bird_epoch_2480.mp4
The bird got through a pipe
Epoch 2500, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 2500 to ./videos/flappy_bird_epoch_2500.mp4
The bird got through a pipe
Epoch 2520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2520 to ./videos/flappy_bird_epoch_2520.mp4
Epoch 2540, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2540 to ./videos/flappy_bird_epoch_2540.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 2560, Total Reward: -1.4999999999999978, Epsilon: 0.1000
Saved video for epoch 2560 to ./videos/flappy_bird_epoch_2560.mp4
The bird got through a pipe
Epoch 2580, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 2580 to ./videos/flappy_bird_epoch_2580.mp4
The bird got through a pipe
Epoch 2600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2600 to ./videos/flappy_bird_epoch_2600.mp4
Epoch 2620, Total Reward: -0.4999999999999982, Epsilon: 0.1000
Saved video for epoch 2620 to ./videos/flappy_bird_epoch_2620.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 2640, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 2640 to ./videos/flappy_bird_epoch_2640.mp4
Epoch 2660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2660 to ./videos/flappy_bird_epoch_2660.mp4
Epoch 2680, Total Reward: -1.4999999999999987, Epsilon: 0.1000
Saved video for epoch 2680 to ./videos/flappy_bird_epoch_2680.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 2700, Total Reward: 0.6000000000000014, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 2700 to ./videos/flappy_bird_epoch_2700.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 2720, Total Reward: 1.1000000000000019, Epsilon: 0.1000
Saved video for epoch 2720 to ./videos/flappy_bird_epoch_2720.mp4
Epoch 2740, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2740 to ./videos/flappy_bird_epoch_2740.mp4
The bird got through a pipe
Epoch 2760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2760 to ./videos/flappy_bird_epoch_2760.mp4
Epoch 2780, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 2780 to ./videos/flappy_bird_epoch_2780.mp4
Epoch 2800, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2800 to ./videos/flappy_bird_epoch_2800.mp4
The bird got through a pipe
Epoch 2820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2820 to ./videos/flappy_bird_epoch_2820.mp4
Epoch 2840, Total Reward: 1.0000000000000018, Epsilon: 0.1000
Saved video for epoch 2840 to ./videos/flappy_bird_epoch_2840.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 2860, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 2860 to ./videos/flappy_bird_epoch_2860.mp4
Epoch 2880, Total Reward: -1.4999999999999987, Epsilon: 0.1000
Saved video for epoch 2880 to ./videos/flappy_bird_epoch_2880.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 2900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2900 to ./videos/flappy_bird_epoch_2900.mp4
Epoch 2920, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2920 to ./videos/flappy_bird_epoch_2920.mp4
The bird got through a pipe
Epoch 2940, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 2940 to ./videos/flappy_bird_epoch_2940.mp4
The bird got through a pipe
Epoch 2960, Total Reward: 1.6999999999999993, Epsilon: 0.1000
Saved video for epoch 2960 to ./videos/flappy_bird_epoch_2960.mp4
The bird got through a pipe
Epoch 2980, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 2980 to ./videos/flappy_bird_epoch_2980.mp4
The bird got through a pipe
Epoch 3000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3000 to ./videos/flappy_bird_epoch_3000.mp4
The bird got through a pipe
Epoch 3020, Total Reward: 1.5000000000000018, Epsilon: 0.1000
Saved video for epoch 3020 to ./videos/flappy_bird_epoch_3020.mp4
Epoch 3040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3040 to ./videos/flappy_bird_epoch_3040.mp4
Epoch 3060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 3060 to ./videos/flappy_bird_epoch_3060.mp4
The bird got through a pipe
Epoch 3080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3080 to ./videos/flappy_bird_epoch_3080.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 3100, Total Reward: 1.9000000000000021, Epsilon: 0.1000
Saved video for epoch 3100 to ./videos/flappy_bird_epoch_3100.mp4
Epoch 3120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3120 to ./videos/flappy_bird_epoch_3120.mp4
Epoch 3140, Total Reward: 1.9000000000000021, Epsilon: 0.1000
Saved video for epoch 3140 to ./videos/flappy_bird_epoch_3140.mp4
Epoch 3160, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 3160 to ./videos/flappy_bird_epoch_3160.mp4
The bird got through a pipe
Epoch 3180, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 3180 to ./videos/flappy_bird_epoch_3180.mp4
Epoch 3200, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 3200 to ./videos/flappy_bird_epoch_3200.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 3220, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3220 to ./videos/flappy_bird_epoch_3220.mp4
Epoch 3240, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3240 to ./videos/flappy_bird_epoch_3240.mp4
Epoch 3260, Total Reward: -5.099999999999999, Epsilon: 0.1000
Saved video for epoch 3260 to ./videos/flappy_bird_epoch_3260.mp4
Epoch 3280, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 3280 to ./videos/flappy_bird_epoch_3280.mp4
Epoch 3300, Total Reward: -5.699999999999999, Epsilon: 0.1000
Saved video for epoch 3300 to ./videos/flappy_bird_epoch_3300.mp4
Epoch 3320, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 3320 to ./videos/flappy_bird_epoch_3320.mp4
Epoch 3340, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 3340 to ./videos/flappy_bird_epoch_3340.mp4
Epoch 3360, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 3360 to ./videos/flappy_bird_epoch_3360.mp4
The bird got through a pipe
Epoch 3380, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3380 to ./videos/flappy_bird_epoch_3380.mp4
The bird got through a pipe
Epoch 3400, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 3400 to ./videos/flappy_bird_epoch_3400.mp4
Epoch 3420, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3420 to ./videos/flappy_bird_epoch_3420.mp4
Epoch 3440, Total Reward: 1.1000000000000019, Epsilon: 0.1000
Saved video for epoch 3440 to ./videos/flappy_bird_epoch_3440.mp4
The bird got through a pipe
Epoch 3460, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3460 to ./videos/flappy_bird_epoch_3460.mp4
Epoch 3480, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 3480 to ./videos/flappy_bird_epoch_3480.mp4
Epoch 3500, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 3500 to ./videos/flappy_bird_epoch_3500.mp4
The bird got through a pipe
Epoch 3520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3520 to ./videos/flappy_bird_epoch_3520.mp4
Epoch 3540, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3540 to ./videos/flappy_bird_epoch_3540.mp4
The bird got through a pipe
Epoch 3560, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 3560 to ./videos/flappy_bird_epoch_3560.mp4
Epoch 3580, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 3580 to ./videos/flappy_bird_epoch_3580.mp4
Epoch 3600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3600 to ./videos/flappy_bird_epoch_3600.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 3620, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 3620 to ./videos/flappy_bird_epoch_3620.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 3640, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3640 to ./videos/flappy_bird_epoch_3640.mp4
Epoch 3660, Total Reward: 0.5000000000000018, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 3660 to ./videos/flappy_bird_epoch_3660.mp4
The bird got through a pipe
Epoch 3680, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 3680 to ./videos/flappy_bird_epoch_3680.mp4
The bird got through a pipe
Epoch 3700, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3700 to ./videos/flappy_bird_epoch_3700.mp4
The bird got through a pipe
Epoch 3720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3720 to ./videos/flappy_bird_epoch_3720.mp4
Epoch 3740, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 3740 to ./videos/flappy_bird_epoch_3740.mp4
The bird got through a pipe
Epoch 3760, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 3760 to ./videos/flappy_bird_epoch_3760.mp4
Epoch 3780, Total Reward: 1.1000000000000019, Epsilon: 0.1000
Saved video for epoch 3780 to ./videos/flappy_bird_epoch_3780.mp4
Epoch 3800, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3800 to ./videos/flappy_bird_epoch_3800.mp4
Epoch 3820, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 3820 to ./videos/flappy_bird_epoch_3820.mp4
The bird got through a pipe
Epoch 3840, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 3840 to ./videos/flappy_bird_epoch_3840.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 3860, Total Reward: -1.4999999999999987, Epsilon: 0.1000
Saved video for epoch 3860 to ./videos/flappy_bird_epoch_3860.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 3880, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3880 to ./videos/flappy_bird_epoch_3880.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 3900, Total Reward: 1.1000000000000019, Epsilon: 0.1000
Saved video for epoch 3900 to ./videos/flappy_bird_epoch_3900.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 3920, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3920 to ./videos/flappy_bird_epoch_3920.mp4
Epoch 3940, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 3940 to ./videos/flappy_bird_epoch_3940.mp4
The bird got through a pipe
Epoch 3960, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 3960 to ./videos/flappy_bird_epoch_3960.mp4
The bird got through a pipe
Epoch 3980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 3980 to ./videos/flappy_bird_epoch_3980.mp4
Epoch 4000, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 4000 to ./videos/flappy_bird_epoch_4000.mp4
Epoch 4020, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 4020 to ./videos/flappy_bird_epoch_4020.mp4
The bird got through a pipe
Epoch 4040, Total Reward: -1.899999999999999, Epsilon: 0.1000
Saved video for epoch 4040 to ./videos/flappy_bird_epoch_4040.mp4
The bird got through a pipe
Epoch 4060, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 4060 to ./videos/flappy_bird_epoch_4060.mp4
Epoch 4080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4080 to ./videos/flappy_bird_epoch_4080.mp4
Epoch 4100, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 4100 to ./videos/flappy_bird_epoch_4100.mp4
Epoch 4120, Total Reward: 3.299999999999997, Epsilon: 0.1000
Saved video for epoch 4120 to ./videos/flappy_bird_epoch_4120.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 4140, Total Reward: 0.9000000000000004, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 4140 to ./videos/flappy_bird_epoch_4140.mp4
Epoch 4160, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 4160 to ./videos/flappy_bird_epoch_4160.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 4180, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 4180 to ./videos/flappy_bird_epoch_4180.mp4
Epoch 4200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4200 to ./videos/flappy_bird_epoch_4200.mp4
The bird got through a pipe
Epoch 4220, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 4220 to ./videos/flappy_bird_epoch_4220.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 4240, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4240 to ./videos/flappy_bird_epoch_4240.mp4
Epoch 4260, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 4260 to ./videos/flappy_bird_epoch_4260.mp4
Epoch 4280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4280 to ./videos/flappy_bird_epoch_4280.mp4
Epoch 4300, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 4300 to ./videos/flappy_bird_epoch_4300.mp4
Epoch 4320, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4320 to ./videos/flappy_bird_epoch_4320.mp4
The bird got through a pipe
Epoch 4340, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4340 to ./videos/flappy_bird_epoch_4340.mp4
Epoch 4360, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 4360 to ./videos/flappy_bird_epoch_4360.mp4
Epoch 4380, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 4380 to ./videos/flappy_bird_epoch_4380.mp4
Epoch 4400, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4400 to ./videos/flappy_bird_epoch_4400.mp4
The bird got through a pipe
Epoch 4420, Total Reward: 1.0000000000000018, Epsilon: 0.1000
Saved video for epoch 4420 to ./videos/flappy_bird_epoch_4420.mp4
The bird got through a pipe
Epoch 4440, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4440 to ./videos/flappy_bird_epoch_4440.mp4
Epoch 4460, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 4460 to ./videos/flappy_bird_epoch_4460.mp4
Epoch 4480, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 4480 to ./videos/flappy_bird_epoch_4480.mp4
The bird got through a pipe
Epoch 4500, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 4500 to ./videos/flappy_bird_epoch_4500.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 4520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4520 to ./videos/flappy_bird_epoch_4520.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 4540, Total Reward: 1.1000000000000019, Epsilon: 0.1000
Saved video for epoch 4540 to ./videos/flappy_bird_epoch_4540.mp4
The bird got through a pipe
Epoch 4560, Total Reward: -1.2999999999999985, Epsilon: 0.1000
Saved video for epoch 4560 to ./videos/flappy_bird_epoch_4560.mp4
The bird got through a pipe
Epoch 4580, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 4580 to ./videos/flappy_bird_epoch_4580.mp4
The bird got through a pipe
Epoch 4600, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 4600 to ./videos/flappy_bird_epoch_4600.mp4
Epoch 4620, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4620 to ./videos/flappy_bird_epoch_4620.mp4
Epoch 4640, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 4640 to ./videos/flappy_bird_epoch_4640.mp4
The bird got through a pipe
Epoch 4660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4660 to ./videos/flappy_bird_epoch_4660.mp4
Epoch 4680, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 4680 to ./videos/flappy_bird_epoch_4680.mp4
Epoch 4700, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4700 to ./videos/flappy_bird_epoch_4700.mp4
Epoch 4720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4720 to ./videos/flappy_bird_epoch_4720.mp4
Epoch 4740, Total Reward: -0.9999999999999987, Epsilon: 0.1000
Saved video for epoch 4740 to ./videos/flappy_bird_epoch_4740.mp4
Epoch 4760, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 4760 to ./videos/flappy_bird_epoch_4760.mp4
Epoch 4780, Total Reward: -5.699999999999999, Epsilon: 0.1000
Saved video for epoch 4780 to ./videos/flappy_bird_epoch_4780.mp4
Epoch 4800, Total Reward: -5.699999999999999, Epsilon: 0.1000
Saved video for epoch 4800 to ./videos/flappy_bird_epoch_4800.mp4
Epoch 4820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4820 to ./videos/flappy_bird_epoch_4820.mp4
The bird got through a pipe
Epoch 4840, Total Reward: -0.3999999999999986, Epsilon: 0.1000
Saved video for epoch 4840 to ./videos/flappy_bird_epoch_4840.mp4
The bird got through a pipe
Epoch 4860, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 4860 to ./videos/flappy_bird_epoch_4860.mp4
Epoch 4880, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 4880 to ./videos/flappy_bird_epoch_4880.mp4
Epoch 4900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 4900 to ./videos/flappy_bird_epoch_4900.mp4
Epoch 4920, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 4920 to ./videos/flappy_bird_epoch_4920.mp4
Epoch 4940, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 4940 to ./videos/flappy_bird_epoch_4940.mp4
Epoch 4960, Total Reward: -1.9999999999999987, Epsilon: 0.1000
Saved video for epoch 4960 to ./videos/flappy_bird_epoch_4960.mp4
Epoch 4980, Total Reward: -1.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 4980 to ./videos/flappy_bird_epoch_4980.mp4
Epoch 5000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5000 to ./videos/flappy_bird_epoch_5000.mp4
The bird got through a pipe
Epoch 5020, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5020 to ./videos/flappy_bird_epoch_5020.mp4
The bird got through a pipe
Epoch 5040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 5040 to ./videos/flappy_bird_epoch_5040.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 5060, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 5060 to ./videos/flappy_bird_epoch_5060.mp4
Epoch 5080, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 5080 to ./videos/flappy_bird_epoch_5080.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 5100, Total Reward: 14.800000000000022, Epsilon: 0.1000
Saved video for epoch 5100 to ./videos/flappy_bird_epoch_5100.mp4
The bird got through a pipe
Epoch 5120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5120 to ./videos/flappy_bird_epoch_5120.mp4
Epoch 5140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5140 to ./videos/flappy_bird_epoch_5140.mp4
The bird got through a pipe
Epoch 5160, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 5160 to ./videos/flappy_bird_epoch_5160.mp4
Epoch 5180, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5180 to ./videos/flappy_bird_epoch_5180.mp4
The bird got through a pipe
Epoch 5200, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 5200 to ./videos/flappy_bird_epoch_5200.mp4
Epoch 5220, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5220 to ./videos/flappy_bird_epoch_5220.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 5240, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5240 to ./videos/flappy_bird_epoch_5240.mp4
The bird got through a pipe
Epoch 5260, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 5260 to ./videos/flappy_bird_epoch_5260.mp4
Epoch 5280, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 5280 to ./videos/flappy_bird_epoch_5280.mp4
Epoch 5300, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 5300 to ./videos/flappy_bird_epoch_5300.mp4
Epoch 5320, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5320 to ./videos/flappy_bird_epoch_5320.mp4
Epoch 5340, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 5340 to ./videos/flappy_bird_epoch_5340.mp4
The bird got through a pipe
Epoch 5360, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5360 to ./videos/flappy_bird_epoch_5360.mp4
Epoch 5380, Total Reward: -1.0999999999999988, Epsilon: 0.1000
Saved video for epoch 5380 to ./videos/flappy_bird_epoch_5380.mp4
Epoch 5400, Total Reward: -0.4999999999999982, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 5400 to ./videos/flappy_bird_epoch_5400.mp4
The bird got through a pipe
Epoch 5420, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 5420 to ./videos/flappy_bird_epoch_5420.mp4
Epoch 5440, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 5440 to ./videos/flappy_bird_epoch_5440.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 5460, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5460 to ./videos/flappy_bird_epoch_5460.mp4
The bird got through a pipe
Epoch 5480, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5480 to ./videos/flappy_bird_epoch_5480.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 5500, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 5500 to ./videos/flappy_bird_epoch_5500.mp4
The bird got through a pipe
Epoch 5520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5520 to ./videos/flappy_bird_epoch_5520.mp4
The bird got through a pipe
Epoch 5540, Total Reward: 17.200000000000024, Epsilon: 0.1000
Saved video for epoch 5540 to ./videos/flappy_bird_epoch_5540.mp4
Epoch 5560, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 5560 to ./videos/flappy_bird_epoch_5560.mp4
The bird got through a pipe
Epoch 5580, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 5580 to ./videos/flappy_bird_epoch_5580.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 5600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5600 to ./videos/flappy_bird_epoch_5600.mp4
Epoch 5620, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5620 to ./videos/flappy_bird_epoch_5620.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 5640, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5640 to ./videos/flappy_bird_epoch_5640.mp4
Epoch 5660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5660 to ./videos/flappy_bird_epoch_5660.mp4
Epoch 5680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5680 to ./videos/flappy_bird_epoch_5680.mp4
Epoch 5700, Total Reward: -0.9999999999999987, Epsilon: 0.1000
Saved video for epoch 5700 to ./videos/flappy_bird_epoch_5700.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 5720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5720 to ./videos/flappy_bird_epoch_5720.mp4
The bird got through a pipe
Epoch 5740, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5740 to ./videos/flappy_bird_epoch_5740.mp4
The bird got through a pipe
Epoch 5760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5760 to ./videos/flappy_bird_epoch_5760.mp4
Epoch 5780, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5780 to ./videos/flappy_bird_epoch_5780.mp4
The bird got through a pipe
Epoch 5800, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 5800 to ./videos/flappy_bird_epoch_5800.mp4
The bird got through a pipe
Epoch 5820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5820 to ./videos/flappy_bird_epoch_5820.mp4
The bird got through a pipe
Epoch 5840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5840 to ./videos/flappy_bird_epoch_5840.mp4
Epoch 5860, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 5860 to ./videos/flappy_bird_epoch_5860.mp4
The bird got through a pipe
Epoch 5880, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5880 to ./videos/flappy_bird_epoch_5880.mp4
Epoch 5900, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 5900 to ./videos/flappy_bird_epoch_5900.mp4
The bird got through a pipe
Epoch 5920, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 5920 to ./videos/flappy_bird_epoch_5920.mp4
The bird got through a pipe
Epoch 5940, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5940 to ./videos/flappy_bird_epoch_5940.mp4
Epoch 5960, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 5960 to ./videos/flappy_bird_epoch_5960.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 5980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 5980 to ./videos/flappy_bird_epoch_5980.mp4
Epoch 6000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6000 to ./videos/flappy_bird_epoch_6000.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 6020, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 6020 to ./videos/flappy_bird_epoch_6020.mp4
Epoch 6040, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 6040 to ./videos/flappy_bird_epoch_6040.mp4
Epoch 6060, Total Reward: 2.299999999999999, Epsilon: 0.1000
Saved video for epoch 6060 to ./videos/flappy_bird_epoch_6060.mp4
The bird got through a pipe
Epoch 6080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6080 to ./videos/flappy_bird_epoch_6080.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 6100, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 6100 to ./videos/flappy_bird_epoch_6100.mp4
Epoch 6120, Total Reward: -1.4999999999999982, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 6120 to ./videos/flappy_bird_epoch_6120.mp4
Epoch 6140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6140 to ./videos/flappy_bird_epoch_6140.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 6160, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6160 to ./videos/flappy_bird_epoch_6160.mp4
Epoch 6180, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 6180 to ./videos/flappy_bird_epoch_6180.mp4
The bird got through a pipe
Epoch 6200, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 6200 to ./videos/flappy_bird_epoch_6200.mp4
Epoch 6220, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 6220 to ./videos/flappy_bird_epoch_6220.mp4
Epoch 6240, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 6240 to ./videos/flappy_bird_epoch_6240.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 6260, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 6260 to ./videos/flappy_bird_epoch_6260.mp4
Epoch 6280, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 6280 to ./videos/flappy_bird_epoch_6280.mp4
Epoch 6300, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 6300 to ./videos/flappy_bird_epoch_6300.mp4
Epoch 6320, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 6320 to ./videos/flappy_bird_epoch_6320.mp4
Epoch 6340, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 6340 to ./videos/flappy_bird_epoch_6340.mp4
Epoch 6360, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 6360 to ./videos/flappy_bird_epoch_6360.mp4
Epoch 6380, Total Reward: 1.7763568394002505e-15, Epsilon: 0.1000
Saved video for epoch 6380 to ./videos/flappy_bird_epoch_6380.mp4
The bird got through a pipe
Epoch 6400, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6400 to ./videos/flappy_bird_epoch_6400.mp4
Epoch 6420, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 6420 to ./videos/flappy_bird_epoch_6420.mp4
Epoch 6440, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 6440 to ./videos/flappy_bird_epoch_6440.mp4
The bird got through a pipe
Epoch 6460, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 6460 to ./videos/flappy_bird_epoch_6460.mp4
Epoch 6480, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 6480 to ./videos/flappy_bird_epoch_6480.mp4
Epoch 6500, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 6500 to ./videos/flappy_bird_epoch_6500.mp4
Epoch 6520, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 6520 to ./videos/flappy_bird_epoch_6520.mp4
Epoch 6540, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 6540 to ./videos/flappy_bird_epoch_6540.mp4
Epoch 6560, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6560 to ./videos/flappy_bird_epoch_6560.mp4
The bird got through a pipe
Epoch 6580, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 6580 to ./videos/flappy_bird_epoch_6580.mp4
Epoch 6600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6600 to ./videos/flappy_bird_epoch_6600.mp4
Epoch 6620, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 6620 to ./videos/flappy_bird_epoch_6620.mp4
Epoch 6640, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 6640 to ./videos/flappy_bird_epoch_6640.mp4
Epoch 6660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6660 to ./videos/flappy_bird_epoch_6660.mp4
The bird got through a pipe
Epoch 6680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6680 to ./videos/flappy_bird_epoch_6680.mp4
Epoch 6700, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 6700 to ./videos/flappy_bird_epoch_6700.mp4
Epoch 6720, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 6720 to ./videos/flappy_bird_epoch_6720.mp4
Epoch 6740, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 6740 to ./videos/flappy_bird_epoch_6740.mp4
Epoch 6760, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 6760 to ./videos/flappy_bird_epoch_6760.mp4
Epoch 6780, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6780 to ./videos/flappy_bird_epoch_6780.mp4
The bird got through a pipe
Epoch 6800, Total Reward: 1.1000000000000014, Epsilon: 0.1000
Saved video for epoch 6800 to ./videos/flappy_bird_epoch_6800.mp4
The bird got through a pipe
Epoch 6820, Total Reward: 2.799999999999999, Epsilon: 0.1000
Saved video for epoch 6820 to ./videos/flappy_bird_epoch_6820.mp4
Epoch 6840, Total Reward: 1.1999999999999993, Epsilon: 0.1000
Saved video for epoch 6840 to ./videos/flappy_bird_epoch_6840.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 6860, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6860 to ./videos/flappy_bird_epoch_6860.mp4
Epoch 6880, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6880 to ./videos/flappy_bird_epoch_6880.mp4
Epoch 6900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6900 to ./videos/flappy_bird_epoch_6900.mp4
Epoch 6920, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 6920 to ./videos/flappy_bird_epoch_6920.mp4
Epoch 6940, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6940 to ./videos/flappy_bird_epoch_6940.mp4
Epoch 6960, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6960 to ./videos/flappy_bird_epoch_6960.mp4
Epoch 6980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 6980 to ./videos/flappy_bird_epoch_6980.mp4
Epoch 7000, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 7000 to ./videos/flappy_bird_epoch_7000.mp4
Epoch 7020, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7020 to ./videos/flappy_bird_epoch_7020.mp4
The bird got through a pipe
Epoch 7040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 7040 to ./videos/flappy_bird_epoch_7040.mp4
The bird got through a pipe
Epoch 7060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7060 to ./videos/flappy_bird_epoch_7060.mp4
The bird got through a pipe
Epoch 7080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7080 to ./videos/flappy_bird_epoch_7080.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 7100, Total Reward: 19.000000000000004, Epsilon: 0.1000
Saved video for epoch 7100 to ./videos/flappy_bird_epoch_7100.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 7120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7120 to ./videos/flappy_bird_epoch_7120.mp4
Epoch 7140, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 7140 to ./videos/flappy_bird_epoch_7140.mp4
Epoch 7160, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7160 to ./videos/flappy_bird_epoch_7160.mp4
Epoch 7180, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 7180 to ./videos/flappy_bird_epoch_7180.mp4
Epoch 7200, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 7200 to ./videos/flappy_bird_epoch_7200.mp4
Epoch 7220, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7220 to ./videos/flappy_bird_epoch_7220.mp4
Epoch 7240, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 7240 to ./videos/flappy_bird_epoch_7240.mp4
Epoch 7260, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7260 to ./videos/flappy_bird_epoch_7260.mp4
Epoch 7280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7280 to ./videos/flappy_bird_epoch_7280.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 7300, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 7300 to ./videos/flappy_bird_epoch_7300.mp4
Epoch 7320, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 7320 to ./videos/flappy_bird_epoch_7320.mp4
The bird got through a pipe
Epoch 7340, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 7340 to ./videos/flappy_bird_epoch_7340.mp4
Epoch 7360, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7360 to ./videos/flappy_bird_epoch_7360.mp4
Epoch 7380, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7380 to ./videos/flappy_bird_epoch_7380.mp4
Epoch 7400, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 7400 to ./videos/flappy_bird_epoch_7400.mp4
Epoch 7420, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 7420 to ./videos/flappy_bird_epoch_7420.mp4
Epoch 7440, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 7440 to ./videos/flappy_bird_epoch_7440.mp4
Epoch 7460, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7460 to ./videos/flappy_bird_epoch_7460.mp4
The bird got through a pipe
Epoch 7480, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7480 to ./videos/flappy_bird_epoch_7480.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 7500, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7500 to ./videos/flappy_bird_epoch_7500.mp4
Epoch 7520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7520 to ./videos/flappy_bird_epoch_7520.mp4
Epoch 7540, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7540 to ./videos/flappy_bird_epoch_7540.mp4
Epoch 7560, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7560 to ./videos/flappy_bird_epoch_7560.mp4
Epoch 7580, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7580 to ./videos/flappy_bird_epoch_7580.mp4
The bird got through a pipe
Epoch 7600, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 7600 to ./videos/flappy_bird_epoch_7600.mp4
Epoch 7620, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7620 to ./videos/flappy_bird_epoch_7620.mp4
Epoch 7640, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 7640 to ./videos/flappy_bird_epoch_7640.mp4
Epoch 7660, Total Reward: 1.5000000000000018, Epsilon: 0.1000
Saved video for epoch 7660 to ./videos/flappy_bird_epoch_7660.mp4
Epoch 7680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7680 to ./videos/flappy_bird_epoch_7680.mp4
The bird got through a pipe
Epoch 7700, Total Reward: 1.1000000000000014, Epsilon: 0.1000
Saved video for epoch 7700 to ./videos/flappy_bird_epoch_7700.mp4
The bird got through a pipe
Epoch 7720, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 7720 to ./videos/flappy_bird_epoch_7720.mp4
Epoch 7740, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 7740 to ./videos/flappy_bird_epoch_7740.mp4
Epoch 7760, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 7760 to ./videos/flappy_bird_epoch_7760.mp4
Epoch 7780, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 7780 to ./videos/flappy_bird_epoch_7780.mp4
Epoch 7800, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 7800 to ./videos/flappy_bird_epoch_7800.mp4
Epoch 7820, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 7820 to ./videos/flappy_bird_epoch_7820.mp4
Epoch 7840, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 7840 to ./videos/flappy_bird_epoch_7840.mp4
Epoch 7860, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7860 to ./videos/flappy_bird_epoch_7860.mp4
Epoch 7880, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 7880 to ./videos/flappy_bird_epoch_7880.mp4
Epoch 7900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7900 to ./videos/flappy_bird_epoch_7900.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 7920, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 7920 to ./videos/flappy_bird_epoch_7920.mp4
Epoch 7940, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 7940 to ./videos/flappy_bird_epoch_7940.mp4
Epoch 7960, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 7960 to ./videos/flappy_bird_epoch_7960.mp4
Epoch 7980, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 7980 to ./videos/flappy_bird_epoch_7980.mp4
Epoch 8000, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 8000 to ./videos/flappy_bird_epoch_8000.mp4
Epoch 8020, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 8020 to ./videos/flappy_bird_epoch_8020.mp4
Epoch 8040, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 8040 to ./videos/flappy_bird_epoch_8040.mp4
Epoch 8060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8060 to ./videos/flappy_bird_epoch_8060.mp4
Epoch 8080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8080 to ./videos/flappy_bird_epoch_8080.mp4
Epoch 8100, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8100 to ./videos/flappy_bird_epoch_8100.mp4
Epoch 8120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8120 to ./videos/flappy_bird_epoch_8120.mp4
Epoch 8140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8140 to ./videos/flappy_bird_epoch_8140.mp4
Epoch 8160, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8160 to ./videos/flappy_bird_epoch_8160.mp4
The bird got through a pipe
Epoch 8180, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8180 to ./videos/flappy_bird_epoch_8180.mp4
Epoch 8200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8200 to ./videos/flappy_bird_epoch_8200.mp4
Epoch 8220, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 8220 to ./videos/flappy_bird_epoch_8220.mp4
Epoch 8240, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8240 to ./videos/flappy_bird_epoch_8240.mp4
Epoch 8260, Total Reward: -0.9999999999999978, Epsilon: 0.1000
Saved video for epoch 8260 to ./videos/flappy_bird_epoch_8260.mp4
Epoch 8280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8280 to ./videos/flappy_bird_epoch_8280.mp4
The bird got through a pipe
Epoch 8300, Total Reward: 23.9, Epsilon: 0.1000
Saved video for epoch 8300 to ./videos/flappy_bird_epoch_8300.mp4
The bird got through a pipe
Epoch 8320, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 8320 to ./videos/flappy_bird_epoch_8320.mp4
The bird got through a pipe
Epoch 8340, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 8340 to ./videos/flappy_bird_epoch_8340.mp4
Epoch 8360, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 8360 to ./videos/flappy_bird_epoch_8360.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 8380, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 8380 to ./videos/flappy_bird_epoch_8380.mp4
The bird got through a pipe
Epoch 8400, Total Reward: 1.9000000000000004, Epsilon: 0.1000
Saved video for epoch 8400 to ./videos/flappy_bird_epoch_8400.mp4
Epoch 8420, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 8420 to ./videos/flappy_bird_epoch_8420.mp4
The bird got through a pipe
Epoch 8440, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8440 to ./videos/flappy_bird_epoch_8440.mp4
Epoch 8460, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 8460 to ./videos/flappy_bird_epoch_8460.mp4
Epoch 8480, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8480 to ./videos/flappy_bird_epoch_8480.mp4
Epoch 8500, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8500 to ./videos/flappy_bird_epoch_8500.mp4
Epoch 8520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8520 to ./videos/flappy_bird_epoch_8520.mp4
Epoch 8540, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 8540 to ./videos/flappy_bird_epoch_8540.mp4
Epoch 8560, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8560 to ./videos/flappy_bird_epoch_8560.mp4
Epoch 8580, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8580 to ./videos/flappy_bird_epoch_8580.mp4
Epoch 8600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8600 to ./videos/flappy_bird_epoch_8600.mp4
Epoch 8620, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8620 to ./videos/flappy_bird_epoch_8620.mp4
Epoch 8640, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8640 to ./videos/flappy_bird_epoch_8640.mp4
Epoch 8660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8660 to ./videos/flappy_bird_epoch_8660.mp4
Epoch 8680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8680 to ./videos/flappy_bird_epoch_8680.mp4
Epoch 8700, Total Reward: 1.799999999999999, Epsilon: 0.1000
Saved video for epoch 8700 to ./videos/flappy_bird_epoch_8700.mp4
Epoch 8720, Total Reward: 4.440892098500626e-16, Epsilon: 0.1000
Saved video for epoch 8720 to ./videos/flappy_bird_epoch_8720.mp4
Epoch 8740, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8740 to ./videos/flappy_bird_epoch_8740.mp4
Epoch 8760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8760 to ./videos/flappy_bird_epoch_8760.mp4
Epoch 8780, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 8780 to ./videos/flappy_bird_epoch_8780.mp4
Epoch 8800, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8800 to ./videos/flappy_bird_epoch_8800.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 8820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8820 to ./videos/flappy_bird_epoch_8820.mp4
Epoch 8840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 8840 to ./videos/flappy_bird_epoch_8840.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 8860, Total Reward: 2.799999999999999, Epsilon: 0.1000
Saved video for epoch 8860 to ./videos/flappy_bird_epoch_8860.mp4
The bird got through a pipe
Epoch 8880, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 8880 to ./videos/flappy_bird_epoch_8880.mp4
The bird got through a pipe
Epoch 8900, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 8900 to ./videos/flappy_bird_epoch_8900.mp4
Epoch 8920, Total Reward: -4.499999999999998, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 8920 to ./videos/flappy_bird_epoch_8920.mp4
Epoch 8940, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 8940 to ./videos/flappy_bird_epoch_8940.mp4
Epoch 8960, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 8960 to ./videos/flappy_bird_epoch_8960.mp4
Epoch 8980, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 8980 to ./videos/flappy_bird_epoch_8980.mp4
Epoch 9000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9000 to ./videos/flappy_bird_epoch_9000.mp4
Epoch 9020, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 9020 to ./videos/flappy_bird_epoch_9020.mp4
Epoch 9040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9040 to ./videos/flappy_bird_epoch_9040.mp4
Epoch 9060, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 9060 to ./videos/flappy_bird_epoch_9060.mp4
Epoch 9080, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 9080 to ./videos/flappy_bird_epoch_9080.mp4
Epoch 9100, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 9100 to ./videos/flappy_bird_epoch_9100.mp4
Epoch 9120, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 9120 to ./videos/flappy_bird_epoch_9120.mp4
Epoch 9140, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 9140 to ./videos/flappy_bird_epoch_9140.mp4
Epoch 9160, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9160 to ./videos/flappy_bird_epoch_9160.mp4
Epoch 9180, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 9180 to ./videos/flappy_bird_epoch_9180.mp4
Epoch 9200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9200 to ./videos/flappy_bird_epoch_9200.mp4
The bird got through a pipe
Epoch 9220, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9220 to ./videos/flappy_bird_epoch_9220.mp4
The bird got through a pipe
Epoch 9240, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 9240 to ./videos/flappy_bird_epoch_9240.mp4
Epoch 9260, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 9260 to ./videos/flappy_bird_epoch_9260.mp4
Epoch 9280, Total Reward: -5.099999999999999, Epsilon: 0.1000
Saved video for epoch 9280 to ./videos/flappy_bird_epoch_9280.mp4
Epoch 9300, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 9300 to ./videos/flappy_bird_epoch_9300.mp4
Epoch 9320, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 9320 to ./videos/flappy_bird_epoch_9320.mp4
Epoch 9340, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 9340 to ./videos/flappy_bird_epoch_9340.mp4
Epoch 9360, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 9360 to ./videos/flappy_bird_epoch_9360.mp4
Epoch 9380, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 9380 to ./videos/flappy_bird_epoch_9380.mp4
Epoch 9400, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9400 to ./videos/flappy_bird_epoch_9400.mp4
Epoch 9420, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 9420 to ./videos/flappy_bird_epoch_9420.mp4
Epoch 9440, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9440 to ./videos/flappy_bird_epoch_9440.mp4
Epoch 9460, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9460 to ./videos/flappy_bird_epoch_9460.mp4
Epoch 9480, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 9480 to ./videos/flappy_bird_epoch_9480.mp4
The bird got through a pipe
Epoch 9500, Total Reward: 17.200000000000024, Epsilon: 0.1000
Saved video for epoch 9500 to ./videos/flappy_bird_epoch_9500.mp4
Epoch 9520, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 9520 to ./videos/flappy_bird_epoch_9520.mp4
Epoch 9540, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 9540 to ./videos/flappy_bird_epoch_9540.mp4
Epoch 9560, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 9560 to ./videos/flappy_bird_epoch_9560.mp4
Epoch 9580, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9580 to ./videos/flappy_bird_epoch_9580.mp4
Epoch 9600, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 9600 to ./videos/flappy_bird_epoch_9600.mp4
Epoch 9620, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 9620 to ./videos/flappy_bird_epoch_9620.mp4
Epoch 9640, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 9640 to ./videos/flappy_bird_epoch_9640.mp4
Epoch 9660, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 9660 to ./videos/flappy_bird_epoch_9660.mp4
Epoch 9680, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 9680 to ./videos/flappy_bird_epoch_9680.mp4
The bird got through a pipe
Epoch 9700, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9700 to ./videos/flappy_bird_epoch_9700.mp4
Epoch 9720, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 9720 to ./videos/flappy_bird_epoch_9720.mp4
Epoch 9740, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 9740 to ./videos/flappy_bird_epoch_9740.mp4
Epoch 9760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9760 to ./videos/flappy_bird_epoch_9760.mp4
Epoch 9780, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 9780 to ./videos/flappy_bird_epoch_9780.mp4
Epoch 9800, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9800 to ./videos/flappy_bird_epoch_9800.mp4
Epoch 9820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9820 to ./videos/flappy_bird_epoch_9820.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 9840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9840 to ./videos/flappy_bird_epoch_9840.mp4
Epoch 9860, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9860 to ./videos/flappy_bird_epoch_9860.mp4
Epoch 9880, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9880 to ./videos/flappy_bird_epoch_9880.mp4
Epoch 9900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9900 to ./videos/flappy_bird_epoch_9900.mp4
The bird got through a pipe
Epoch 9920, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 9920 to ./videos/flappy_bird_epoch_9920.mp4
Epoch 9940, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 9940 to ./videos/flappy_bird_epoch_9940.mp4
Epoch 9960, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 9960 to ./videos/flappy_bird_epoch_9960.mp4
Epoch 9980, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 9980 to ./videos/flappy_bird_epoch_9980.mp4
Epoch 10000, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 10000 to ./videos/flappy_bird_epoch_10000.mp4
Epoch 10020, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 10020 to ./videos/flappy_bird_epoch_10020.mp4
Epoch 10040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 10040 to ./videos/flappy_bird_epoch_10040.mp4
Epoch 10060, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 10060 to ./videos/flappy_bird_epoch_10060.mp4
Epoch 10080, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 10080 to ./videos/flappy_bird_epoch_10080.mp4
Epoch 10100, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 10100 to ./videos/flappy_bird_epoch_10100.mp4
Epoch 10120, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 10120 to ./videos/flappy_bird_epoch_10120.mp4
Epoch 10140, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 10140 to ./videos/flappy_bird_epoch_10140.mp4
Epoch 10160, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 10160 to ./videos/flappy_bird_epoch_10160.mp4
Epoch 10180, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 10180 to ./videos/flappy_bird_epoch_10180.mp4
Epoch 10200, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 10200 to ./videos/flappy_bird_epoch_10200.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 10220, Total Reward: 20.799999999999997, Epsilon: 0.1000
Saved video for epoch 10220 to ./videos/flappy_bird_epoch_10220.mp4
The bird got through a pipe
Epoch 10240, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 10240 to ./videos/flappy_bird_epoch_10240.mp4
Epoch 10260, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 10260 to ./videos/flappy_bird_epoch_10260.mp4
Epoch 10280, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 10280 to ./videos/flappy_bird_epoch_10280.mp4
Epoch 10300, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 10300 to ./videos/flappy_bird_epoch_10300.mp4
Epoch 10320, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 10320 to ./videos/flappy_bird_epoch_10320.mp4
Epoch 10340, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 10340 to ./videos/flappy_bird_epoch_10340.mp4
Epoch 10360, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 10360 to ./videos/flappy_bird_epoch_10360.mp4
Epoch 10380, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 10380 to ./videos/flappy_bird_epoch_10380.mp4
Epoch 10400, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 10400 to ./videos/flappy_bird_epoch_10400.mp4
Epoch 10420, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 10420 to ./videos/flappy_bird_epoch_10420.mp4
Epoch 10440, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 10440 to ./videos/flappy_bird_epoch_10440.mp4
Epoch 10460, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 10460 to ./videos/flappy_bird_epoch_10460.mp4
Epoch 10480, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 10480 to ./videos/flappy_bird_epoch_10480.mp4
Epoch 10500, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 10500 to ./videos/flappy_bird_epoch_10500.mp4
Epoch 10520, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 10520 to ./videos/flappy_bird_epoch_10520.mp4
Epoch 10540, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 10540 to ./videos/flappy_bird_epoch_10540.mp4
Epoch 10560, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 10560 to ./videos/flappy_bird_epoch_10560.mp4
Epoch 10580, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 10580 to ./videos/flappy_bird_epoch_10580.mp4
Epoch 10600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 10600 to ./videos/flappy_bird_epoch_10600.mp4
Epoch 10620, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 10620 to ./videos/flappy_bird_epoch_10620.mp4
Epoch 10640, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 10640 to ./videos/flappy_bird_epoch_10640.mp4
Epoch 10660, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 10660 to ./videos/flappy_bird_epoch_10660.mp4
Epoch 10680, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 10680 to ./videos/flappy_bird_epoch_10680.mp4
Epoch 10700, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 10700 to ./videos/flappy_bird_epoch_10700.mp4
Epoch 10720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 10720 to ./videos/flappy_bird_epoch_10720.mp4
Epoch 10740, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 10740 to ./videos/flappy_bird_epoch_10740.mp4
Epoch 10760, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 10760 to ./videos/flappy_bird_epoch_10760.mp4
Epoch 10780, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 10780 to ./videos/flappy_bird_epoch_10780.mp4
Epoch 10800, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 10800 to ./videos/flappy_bird_epoch_10800.mp4
Epoch 10820, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 10820 to ./videos/flappy_bird_epoch_10820.mp4
Epoch 10840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 10840 to ./videos/flappy_bird_epoch_10840.mp4
Epoch 10860, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 10860 to ./videos/flappy_bird_epoch_10860.mp4
Epoch 10880, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 10880 to ./videos/flappy_bird_epoch_10880.mp4
Epoch 10900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 10900 to ./videos/flappy_bird_epoch_10900.mp4
The bird got through a pipe
Epoch 10920, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 10920 to ./videos/flappy_bird_epoch_10920.mp4
Epoch 10940, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 10940 to ./videos/flappy_bird_epoch_10940.mp4
The bird got through a pipe
Epoch 10960, Total Reward: 18.700000000000003, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 10960 to ./videos/flappy_bird_epoch_10960.mp4
The bird got through a pipe
Epoch 10980, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 10980 to ./videos/flappy_bird_epoch_10980.mp4
Epoch 11000, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 11000 to ./videos/flappy_bird_epoch_11000.mp4
Epoch 11020, Total Reward: 0.6000000000000014, Epsilon: 0.1000
Saved video for epoch 11020 to ./videos/flappy_bird_epoch_11020.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11040, Total Reward: 20.300000000000004, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 11040 to ./videos/flappy_bird_epoch_11040.mp4
The bird got through a pipe
Epoch 11060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11060 to ./videos/flappy_bird_epoch_11060.mp4
The bird got through a pipe
Epoch 11080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11080 to ./videos/flappy_bird_epoch_11080.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11100, Total Reward: 18.200000000000003, Epsilon: 0.1000
Saved video for epoch 11100 to ./videos/flappy_bird_epoch_11100.mp4
Epoch 11120, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 11120 to ./videos/flappy_bird_epoch_11120.mp4
Epoch 11140, Total Reward: 2.1000000000000014, Epsilon: 0.1000
Saved video for epoch 11140 to ./videos/flappy_bird_epoch_11140.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11160, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11160 to ./videos/flappy_bird_epoch_11160.mp4
Epoch 11180, Total Reward: 1.7763568394002505e-15, Epsilon: 0.1000
Saved video for epoch 11180 to ./videos/flappy_bird_epoch_11180.mp4
Epoch 11200, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 11200 to ./videos/flappy_bird_epoch_11200.mp4
Epoch 11220, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 11220 to ./videos/flappy_bird_epoch_11220.mp4
Epoch 11240, Total Reward: -5.699999999999999, Epsilon: 0.1000
Saved video for epoch 11240 to ./videos/flappy_bird_epoch_11240.mp4
The bird got through a pipe
Epoch 11260, Total Reward: 20.30000000000001, Epsilon: 0.1000
Saved video for epoch 11260 to ./videos/flappy_bird_epoch_11260.mp4
Epoch 11280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11280 to ./videos/flappy_bird_epoch_11280.mp4
Epoch 11300, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11300 to ./videos/flappy_bird_epoch_11300.mp4
The bird got through a pipe
Epoch 11320, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11320 to ./videos/flappy_bird_epoch_11320.mp4
The bird got through a pipe
Epoch 11340, Total Reward: 1.7763568394002505e-15, Epsilon: 0.1000
Saved video for epoch 11340 to ./videos/flappy_bird_epoch_11340.mp4
Epoch 11360, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 11360 to ./videos/flappy_bird_epoch_11360.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 11380, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11380 to ./videos/flappy_bird_epoch_11380.mp4
Epoch 11400, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11400 to ./videos/flappy_bird_epoch_11400.mp4
Epoch 11420, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 11420 to ./videos/flappy_bird_epoch_11420.mp4
The bird got through a pipe
Epoch 11440, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11440 to ./videos/flappy_bird_epoch_11440.mp4
Epoch 11460, Total Reward: 1.6000000000000019, Epsilon: 0.1000
Saved video for epoch 11460 to ./videos/flappy_bird_epoch_11460.mp4
Epoch 11480, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 11480 to ./videos/flappy_bird_epoch_11480.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11500, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 11500 to ./videos/flappy_bird_epoch_11500.mp4
Epoch 11520, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 11520 to ./videos/flappy_bird_epoch_11520.mp4
Epoch 11540, Total Reward: 1.7763568394002505e-15, Epsilon: 0.1000
Saved video for epoch 11540 to ./videos/flappy_bird_epoch_11540.mp4
Epoch 11560, Total Reward: 1.6000000000000019, Epsilon: 0.1000
Saved video for epoch 11560 to ./videos/flappy_bird_epoch_11560.mp4
Epoch 11580, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 11580 to ./videos/flappy_bird_epoch_11580.mp4
Epoch 11600, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 11600 to ./videos/flappy_bird_epoch_11600.mp4
The bird got through a pipe
Epoch 11620, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 11620 to ./videos/flappy_bird_epoch_11620.mp4
The bird got through a pipe
Epoch 11640, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 11640 to ./videos/flappy_bird_epoch_11640.mp4
The bird got through a pipe
Epoch 11660, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 11660 to ./videos/flappy_bird_epoch_11660.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11680 to ./videos/flappy_bird_epoch_11680.mp4
Epoch 11700, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 11700 to ./videos/flappy_bird_epoch_11700.mp4
The bird got through a pipe
Epoch 11720, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 11720 to ./videos/flappy_bird_epoch_11720.mp4
Epoch 11740, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11740 to ./videos/flappy_bird_epoch_11740.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11760 to ./videos/flappy_bird_epoch_11760.mp4
The bird got through a pipe
Epoch 11780, Total Reward: 1.700000000000002, Epsilon: 0.1000
Saved video for epoch 11780 to ./videos/flappy_bird_epoch_11780.mp4
The bird got through a pipe
Epoch 11800, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11800 to ./videos/flappy_bird_epoch_11800.mp4
Epoch 11820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11820 to ./videos/flappy_bird_epoch_11820.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11840 to ./videos/flappy_bird_epoch_11840.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 11860, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11860 to ./videos/flappy_bird_epoch_11860.mp4
The bird got through a pipe
Epoch 11880, Total Reward: -1.9999999999999987, Epsilon: 0.1000
Saved video for epoch 11880 to ./videos/flappy_bird_epoch_11880.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11900, Total Reward: 2.4999999999999982, Epsilon: 0.1000
Saved video for epoch 11900 to ./videos/flappy_bird_epoch_11900.mp4
Epoch 11920, Total Reward: 0.5000000000000018, Epsilon: 0.1000
Saved video for epoch 11920 to ./videos/flappy_bird_epoch_11920.mp4
Epoch 11940, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11940 to ./videos/flappy_bird_epoch_11940.mp4
Epoch 11960, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 11960 to ./videos/flappy_bird_epoch_11960.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 11980, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 11980 to ./videos/flappy_bird_epoch_11980.mp4
The bird got through a pipe
Epoch 12000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12000 to ./videos/flappy_bird_epoch_12000.mp4
The bird got through a pipe
Epoch 12020, Total Reward: -0.39999999999999813, Epsilon: 0.1000
Saved video for epoch 12020 to ./videos/flappy_bird_epoch_12020.mp4
Epoch 12040, Total Reward: 0.5000000000000018, Epsilon: 0.1000
Saved video for epoch 12040 to ./videos/flappy_bird_epoch_12040.mp4
The bird got through a pipe
Epoch 12060, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 12060 to ./videos/flappy_bird_epoch_12060.mp4
The bird got through a pipe
Epoch 12080, Total Reward: 24.200000000000003, Epsilon: 0.1000
Saved video for epoch 12080 to ./videos/flappy_bird_epoch_12080.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12100, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 12100 to ./videos/flappy_bird_epoch_12100.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12120 to ./videos/flappy_bird_epoch_12120.mp4
Epoch 12140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12140 to ./videos/flappy_bird_epoch_12140.mp4
Epoch 12160, Total Reward: 0.5000000000000018, Epsilon: 0.1000
Saved video for epoch 12160 to ./videos/flappy_bird_epoch_12160.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12180, Total Reward: 24.400000000000006, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 12180 to ./videos/flappy_bird_epoch_12180.mp4
The bird got through a pipe
Epoch 12200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12200 to ./videos/flappy_bird_epoch_12200.mp4
Epoch 12220, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12220 to ./videos/flappy_bird_epoch_12220.mp4
Epoch 12240, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 12240 to ./videos/flappy_bird_epoch_12240.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 12260, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12260 to ./videos/flappy_bird_epoch_12260.mp4
Epoch 12280, Total Reward: 1.6999999999999993, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 12280 to ./videos/flappy_bird_epoch_12280.mp4
The bird got through a pipe
Epoch 12300, Total Reward: 20.3, Epsilon: 0.1000
Saved video for epoch 12300 to ./videos/flappy_bird_epoch_12300.mp4
Epoch 12320, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 12320 to ./videos/flappy_bird_epoch_12320.mp4
Epoch 12340, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 12340 to ./videos/flappy_bird_epoch_12340.mp4
Epoch 12360, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 12360 to ./videos/flappy_bird_epoch_12360.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 12380, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 12380 to ./videos/flappy_bird_epoch_12380.mp4
Epoch 12400, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12400 to ./videos/flappy_bird_epoch_12400.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 12420, Total Reward: 14.20000000000002, Epsilon: 0.1000
Saved video for epoch 12420 to ./videos/flappy_bird_epoch_12420.mp4
The bird got through a pipe
Epoch 12440, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12440 to ./videos/flappy_bird_epoch_12440.mp4
The bird got through a pipe
Epoch 12460, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 12460 to ./videos/flappy_bird_epoch_12460.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12480, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12480 to ./videos/flappy_bird_epoch_12480.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12500, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12500 to ./videos/flappy_bird_epoch_12500.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12520, Total Reward: 24.200000000000003, Epsilon: 0.1000
Saved video for epoch 12520 to ./videos/flappy_bird_epoch_12520.mp4
Epoch 12540, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12540 to ./videos/flappy_bird_epoch_12540.mp4
The bird got through a pipe
Epoch 12560, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 12560 to ./videos/flappy_bird_epoch_12560.mp4
The bird got through a pipe
Epoch 12580, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12580 to ./videos/flappy_bird_epoch_12580.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12600 to ./videos/flappy_bird_epoch_12600.mp4
The bird got through a pipe
Epoch 12620, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 12620 to ./videos/flappy_bird_epoch_12620.mp4
The bird got through a pipe
Epoch 12640, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12640 to ./videos/flappy_bird_epoch_12640.mp4
The bird got through a pipe
Epoch 12660, Total Reward: -3.299999999999998, Epsilon: 0.1000
Epoch 12660, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 12660 to ./videos/flappy_bird_epoch_12660.mp4
The bird got through a pipe
Epoch 12680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12680 to ./videos/flappy_bird_epoch_12680.mp4
Epoch 12700, Total Reward: -1.4999999999999987, Epsilon: 0.1000
Saved video for epoch 12700 to ./videos/flappy_bird_epoch_12700.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12720, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 12720 to ./videos/flappy_bird_epoch_12720.mp4
Epoch 12740, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 12740 to ./videos/flappy_bird_epoch_12740.mp4
Epoch 12760, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 12760 to ./videos/flappy_bird_epoch_12760.mp4
The bird got through a pipe
Epoch 12780, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12780 to ./videos/flappy_bird_epoch_12780.mp4
Epoch 12800, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 12800 to ./videos/flappy_bird_epoch_12800.mp4
Epoch 12820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12820 to ./videos/flappy_bird_epoch_12820.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12840 to ./videos/flappy_bird_epoch_12840.mp4
The bird got through a pipe
Epoch 12860, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 12860 to ./videos/flappy_bird_epoch_12860.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 12880, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 12880 to ./videos/flappy_bird_epoch_12880.mp4
Epoch 12900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12900 to ./videos/flappy_bird_epoch_12900.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12920, Total Reward: 2.799999999999999, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 12920 to ./videos/flappy_bird_epoch_12920.mp4
The bird got through a pipe
Epoch 12940, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12940 to ./videos/flappy_bird_epoch_12940.mp4
The bird got through a pipe
Epoch 12960, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 12960 to ./videos/flappy_bird_epoch_12960.mp4
The bird got through a pipe
Epoch 12980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12980 to ./videos/flappy_bird_epoch_12980.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13000, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 13000 to ./videos/flappy_bird_epoch_13000.mp4
Epoch 13020, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 13020 to ./videos/flappy_bird_epoch_13020.mp4
The bird got through a pipe
Epoch 13040, Total Reward: 14.20000000000002, Epsilon: 0.1000
Saved video for epoch 13040 to ./videos/flappy_bird_epoch_13040.mp4
The bird got through a pipe
Epoch 13060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13060 to ./videos/flappy_bird_epoch_13060.mp4
Epoch 13080, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13080 to ./videos/flappy_bird_epoch_13080.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13100, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13100 to ./videos/flappy_bird_epoch_13100.mp4
Epoch 13120, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 13120 to ./videos/flappy_bird_epoch_13120.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13140 to ./videos/flappy_bird_epoch_13140.mp4
Epoch 13160, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 13160 to ./videos/flappy_bird_epoch_13160.mp4
Epoch 13180, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 13180 to ./videos/flappy_bird_epoch_13180.mp4
The bird got through a pipe
Epoch 13200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13200 to ./videos/flappy_bird_epoch_13200.mp4
The bird got through a pipe
Epoch 13220, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 13220 to ./videos/flappy_bird_epoch_13220.mp4
Epoch 13240, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 13240 to ./videos/flappy_bird_epoch_13240.mp4
Epoch 13260, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13260 to ./videos/flappy_bird_epoch_13260.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13280 to ./videos/flappy_bird_epoch_13280.mp4
The bird got through a pipe
Epoch 13300, Total Reward: 1.799999999999999, Epsilon: 0.1000
Saved video for epoch 13300 to ./videos/flappy_bird_epoch_13300.mp4
The bird got through a pipe
Epoch 13320, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13320 to ./videos/flappy_bird_epoch_13320.mp4
The bird got through a pipe
Epoch 13340, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 13340 to ./videos/flappy_bird_epoch_13340.mp4
The bird got through a pipe
Epoch 13360, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13360 to ./videos/flappy_bird_epoch_13360.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 13380, Total Reward: 37.20000000000004, Epsilon: 0.1000
Saved video for epoch 13380 to ./videos/flappy_bird_epoch_13380.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 13400, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13400 to ./videos/flappy_bird_epoch_13400.mp4
The bird got through a pipe
Epoch 13420, Total Reward: 17.800000000000022, Epsilon: 0.1000
Saved video for epoch 13420 to ./videos/flappy_bird_epoch_13420.mp4
The bird got through a pipe
Epoch 13440, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 13440 to ./videos/flappy_bird_epoch_13440.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13460, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13460 to ./videos/flappy_bird_epoch_13460.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 13480, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13480 to ./videos/flappy_bird_epoch_13480.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13500, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13500 to ./videos/flappy_bird_epoch_13500.mp4
Epoch 13520, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 13520 to ./videos/flappy_bird_epoch_13520.mp4
Epoch 13540, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13540 to ./videos/flappy_bird_epoch_13540.mp4
Epoch 13560, Total Reward: 3.1999999999999975, Epsilon: 0.1000
Saved video for epoch 13560 to ./videos/flappy_bird_epoch_13560.mp4
Epoch 13580, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13580 to ./videos/flappy_bird_epoch_13580.mp4
Epoch 13600, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 13600 to ./videos/flappy_bird_epoch_13600.mp4
Epoch 13620, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 13620 to ./videos/flappy_bird_epoch_13620.mp4
Epoch 13640, Total Reward: -2.4999999999999987, Epsilon: 0.1000
Saved video for epoch 13640 to ./videos/flappy_bird_epoch_13640.mp4
The bird got through a pipe
Epoch 13660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13660 to ./videos/flappy_bird_epoch_13660.mp4
Epoch 13680, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13680 to ./videos/flappy_bird_epoch_13680.mp4
Epoch 13700, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 13700 to ./videos/flappy_bird_epoch_13700.mp4
Epoch 13720, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 13720 to ./videos/flappy_bird_epoch_13720.mp4
The bird got through a pipe
Epoch 13740, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13740 to ./videos/flappy_bird_epoch_13740.mp4
The bird got through a pipe
Epoch 13760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13760 to ./videos/flappy_bird_epoch_13760.mp4
The bird got through a pipe
Epoch 13780, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 13780 to ./videos/flappy_bird_epoch_13780.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13800, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13800 to ./videos/flappy_bird_epoch_13800.mp4
The bird got through a pipe
Epoch 13820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13820 to ./videos/flappy_bird_epoch_13820.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13840 to ./videos/flappy_bird_epoch_13840.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13860, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13860 to ./videos/flappy_bird_epoch_13860.mp4
The bird got through a pipe
Epoch 13880, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 13880 to ./videos/flappy_bird_epoch_13880.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 13900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13900 to ./videos/flappy_bird_epoch_13900.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13920, Total Reward: -0.9999999999999987, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 13920 to ./videos/flappy_bird_epoch_13920.mp4
Epoch 13940, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 13940 to ./videos/flappy_bird_epoch_13940.mp4
Epoch 13960, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 13960 to ./videos/flappy_bird_epoch_13960.mp4
Epoch 13980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13980 to ./videos/flappy_bird_epoch_13980.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 14000, Total Reward: -1.9999999999999987, Epsilon: 0.1000
Saved video for epoch 14000 to ./videos/flappy_bird_epoch_14000.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 14020, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 14020 to ./videos/flappy_bird_epoch_14020.mp4
The bird got through a pipe
Epoch 14040, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 14040 to ./videos/flappy_bird_epoch_14040.mp4
Epoch 14060, Total Reward: -0.4999999999999982, Epsilon: 0.1000
Saved video for epoch 14060 to ./videos/flappy_bird_epoch_14060.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 14080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14080 to ./videos/flappy_bird_epoch_14080.mp4
Epoch 14100, Total Reward: 2.1999999999999993, Epsilon: 0.1000
Saved video for epoch 14100 to ./videos/flappy_bird_epoch_14100.mp4
Epoch 14120, Total Reward: 1.6000000000000019, Epsilon: 0.1000
Saved video for epoch 14120 to ./videos/flappy_bird_epoch_14120.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 14140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14140 to ./videos/flappy_bird_epoch_14140.mp4
The bird got through a pipe
Epoch 14160, Total Reward: 19.60000000000001, Epsilon: 0.1000
Saved video for epoch 14160 to ./videos/flappy_bird_epoch_14160.mp4
Epoch 14180, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14180 to ./videos/flappy_bird_epoch_14180.mp4
The bird got through a pipe
Epoch 14200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14200 to ./videos/flappy_bird_epoch_14200.mp4
The bird got through a pipe
The bird got through a pipe
The bird got through a pipe
Epoch 14220, Total Reward: 14.800000000000018, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 14220 to ./videos/flappy_bird_epoch_14220.mp4
The bird got through a pipe
Epoch 14240, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14240 to ./videos/flappy_bird_epoch_14240.mp4
Epoch 14260, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 14260 to ./videos/flappy_bird_epoch_14260.mp4
Epoch 14280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14280 to ./videos/flappy_bird_epoch_14280.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 14300, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 14300 to ./videos/flappy_bird_epoch_14300.mp4
The bird got through a pipe
Epoch 14320, Total Reward: -0.3999999999999986, Epsilon: 0.1000
```