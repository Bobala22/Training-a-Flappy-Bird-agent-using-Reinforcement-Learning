# Experiment 3

## Observations:
- not taking in consideration last 2 states, we were only working on the current state of the game
- the reward when the bird succesfully passed through a pipe would be 20
- too big convolutions

## Hyperparameters:
- learning rate = 0,001
- batch_size = 40
- replay buffer capacity = 10000
- target network update every 10 eps.
- epsilon decay = 0,999

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
    batch_size = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlappyBirdCNN().to(device)
    target_model = FlappyBirdCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Hyperparameters
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.999
    gamma = 0.99
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)

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
        if (epoch + 1) % 10 == 0:
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
Epoch 11240, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 11240 to ./videos/flappy_bird_epoch_11240.mp4
Epoch 11260, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11260 to ./videos/flappy_bird_epoch_11260.mp4
Epoch 11280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11280 to ./videos/flappy_bird_epoch_11280.mp4
The bird got through a pipe
Epoch 11300, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11300 to ./videos/flappy_bird_epoch_11300.mp4
Epoch 11320, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11320 to ./videos/flappy_bird_epoch_11320.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11340, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 11340 to ./videos/flappy_bird_epoch_11340.mp4
Epoch 11360, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 11360 to ./videos/flappy_bird_epoch_11360.mp4
Epoch 11380, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 11380 to ./videos/flappy_bird_epoch_11380.mp4
Epoch 11400, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11400 to ./videos/flappy_bird_epoch_11400.mp4
Epoch 11420, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 11420 to ./videos/flappy_bird_epoch_11420.mp4
Epoch 11440, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 11440 to ./videos/flappy_bird_epoch_11440.mp4
The bird got through a pipe
Epoch 11460, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 11460 to ./videos/flappy_bird_epoch_11460.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11480, Total Reward: 17.3, Epsilon: 0.1000
Saved video for epoch 11480 to ./videos/flappy_bird_epoch_11480.mp4
Epoch 11500, Total Reward: -1.199999999999998, Epsilon: 0.1000
Saved video for epoch 11500 to ./videos/flappy_bird_epoch_11500.mp4
Epoch 11520, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 11520 to ./videos/flappy_bird_epoch_11520.mp4
Epoch 11540, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 11540 to ./videos/flappy_bird_epoch_11540.mp4
The bird got through a pipe
Epoch 11560, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 11560 to ./videos/flappy_bird_epoch_11560.mp4
Epoch 11580, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 11580 to ./videos/flappy_bird_epoch_11580.mp4
Epoch 11600, Total Reward: 1.7763568394002505e-15, Epsilon: 0.1000
Saved video for epoch 11600 to ./videos/flappy_bird_epoch_11600.mp4
Epoch 11620, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11620 to ./videos/flappy_bird_epoch_11620.mp4
Epoch 11640, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 11640 to ./videos/flappy_bird_epoch_11640.mp4
Epoch 11660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11660 to ./videos/flappy_bird_epoch_11660.mp4
The bird got through a pipe
Epoch 11680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11680 to ./videos/flappy_bird_epoch_11680.mp4
The bird got through a pipe
Epoch 11700, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 11700 to ./videos/flappy_bird_epoch_11700.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 11720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11720 to ./videos/flappy_bird_epoch_11720.mp4
The bird got through a pipe
Epoch 11740, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11740 to ./videos/flappy_bird_epoch_11740.mp4
The bird got through a pipe
Epoch 11760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11760 to ./videos/flappy_bird_epoch_11760.mp4
The bird got through a pipe
Epoch 11780, Total Reward: 1.2000000000000015, Epsilon: 0.1000
Saved video for epoch 11780 to ./videos/flappy_bird_epoch_11780.mp4
Epoch 11800, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11800 to ./videos/flappy_bird_epoch_11800.mp4
Epoch 11820, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 11820 to ./videos/flappy_bird_epoch_11820.mp4
Epoch 11840, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 11840 to ./videos/flappy_bird_epoch_11840.mp4
Epoch 11860, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 11860 to ./videos/flappy_bird_epoch_11860.mp4
Epoch 11880, Total Reward: 8.881784197001252e-16, Epsilon: 0.1000
Saved video for epoch 11880 to ./videos/flappy_bird_epoch_11880.mp4
Epoch 11900, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 11900 to ./videos/flappy_bird_epoch_11900.mp4
Epoch 11920, Total Reward: -1.9999999999999987, Epsilon: 0.1000
Saved video for epoch 11920 to ./videos/flappy_bird_epoch_11920.mp4
Epoch 11940, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 11940 to ./videos/flappy_bird_epoch_11940.mp4
Epoch 11960, Total Reward: -0.4999999999999982, Epsilon: 0.1000
Saved video for epoch 11960 to ./videos/flappy_bird_epoch_11960.mp4
Epoch 11980, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 11980 to ./videos/flappy_bird_epoch_11980.mp4
Epoch 12000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12000 to ./videos/flappy_bird_epoch_12000.mp4
Epoch 12020, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 12020 to ./videos/flappy_bird_epoch_12020.mp4
The bird got through a pipe
Epoch 12040, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 12040 to ./videos/flappy_bird_epoch_12040.mp4
Epoch 12060, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 12060 to ./videos/flappy_bird_epoch_12060.mp4
Epoch 12080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12080 to ./videos/flappy_bird_epoch_12080.mp4
Epoch 12100, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12100 to ./videos/flappy_bird_epoch_12100.mp4
The bird got through a pipe
Epoch 12120, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 12120 to ./videos/flappy_bird_epoch_12120.mp4
Epoch 12140, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 12140 to ./videos/flappy_bird_epoch_12140.mp4
Epoch 12160, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 12160 to ./videos/flappy_bird_epoch_12160.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 12180, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 12180 to ./videos/flappy_bird_epoch_12180.mp4
Epoch 12200, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 12200 to ./videos/flappy_bird_epoch_12200.mp4
Epoch 12220, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 12220 to ./videos/flappy_bird_epoch_12220.mp4
Epoch 12240, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12240 to ./videos/flappy_bird_epoch_12240.mp4
Epoch 12260, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 12260 to ./videos/flappy_bird_epoch_12260.mp4
Epoch 12280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12280 to ./videos/flappy_bird_epoch_12280.mp4
Epoch 12300, Total Reward: -0.19999999999999796, Epsilon: 0.1000
Saved video for epoch 12300 to ./videos/flappy_bird_epoch_12300.mp4
Epoch 12320, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 12320 to ./videos/flappy_bird_epoch_12320.mp4
Epoch 12340, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 12340 to ./videos/flappy_bird_epoch_12340.mp4
The bird got through a pipe
Epoch 12360, Total Reward: 17.400000000000002, Epsilon: 0.1000
Saved video for epoch 12360 to ./videos/flappy_bird_epoch_12360.mp4
Epoch 12380, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12380 to ./videos/flappy_bird_epoch_12380.mp4
Epoch 12400, Total Reward: 2.1999999999999993, Epsilon: 0.1000
Saved video for epoch 12400 to ./videos/flappy_bird_epoch_12400.mp4
Epoch 12420, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 12420 to ./videos/flappy_bird_epoch_12420.mp4
Epoch 12440, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 12440 to ./videos/flappy_bird_epoch_12440.mp4
Epoch 12460, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12460 to ./videos/flappy_bird_epoch_12460.mp4
Epoch 12480, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 12480 to ./videos/flappy_bird_epoch_12480.mp4
Epoch 12500, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12500 to ./videos/flappy_bird_epoch_12500.mp4
Epoch 12520, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 12520 to ./videos/flappy_bird_epoch_12520.mp4
Epoch 12540, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 12540 to ./videos/flappy_bird_epoch_12540.mp4
Epoch 12560, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 12560 to ./videos/flappy_bird_epoch_12560.mp4
Epoch 12580, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 12580 to ./videos/flappy_bird_epoch_12580.mp4
Epoch 12600, Total Reward: -7.499999999999998, Epsilon: 0.1000
Saved video for epoch 12600 to ./videos/flappy_bird_epoch_12600.mp4
The bird got through a pipe
Epoch 12620, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 12620 to ./videos/flappy_bird_epoch_12620.mp4
Epoch 12640, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 12640 to ./videos/flappy_bird_epoch_12640.mp4
Epoch 12660, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 12660 to ./videos/flappy_bird_epoch_12660.mp4
Epoch 12680, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 12680 to ./videos/flappy_bird_epoch_12680.mp4
Epoch 12700, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 12700 to ./videos/flappy_bird_epoch_12700.mp4
Epoch 12720, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 12720 to ./videos/flappy_bird_epoch_12720.mp4
Epoch 12740, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 12740 to ./videos/flappy_bird_epoch_12740.mp4
Epoch 12760, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 12760 to ./videos/flappy_bird_epoch_12760.mp4
Epoch 12780, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 12780 to ./videos/flappy_bird_epoch_12780.mp4
Epoch 12800, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 12800 to ./videos/flappy_bird_epoch_12800.mp4
Epoch 12820, Total Reward: -3.0999999999999988, Epsilon: 0.1000
Saved video for epoch 12820 to ./videos/flappy_bird_epoch_12820.mp4
Epoch 12840, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12840 to ./videos/flappy_bird_epoch_12840.mp4
Epoch 12860, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12860 to ./videos/flappy_bird_epoch_12860.mp4
Epoch 12880, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12880 to ./videos/flappy_bird_epoch_12880.mp4
Epoch 12900, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12900 to ./videos/flappy_bird_epoch_12900.mp4
Epoch 12920, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12920 to ./videos/flappy_bird_epoch_12920.mp4
Epoch 12940, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12940 to ./videos/flappy_bird_epoch_12940.mp4
Epoch 12960, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12960 to ./videos/flappy_bird_epoch_12960.mp4
Epoch 12980, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 12980 to ./videos/flappy_bird_epoch_12980.mp4
Epoch 13000, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13000 to ./videos/flappy_bird_epoch_13000.mp4
Epoch 13020, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 13020 to ./videos/flappy_bird_epoch_13020.mp4
Epoch 13040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13040 to ./videos/flappy_bird_epoch_13040.mp4
Epoch 13060, Total Reward: 0.8000000000000025, Epsilon: 0.1000
Saved video for epoch 13060 to ./videos/flappy_bird_epoch_13060.mp4
Epoch 13080, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 13080 to ./videos/flappy_bird_epoch_13080.mp4
Epoch 13100, Total Reward: -5.099999999999998, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 13100 to ./videos/flappy_bird_epoch_13100.mp4
Epoch 13120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13120 to ./videos/flappy_bird_epoch_13120.mp4
Epoch 13140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13140 to ./videos/flappy_bird_epoch_13140.mp4
The bird got through a pipe
Epoch 13160, Total Reward: 1.0000000000000018, Epsilon: 0.1000
Saved video for epoch 13160 to ./videos/flappy_bird_epoch_13160.mp4
Epoch 13180, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 13180 to ./videos/flappy_bird_epoch_13180.mp4
The bird got through a pipe
Epoch 13200, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 13200 to ./videos/flappy_bird_epoch_13200.mp4
Epoch 13220, Total Reward: 1.5999999999999996, Epsilon: 0.1000
Saved video for epoch 13220 to ./videos/flappy_bird_epoch_13220.mp4
Epoch 13240, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13240 to ./videos/flappy_bird_epoch_13240.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13260, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13260 to ./videos/flappy_bird_epoch_13260.mp4
The bird got through a pipe
Epoch 13280, Total Reward: -0.9999999999999987, Epsilon: 0.1000
Saved video for epoch 13280 to ./videos/flappy_bird_epoch_13280.mp4
Epoch 13300, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13300 to ./videos/flappy_bird_epoch_13300.mp4
Epoch 13320, Total Reward: -1.4999999999999987, Epsilon: 0.1000
Saved video for epoch 13320 to ./videos/flappy_bird_epoch_13320.mp4
The bird got through a pipe
Epoch 13340, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13340 to ./videos/flappy_bird_epoch_13340.mp4
Epoch 13360, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13360 to ./videos/flappy_bird_epoch_13360.mp4
The bird got through a pipe
Epoch 13380, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13380 to ./videos/flappy_bird_epoch_13380.mp4
The bird got through a pipe
Epoch 13400, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 13400 to ./videos/flappy_bird_epoch_13400.mp4
The bird got through a pipe
Epoch 13420, Total Reward: 1.7763568394002505e-15, Epsilon: 0.1000
Saved video for epoch 13420 to ./videos/flappy_bird_epoch_13420.mp4
Epoch 13440, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13440 to ./videos/flappy_bird_epoch_13440.mp4
Epoch 13460, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13460 to ./videos/flappy_bird_epoch_13460.mp4
The bird got through a pipe
Epoch 13480, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13480 to ./videos/flappy_bird_epoch_13480.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13500, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 13500 to ./videos/flappy_bird_epoch_13500.mp4
Epoch 13520, Total Reward: -1.4999999999999987, Epsilon: 0.1000
Saved video for epoch 13520 to ./videos/flappy_bird_epoch_13520.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13540, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13540 to ./videos/flappy_bird_epoch_13540.mp4
The bird got through a pipe
Epoch 13560, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13560 to ./videos/flappy_bird_epoch_13560.mp4
The bird got through a pipe
Epoch 13580, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13580 to ./videos/flappy_bird_epoch_13580.mp4
Epoch 13600, Total Reward: -1.299999999999998, Epsilon: 0.1000
Saved video for epoch 13600 to ./videos/flappy_bird_epoch_13600.mp4
Epoch 13620, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13620 to ./videos/flappy_bird_epoch_13620.mp4
Epoch 13640, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13640 to ./videos/flappy_bird_epoch_13640.mp4
The bird got through a pipe
Epoch 13660, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 13660 to ./videos/flappy_bird_epoch_13660.mp4
Epoch 13680, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 13680 to ./videos/flappy_bird_epoch_13680.mp4
The bird got through a pipe
Epoch 13700, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13700 to ./videos/flappy_bird_epoch_13700.mp4
Epoch 13720, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 13720 to ./videos/flappy_bird_epoch_13720.mp4
Epoch 13740, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 13740 to ./videos/flappy_bird_epoch_13740.mp4
The bird got through a pipe
Epoch 13760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 13760 to ./videos/flappy_bird_epoch_13760.mp4
Epoch 13780, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 13780 to ./videos/flappy_bird_epoch_13780.mp4
Epoch 13800, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13800 to ./videos/flappy_bird_epoch_13800.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13820 to ./videos/flappy_bird_epoch_13820.mp4
The bird got through a pipe
Epoch 13840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13840 to ./videos/flappy_bird_epoch_13840.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 13860, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 13860 to ./videos/flappy_bird_epoch_13860.mp4
The bird got through a pipe
Epoch 13880, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 13880 to ./videos/flappy_bird_epoch_13880.mp4
Epoch 13900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
The bird got through a pipe
Saved video for epoch 13900 to ./videos/flappy_bird_epoch_13900.mp4
The bird got through a pipe
Epoch 13920, Total Reward: 1.6999999999999993, Epsilon: 0.1000
Saved video for epoch 13920 to ./videos/flappy_bird_epoch_13920.mp4
The bird got through a pipe
Epoch 13940, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 13940 to ./videos/flappy_bird_epoch_13940.mp4
The bird got through a pipe
Epoch 13960, Total Reward: -0.6999999999999991, Epsilon: 0.1000
Saved video for epoch 13960 to ./videos/flappy_bird_epoch_13960.mp4
Epoch 13980, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 13980 to ./videos/flappy_bird_epoch_13980.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 14000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14000 to ./videos/flappy_bird_epoch_14000.mp4
Epoch 14020, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 14020 to ./videos/flappy_bird_epoch_14020.mp4
Epoch 14040, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14040 to ./videos/flappy_bird_epoch_14040.mp4
Epoch 14060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14060 to ./videos/flappy_bird_epoch_14060.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 14080, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 14080 to ./videos/flappy_bird_epoch_14080.mp4
The bird got through a pipe
Epoch 14100, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14100 to ./videos/flappy_bird_epoch_14100.mp4
Epoch 14120, Total Reward: 0.6000000000000019, Epsilon: 0.1000
Saved video for epoch 14120 to ./videos/flappy_bird_epoch_14120.mp4
Epoch 14140, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14140 to ./videos/flappy_bird_epoch_14140.mp4
Epoch 14160, Total Reward: 0.40000000000000213, Epsilon: 0.1000
Saved video for epoch 14160 to ./videos/flappy_bird_epoch_14160.mp4
Epoch 14180, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14180 to ./videos/flappy_bird_epoch_14180.mp4
The bird got through a pipe
Epoch 14200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14200 to ./videos/flappy_bird_epoch_14200.mp4
Epoch 14220, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14220 to ./videos/flappy_bird_epoch_14220.mp4
The bird got through a pipe
Epoch 14240, Total Reward: 1.4000000000000017, Epsilon: 0.1000
Saved video for epoch 14240 to ./videos/flappy_bird_epoch_14240.mp4
Epoch 14260, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14260 to ./videos/flappy_bird_epoch_14260.mp4
Epoch 14280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14280 to ./videos/flappy_bird_epoch_14280.mp4
Epoch 14300, Total Reward: 2.799999999999999, Epsilon: 0.1000
Saved video for epoch 14300 to ./videos/flappy_bird_epoch_14300.mp4
Epoch 14320, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14320 to ./videos/flappy_bird_epoch_14320.mp4
Epoch 14340, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14340 to ./videos/flappy_bird_epoch_14340.mp4
Epoch 14360, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 14360 to ./videos/flappy_bird_epoch_14360.mp4
Epoch 14380, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 14380 to ./videos/flappy_bird_epoch_14380.mp4
Epoch 14400, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14400 to ./videos/flappy_bird_epoch_14400.mp4
Epoch 14420, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 14420 to ./videos/flappy_bird_epoch_14420.mp4
The bird got through a pipe
Epoch 14440, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14440 to ./videos/flappy_bird_epoch_14440.mp4
Epoch 14460, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14460 to ./videos/flappy_bird_epoch_14460.mp4
Epoch 14480, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 14480 to ./videos/flappy_bird_epoch_14480.mp4
Epoch 14500, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 14500 to ./videos/flappy_bird_epoch_14500.mp4
The bird got through a pipe
Epoch 14520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14520 to ./videos/flappy_bird_epoch_14520.mp4
Epoch 14540, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 14540 to ./videos/flappy_bird_epoch_14540.mp4
Epoch 14560, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 14560 to ./videos/flappy_bird_epoch_14560.mp4
Epoch 14580, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14580 to ./videos/flappy_bird_epoch_14580.mp4
Epoch 14600, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 14600 to ./videos/flappy_bird_epoch_14600.mp4
Epoch 14620, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14620 to ./videos/flappy_bird_epoch_14620.mp4
Epoch 14640, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14640 to ./videos/flappy_bird_epoch_14640.mp4
Epoch 14660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14660 to ./videos/flappy_bird_epoch_14660.mp4
Epoch 14680, Total Reward: 0.6000000000000014, Epsilon: 0.1000
Saved video for epoch 14680 to ./videos/flappy_bird_epoch_14680.mp4
Epoch 14700, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14700 to ./videos/flappy_bird_epoch_14700.mp4
Epoch 14720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14720 to ./videos/flappy_bird_epoch_14720.mp4
Epoch 14740, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14740 to ./videos/flappy_bird_epoch_14740.mp4
The bird got through a pipe
The bird got through a pipe
Epoch 14760, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14760 to ./videos/flappy_bird_epoch_14760.mp4
Epoch 14780, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14780 to ./videos/flappy_bird_epoch_14780.mp4
Epoch 14800, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14800 to ./videos/flappy_bird_epoch_14800.mp4
Epoch 14820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14820 to ./videos/flappy_bird_epoch_14820.mp4
Epoch 14840, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14840 to ./videos/flappy_bird_epoch_14840.mp4
Epoch 14860, Total Reward: -3.4999999999999987, Epsilon: 0.1000
Saved video for epoch 14860 to ./videos/flappy_bird_epoch_14860.mp4
The bird got through a pipe
Epoch 14880, Total Reward: 1.6999999999999993, Epsilon: 0.1000
Saved video for epoch 14880 to ./videos/flappy_bird_epoch_14880.mp4
Epoch 14900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14900 to ./videos/flappy_bird_epoch_14900.mp4
Epoch 14920, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14920 to ./videos/flappy_bird_epoch_14920.mp4
Epoch 14940, Total Reward: 1.9000000000000021, Epsilon: 0.1000
Saved video for epoch 14940 to ./videos/flappy_bird_epoch_14940.mp4
Epoch 14960, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 14960 to ./videos/flappy_bird_epoch_14960.mp4
Epoch 14980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 14980 to ./videos/flappy_bird_epoch_14980.mp4
Epoch 15000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15000 to ./videos/flappy_bird_epoch_15000.mp4
Epoch 15020, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 15020 to ./videos/flappy_bird_epoch_15020.mp4
Epoch 15040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15040 to ./videos/flappy_bird_epoch_15040.mp4
Epoch 15060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15060 to ./videos/flappy_bird_epoch_15060.mp4
The bird got through a pipe
Epoch 15080, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 15080 to ./videos/flappy_bird_epoch_15080.mp4
Epoch 15100, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 15100 to ./videos/flappy_bird_epoch_15100.mp4
Epoch 15120, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15120 to ./videos/flappy_bird_epoch_15120.mp4
Epoch 15140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15140 to ./videos/flappy_bird_epoch_15140.mp4
Epoch 15160, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15160 to ./videos/flappy_bird_epoch_15160.mp4
Epoch 15180, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15180 to ./videos/flappy_bird_epoch_15180.mp4
Epoch 15200, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15200 to ./videos/flappy_bird_epoch_15200.mp4
Epoch 15220, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15220 to ./videos/flappy_bird_epoch_15220.mp4
Epoch 15240, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15240 to ./videos/flappy_bird_epoch_15240.mp4
Epoch 15260, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 15260 to ./videos/flappy_bird_epoch_15260.mp4
Epoch 15280, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15280 to ./videos/flappy_bird_epoch_15280.mp4
Epoch 15300, Total Reward: 1.0000000000000018, Epsilon: 0.1000
Saved video for epoch 15300 to ./videos/flappy_bird_epoch_15300.mp4
Epoch 15320, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15320 to ./videos/flappy_bird_epoch_15320.mp4
Epoch 15340, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15340 to ./videos/flappy_bird_epoch_15340.mp4
Epoch 15360, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 15360 to ./videos/flappy_bird_epoch_15360.mp4
Epoch 15380, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15380 to ./videos/flappy_bird_epoch_15380.mp4
Epoch 15400, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 15400 to ./videos/flappy_bird_epoch_15400.mp4
Epoch 15420, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 15420 to ./videos/flappy_bird_epoch_15420.mp4
Epoch 15440, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15440 to ./videos/flappy_bird_epoch_15440.mp4
Epoch 15460, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15460 to ./videos/flappy_bird_epoch_15460.mp4
Epoch 15480, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 15480 to ./videos/flappy_bird_epoch_15480.mp4
Epoch 15500, Total Reward: -1.4999999999999978, Epsilon: 0.1000
Saved video for epoch 15500 to ./videos/flappy_bird_epoch_15500.mp4
Epoch 15520, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15520 to ./videos/flappy_bird_epoch_15520.mp4
Epoch 15540, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 15540 to ./videos/flappy_bird_epoch_15540.mp4
Epoch 15560, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15560 to ./videos/flappy_bird_epoch_15560.mp4
Epoch 15580, Total Reward: 2.0, Epsilon: 0.1000
Saved video for epoch 15580 to ./videos/flappy_bird_epoch_15580.mp4
Epoch 15600, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15600 to ./videos/flappy_bird_epoch_15600.mp4
Epoch 15620, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15620 to ./videos/flappy_bird_epoch_15620.mp4
Epoch 15640, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 15640 to ./videos/flappy_bird_epoch_15640.mp4
Epoch 15660, Total Reward: 0.10000000000000142, Epsilon: 0.1000
Saved video for epoch 15660 to ./videos/flappy_bird_epoch_15660.mp4
Epoch 15680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15680 to ./videos/flappy_bird_epoch_15680.mp4
Epoch 15700, Total Reward: 0.20000000000000107, Epsilon: 0.1000
Saved video for epoch 15700 to ./videos/flappy_bird_epoch_15700.mp4
Epoch 15720, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15720 to ./videos/flappy_bird_epoch_15720.mp4
Epoch 15740, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 15740 to ./videos/flappy_bird_epoch_15740.mp4
Epoch 15760, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15760 to ./videos/flappy_bird_epoch_15760.mp4
Epoch 15780, Total Reward: -1.4999999999999978, Epsilon: 0.1000
Saved video for epoch 15780 to ./videos/flappy_bird_epoch_15780.mp4
Epoch 15800, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15800 to ./videos/flappy_bird_epoch_15800.mp4
Epoch 15820, Total Reward: 1.800000000000002, Epsilon: 0.1000
Saved video for epoch 15820 to ./videos/flappy_bird_epoch_15820.mp4
Epoch 15840, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15840 to ./videos/flappy_bird_epoch_15840.mp4
Epoch 15860, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 15860 to ./videos/flappy_bird_epoch_15860.mp4
Epoch 15880, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15880 to ./videos/flappy_bird_epoch_15880.mp4
Epoch 15900, Total Reward: -0.3999999999999986, Epsilon: 0.1000
Saved video for epoch 15900 to ./videos/flappy_bird_epoch_15900.mp4
Epoch 15920, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15920 to ./videos/flappy_bird_epoch_15920.mp4
Epoch 15940, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15940 to ./videos/flappy_bird_epoch_15940.mp4
Epoch 15960, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 15960 to ./videos/flappy_bird_epoch_15960.mp4
Epoch 15980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 15980 to ./videos/flappy_bird_epoch_15980.mp4
Epoch 16000, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16000 to ./videos/flappy_bird_epoch_16000.mp4
Epoch 16020, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16020 to ./videos/flappy_bird_epoch_16020.mp4
Epoch 16040, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16040 to ./videos/flappy_bird_epoch_16040.mp4
Epoch 16060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16060 to ./videos/flappy_bird_epoch_16060.mp4
Epoch 16080, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16080 to ./videos/flappy_bird_epoch_16080.mp4
Epoch 16100, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16100 to ./videos/flappy_bird_epoch_16100.mp4
Epoch 16120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16120 to ./videos/flappy_bird_epoch_16120.mp4
Epoch 16140, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16140 to ./videos/flappy_bird_epoch_16140.mp4
Epoch 16160, Total Reward: -3.299999999999998, Epsilon: 0.1000
Saved video for epoch 16160 to ./videos/flappy_bird_epoch_16160.mp4
Epoch 16180, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16180 to ./videos/flappy_bird_epoch_16180.mp4
Epoch 16200, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 16200 to ./videos/flappy_bird_epoch_16200.mp4
Epoch 16220, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16220 to ./videos/flappy_bird_epoch_16220.mp4
Epoch 16240, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16240 to ./videos/flappy_bird_epoch_16240.mp4
Epoch 16260, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 16260 to ./videos/flappy_bird_epoch_16260.mp4
Epoch 16280, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16280 to ./videos/flappy_bird_epoch_16280.mp4
Epoch 16300, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16300 to ./videos/flappy_bird_epoch_16300.mp4
Epoch 16320, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 16320 to ./videos/flappy_bird_epoch_16320.mp4
Epoch 16340, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16340 to ./videos/flappy_bird_epoch_16340.mp4
Epoch 16360, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16360 to ./videos/flappy_bird_epoch_16360.mp4
Epoch 16380, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16380 to ./videos/flappy_bird_epoch_16380.mp4
Epoch 16400, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 16400 to ./videos/flappy_bird_epoch_16400.mp4
Epoch 16420, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16420 to ./videos/flappy_bird_epoch_16420.mp4
Epoch 16440, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16440 to ./videos/flappy_bird_epoch_16440.mp4
Epoch 16460, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16460 to ./videos/flappy_bird_epoch_16460.mp4
Epoch 16480, Total Reward: -2.4999999999999987, Epsilon: 0.1000
Saved video for epoch 16480 to ./videos/flappy_bird_epoch_16480.mp4
Epoch 16500, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16500 to ./videos/flappy_bird_epoch_16500.mp4
Epoch 16520, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16520 to ./videos/flappy_bird_epoch_16520.mp4
Epoch 16540, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 16540 to ./videos/flappy_bird_epoch_16540.mp4
Epoch 16560, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 16560 to ./videos/flappy_bird_epoch_16560.mp4
Epoch 16580, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16580 to ./videos/flappy_bird_epoch_16580.mp4
The bird got through a pipe
Epoch 16600, Total Reward: 19.0, Epsilon: 0.1000
Saved video for epoch 16600 to ./videos/flappy_bird_epoch_16600.mp4
Epoch 16620, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16620 to ./videos/flappy_bird_epoch_16620.mp4
Epoch 16640, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16640 to ./videos/flappy_bird_epoch_16640.mp4
Epoch 16660, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 16660 to ./videos/flappy_bird_epoch_16660.mp4
Epoch 16680, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16680 to ./videos/flappy_bird_epoch_16680.mp4
Epoch 16700, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16700 to ./videos/flappy_bird_epoch_16700.mp4
Epoch 16720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16720 to ./videos/flappy_bird_epoch_16720.mp4
The bird got through a pipe
Epoch 16740, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16740 to ./videos/flappy_bird_epoch_16740.mp4
Epoch 16760, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16760 to ./videos/flappy_bird_epoch_16760.mp4
Epoch 16780, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 16780 to ./videos/flappy_bird_epoch_16780.mp4
Epoch 16800, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16800 to ./videos/flappy_bird_epoch_16800.mp4
Epoch 16820, Total Reward: 1.2000000000000015, Epsilon: 0.1000
Saved video for epoch 16820 to ./videos/flappy_bird_epoch_16820.mp4
Epoch 16840, Total Reward: 2.4999999999999982, Epsilon: 0.1000
Saved video for epoch 16840 to ./videos/flappy_bird_epoch_16840.mp4
Epoch 16860, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16860 to ./videos/flappy_bird_epoch_16860.mp4
Epoch 16880, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16880 to ./videos/flappy_bird_epoch_16880.mp4
Epoch 16900, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16900 to ./videos/flappy_bird_epoch_16900.mp4
Epoch 16920, Total Reward: -1.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16920 to ./videos/flappy_bird_epoch_16920.mp4
Epoch 16940, Total Reward: -0.0999999999999992, Epsilon: 0.1000
Saved video for epoch 16940 to ./videos/flappy_bird_epoch_16940.mp4
Epoch 16960, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 16960 to ./videos/flappy_bird_epoch_16960.mp4
Epoch 16980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 16980 to ./videos/flappy_bird_epoch_16980.mp4
Epoch 17000, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17000 to ./videos/flappy_bird_epoch_17000.mp4
Epoch 17020, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 17020 to ./videos/flappy_bird_epoch_17020.mp4
Epoch 17040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17040 to ./videos/flappy_bird_epoch_17040.mp4
Epoch 17060, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17060 to ./videos/flappy_bird_epoch_17060.mp4
Epoch 17080, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17080 to ./videos/flappy_bird_epoch_17080.mp4
Epoch 17100, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 17100 to ./videos/flappy_bird_epoch_17100.mp4
Epoch 17120, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 17120 to ./videos/flappy_bird_epoch_17120.mp4
The bird got through a pipe
Epoch 17140, Total Reward: 0.3000000000000007, Epsilon: 0.1000
Saved video for epoch 17140 to ./videos/flappy_bird_epoch_17140.mp4
Epoch 17160, Total Reward: 1.1000000000000014, Epsilon: 0.1000
Saved video for epoch 17160 to ./videos/flappy_bird_epoch_17160.mp4
Epoch 17180, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17180 to ./videos/flappy_bird_epoch_17180.mp4
Epoch 17200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17200 to ./videos/flappy_bird_epoch_17200.mp4
Epoch 17220, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17220 to ./videos/flappy_bird_epoch_17220.mp4
Epoch 17240, Total Reward: 0.8000000000000012, Epsilon: 0.1000
Saved video for epoch 17240 to ./videos/flappy_bird_epoch_17240.mp4
Epoch 17260, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17260 to ./videos/flappy_bird_epoch_17260.mp4
Epoch 17280, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17280 to ./videos/flappy_bird_epoch_17280.mp4
Epoch 17300, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17300 to ./videos/flappy_bird_epoch_17300.mp4
Epoch 17320, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17320 to ./videos/flappy_bird_epoch_17320.mp4
Epoch 17340, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 17340 to ./videos/flappy_bird_epoch_17340.mp4
Epoch 17360, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17360 to ./videos/flappy_bird_epoch_17360.mp4
Epoch 17380, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 17380 to ./videos/flappy_bird_epoch_17380.mp4
Epoch 17400, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17400 to ./videos/flappy_bird_epoch_17400.mp4
Epoch 17420, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 17420 to ./videos/flappy_bird_epoch_17420.mp4
Epoch 17440, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 17440 to ./videos/flappy_bird_epoch_17440.mp4
Epoch 17460, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17460 to ./videos/flappy_bird_epoch_17460.mp4
Epoch 17480, Total Reward: 1.0000000000000013, Epsilon: 0.1000
Saved video for epoch 17480 to ./videos/flappy_bird_epoch_17480.mp4
Epoch 17500, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17500 to ./videos/flappy_bird_epoch_17500.mp4
Epoch 17520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17520 to ./videos/flappy_bird_epoch_17520.mp4
Epoch 17540, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17540 to ./videos/flappy_bird_epoch_17540.mp4
Epoch 17560, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17560 to ./videos/flappy_bird_epoch_17560.mp4
Epoch 17580, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17580 to ./videos/flappy_bird_epoch_17580.mp4
Epoch 17600, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 17600 to ./videos/flappy_bird_epoch_17600.mp4
Epoch 17620, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17620 to ./videos/flappy_bird_epoch_17620.mp4
Epoch 17640, Total Reward: 2.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17640 to ./videos/flappy_bird_epoch_17640.mp4
Epoch 17660, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17660 to ./videos/flappy_bird_epoch_17660.mp4
Epoch 17680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17680 to ./videos/flappy_bird_epoch_17680.mp4
Epoch 17700, Total Reward: 2.6999999999999993, Epsilon: 0.1000
Saved video for epoch 17700 to ./videos/flappy_bird_epoch_17700.mp4
Epoch 17720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17720 to ./videos/flappy_bird_epoch_17720.mp4
Epoch 17740, Total Reward: 2.3999999999999986, Epsilon: 0.1000
Saved video for epoch 17740 to ./videos/flappy_bird_epoch_17740.mp4
Epoch 17760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17760 to ./videos/flappy_bird_epoch_17760.mp4
Epoch 17780, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 17780 to ./videos/flappy_bird_epoch_17780.mp4
Epoch 17800, Total Reward: -2.3999999999999986, Epsilon: 0.1000
Saved video for epoch 17800 to ./videos/flappy_bird_epoch_17800.mp4
Epoch 17820, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 17820 to ./videos/flappy_bird_epoch_17820.mp4
Epoch 17840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17840 to ./videos/flappy_bird_epoch_17840.mp4
Epoch 17860, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 17860 to ./videos/flappy_bird_epoch_17860.mp4
Epoch 17880, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17880 to ./videos/flappy_bird_epoch_17880.mp4
Epoch 17900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17900 to ./videos/flappy_bird_epoch_17900.mp4
Epoch 17920, Total Reward: -6.299999999999999, Epsilon: 0.1000
Saved video for epoch 17920 to ./videos/flappy_bird_epoch_17920.mp4
Epoch 17940, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 17940 to ./videos/flappy_bird_epoch_17940.mp4
Epoch 17960, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 17960 to ./videos/flappy_bird_epoch_17960.mp4
Epoch 17980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 17980 to ./videos/flappy_bird_epoch_17980.mp4
Epoch 18000, Total Reward: -0.0999999999999992, Epsilon: 0.1000
Saved video for epoch 18000 to ./videos/flappy_bird_epoch_18000.mp4
Epoch 18020, Total Reward: 1.700000000000002, Epsilon: 0.1000
Saved video for epoch 18020 to ./videos/flappy_bird_epoch_18020.mp4
Epoch 18040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18040 to ./videos/flappy_bird_epoch_18040.mp4
Epoch 18060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18060 to ./videos/flappy_bird_epoch_18060.mp4
Epoch 18080, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 18080 to ./videos/flappy_bird_epoch_18080.mp4
Epoch 18100, Total Reward: 2.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18100 to ./videos/flappy_bird_epoch_18100.mp4
Epoch 18120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18120 to ./videos/flappy_bird_epoch_18120.mp4
Epoch 18140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18140 to ./videos/flappy_bird_epoch_18140.mp4
Epoch 18160, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18160 to ./videos/flappy_bird_epoch_18160.mp4
Epoch 18180, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18180 to ./videos/flappy_bird_epoch_18180.mp4
Epoch 18200, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18200 to ./videos/flappy_bird_epoch_18200.mp4
Epoch 18220, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18220 to ./videos/flappy_bird_epoch_18220.mp4
Epoch 18240, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18240 to ./videos/flappy_bird_epoch_18240.mp4
Epoch 18260, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 18260 to ./videos/flappy_bird_epoch_18260.mp4
Epoch 18280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18280 to ./videos/flappy_bird_epoch_18280.mp4
Epoch 18300, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 18300 to ./videos/flappy_bird_epoch_18300.mp4
Epoch 18320, Total Reward: -5.699999999999999, Epsilon: 0.1000
Saved video for epoch 18320 to ./videos/flappy_bird_epoch_18320.mp4
Epoch 18340, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 18340 to ./videos/flappy_bird_epoch_18340.mp4
Epoch 18360, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 18360 to ./videos/flappy_bird_epoch_18360.mp4
Epoch 18380, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18380 to ./videos/flappy_bird_epoch_18380.mp4
Epoch 18400, Total Reward: 2.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18400 to ./videos/flappy_bird_epoch_18400.mp4
Epoch 18420, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 18420 to ./videos/flappy_bird_epoch_18420.mp4
Epoch 18440, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18440 to ./videos/flappy_bird_epoch_18440.mp4
Epoch 18460, Total Reward: 1.1000000000000019, Epsilon: 0.1000
Saved video for epoch 18460 to ./videos/flappy_bird_epoch_18460.mp4
Epoch 18480, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18480 to ./videos/flappy_bird_epoch_18480.mp4
Epoch 18500, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18500 to ./videos/flappy_bird_epoch_18500.mp4
Epoch 18520, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18520 to ./videos/flappy_bird_epoch_18520.mp4
Epoch 18540, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18540 to ./videos/flappy_bird_epoch_18540.mp4
Epoch 18560, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18560 to ./videos/flappy_bird_epoch_18560.mp4
Epoch 18580, Total Reward: -5.099999999999998, Epsilon: 0.1000
Saved video for epoch 18580 to ./videos/flappy_bird_epoch_18580.mp4
Epoch 18600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18600 to ./videos/flappy_bird_epoch_18600.mp4
Epoch 18620, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 18620 to ./videos/flappy_bird_epoch_18620.mp4
Epoch 18640, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 18640 to ./videos/flappy_bird_epoch_18640.mp4
Epoch 18660, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 18660 to ./videos/flappy_bird_epoch_18660.mp4
Epoch 18680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18680 to ./videos/flappy_bird_epoch_18680.mp4
Epoch 18700, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18700 to ./videos/flappy_bird_epoch_18700.mp4
Epoch 18720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18720 to ./videos/flappy_bird_epoch_18720.mp4
Epoch 18740, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 18740 to ./videos/flappy_bird_epoch_18740.mp4
Epoch 18760, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 18760 to ./videos/flappy_bird_epoch_18760.mp4
Epoch 18780, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18780 to ./videos/flappy_bird_epoch_18780.mp4
Epoch 18800, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18800 to ./videos/flappy_bird_epoch_18800.mp4
Epoch 18820, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18820 to ./videos/flappy_bird_epoch_18820.mp4
Epoch 18840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18840 to ./videos/flappy_bird_epoch_18840.mp4
Epoch 18860, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 18860 to ./videos/flappy_bird_epoch_18860.mp4
Epoch 18880, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 18880 to ./videos/flappy_bird_epoch_18880.mp4
Epoch 18900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18900 to ./videos/flappy_bird_epoch_18900.mp4
Epoch 18920, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18920 to ./videos/flappy_bird_epoch_18920.mp4
Epoch 18940, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 18940 to ./videos/flappy_bird_epoch_18940.mp4
Epoch 18960, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 18960 to ./videos/flappy_bird_epoch_18960.mp4
Epoch 18980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 18980 to ./videos/flappy_bird_epoch_18980.mp4
Epoch 19000, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 19000 to ./videos/flappy_bird_epoch_19000.mp4
Epoch 19020, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19020 to ./videos/flappy_bird_epoch_19020.mp4
Epoch 19040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19040 to ./videos/flappy_bird_epoch_19040.mp4
Epoch 19060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19060 to ./videos/flappy_bird_epoch_19060.mp4
Epoch 19080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19080 to ./videos/flappy_bird_epoch_19080.mp4
Epoch 19020, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19020 to ./videos/flappy_bird_epoch_19020.mp4
Epoch 19040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19040 to ./videos/flappy_bird_epoch_19040.mp4
Epoch 19060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19060 to ./videos/flappy_bird_epoch_19060.mp4
Epoch 19080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19080 to ./videos/flappy_bird_epoch_19080.mp4
Epoch 19100, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Epoch 19020, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19020 to ./videos/flappy_bird_epoch_19020.mp4
Epoch 19040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19040 to ./videos/flappy_bird_epoch_19040.mp4
Epoch 19060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19060 to ./videos/flappy_bird_epoch_19060.mp4
Epoch 19080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19080 to ./videos/flappy_bird_epoch_19080.mp4
Epoch 19040, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19040 to ./videos/flappy_bird_epoch_19040.mp4
Epoch 19060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19060 to ./videos/flappy_bird_epoch_19060.mp4
Epoch 19080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19080 to ./videos/flappy_bird_epoch_19080.mp4
Epoch 19060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19060 to ./videos/flappy_bird_epoch_19060.mp4
Epoch 19080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19080 to ./videos/flappy_bird_epoch_19080.mp4
Epoch 19100, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 19100 to ./videos/flappy_bird_epoch_19100.mp4
Epoch 19120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19120 to ./videos/flappy_bird_epoch_19120.mp4
Epoch 19140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Epoch 19080, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19080 to ./videos/flappy_bird_epoch_19080.mp4
Epoch 19100, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 19100 to ./videos/flappy_bird_epoch_19100.mp4
Epoch 19120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19120 to ./videos/flappy_bird_epoch_19120.mp4
Epoch 19140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Epoch 19100, Total Reward: -0.2999999999999985, Epsilon: 0.1000
Saved video for epoch 19100 to ./videos/flappy_bird_epoch_19100.mp4
Epoch 19120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19120 to ./videos/flappy_bird_epoch_19120.mp4
Epoch 19140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Epoch 19120, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19120 to ./videos/flappy_bird_epoch_19120.mp4
Epoch 19140, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19140 to ./videos/flappy_bird_epoch_19140.mp4
Epoch 19160, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 19160 to ./videos/flappy_bird_epoch_19160.mp4
Saved video for epoch 19140 to ./videos/flappy_bird_epoch_19140.mp4
Epoch 19160, Total Reward: -2.099999999999998, Epsilon: 0.1000
Saved video for epoch 19160 to ./videos/flappy_bird_epoch_19160.mp4
Saved video for epoch 19160 to ./videos/flappy_bird_epoch_19160.mp4
Epoch 19180, Total Reward: 0.3000000000000016, Epsilon: 0.1000
Saved video for epoch 19180 to ./videos/flappy_bird_epoch_19180.mp4
Saved video for epoch 19180 to ./videos/flappy_bird_epoch_19180.mp4
Epoch 19200, Total Reward: -8.7, Epsilon: 0.1000
Saved video for epoch 19200 to ./videos/flappy_bird_epoch_19200.mp4
Epoch 19220, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19220 to ./videos/flappy_bird_epoch_19220.mp4
Epoch 19240, Total Reward: 2.0999999999999996, Epsilon: 0.1000
Saved video for epoch 19240 to ./videos/flappy_bird_epoch_19240.mp4
Epoch 19260, Total Reward: -1.4999999999999982, Epsilon: 0.1000
Saved video for epoch 19260 to ./videos/flappy_bird_epoch_19260.mp4
Epoch 19280, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19280 to ./videos/flappy_bird_epoch_19280.mp4
Epoch 19300, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 19300 to ./videos/flappy_bird_epoch_19300.mp4
Epoch 19320, Total Reward: 2.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19320 to ./videos/flappy_bird_epoch_19320.mp4
Epoch 19340, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19340 to ./videos/flappy_bird_epoch_19340.mp4
Epoch 19360, Total Reward: -3.899999999999998, Epsilon: 0.1000
Saved video for epoch 19360 to ./videos/flappy_bird_epoch_19360.mp4
Epoch 19380, Total Reward: 1.5, Epsilon: 0.1000
Saved video for epoch 19380 to ./videos/flappy_bird_epoch_19380.mp4
Epoch 19400, Total Reward: -4.499999999999998, Epsilon: 0.1000
Saved video for epoch 19400 to ./videos/flappy_bird_epoch_19400.mp4
Epoch 19420, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19420 to ./videos/flappy_bird_epoch_19420.mp4
Epoch 19440, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 19440 to ./videos/flappy_bird_epoch_19440.mp4
Epoch 19460, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19460 to ./videos/flappy_bird_epoch_19460.mp4
Epoch 19480, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 19480 to ./videos/flappy_bird_epoch_19480.mp4
Epoch 19500, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19500 to ./videos/flappy_bird_epoch_19500.mp4
Epoch 19520, Total Reward: 0.10000000000000142, Epsilon: 0.1000
Saved video for epoch 19520 to ./videos/flappy_bird_epoch_19520.mp4
Epoch 19540, Total Reward: -5.699999999999998, Epsilon: 0.1000
Saved video for epoch 19540 to ./videos/flappy_bird_epoch_19540.mp4
Epoch 19560, Total Reward: -6.899999999999999, Epsilon: 0.1000
Saved video for epoch 19560 to ./videos/flappy_bird_epoch_19560.mp4
Epoch 19580, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 19580 to ./videos/flappy_bird_epoch_19580.mp4
Epoch 19600, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19600 to ./videos/flappy_bird_epoch_19600.mp4
Epoch 19620, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 19620 to ./videos/flappy_bird_epoch_19620.mp4
Epoch 19640, Total Reward: -0.29999999999999893, Epsilon: 0.1000
Saved video for epoch 19640 to ./videos/flappy_bird_epoch_19640.mp4
Epoch 19660, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19660 to ./videos/flappy_bird_epoch_19660.mp4
Epoch 19680, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19680 to ./videos/flappy_bird_epoch_19680.mp4
Epoch 19700, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 19700 to ./videos/flappy_bird_epoch_19700.mp4
The bird got through a pipe
Epoch 19720, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19720 to ./videos/flappy_bird_epoch_19720.mp4
Epoch 19740, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19740 to ./videos/flappy_bird_epoch_19740.mp4
Epoch 19760, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19760 to ./videos/flappy_bird_epoch_19760.mp4
Epoch 19780, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 19780 to ./videos/flappy_bird_epoch_19780.mp4
Epoch 19800, Total Reward: -2.699999999999998, Epsilon: 0.1000
Saved video for epoch 19800 to ./videos/flappy_bird_epoch_19800.mp4
Epoch 19820, Total Reward: -9.299999999999999, Epsilon: 0.1000
Saved video for epoch 19820 to ./videos/flappy_bird_epoch_19820.mp4
The bird got through a pipe
Epoch 19840, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19840 to ./videos/flappy_bird_epoch_19840.mp4
Epoch 19860, Total Reward: -2.3999999999999986, Epsilon: 0.1000
Saved video for epoch 19860 to ./videos/flappy_bird_epoch_19860.mp4
Epoch 19880, Total Reward: 1.5000000000000018, Epsilon: 0.1000
Saved video for epoch 19880 to ./videos/flappy_bird_epoch_19880.mp4
Epoch 19900, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19900 to ./videos/flappy_bird_epoch_19900.mp4
Epoch 19920, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19920 to ./videos/flappy_bird_epoch_19920.mp4
Epoch 19940, Total Reward: -8.099999999999998, Epsilon: 0.1000
Saved video for epoch 19940 to ./videos/flappy_bird_epoch_19940.mp4
Epoch 19960, Total Reward: 1.200000000000002, Epsilon: 0.1000
Saved video for epoch 19960 to ./videos/flappy_bird_epoch_19960.mp4
Epoch 19980, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 19980 to ./videos/flappy_bird_epoch_19980.mp4
Epoch 20000, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 20000 to ./videos/flappy_bird_epoch_20000.mp4
Epoch 20020, Total Reward: 0.9000000000000004, Epsilon: 0.1000
Saved video for epoch 20020 to ./videos/flappy_bird_epoch_20020.mp4
Epoch 20040, Total Reward: 1.1000000000000019, Epsilon: 0.1000
Saved video for epoch 20040 to ./videos/flappy_bird_epoch_20040.mp4
Epoch 20060, Total Reward: -0.8999999999999986, Epsilon: 0.1000
Saved video for epoch 20060 to ./videos/flappy_bird_epoch_20060.mp4
```
