# Experiment 2

## Observations:
- optimiser used: optimizer = optim.RMSprop(model.parameters(), lr=0.00025)
- batch_size = 100
- not taking in consideration last 2 states, we were only working on the current state of the game
- the reward when the bird succesfully passed through a pipe would be 10
- too big convolutions

## Hyperparameters:
- learning rate = 0.00025
- batch_size = 100
- replay buffer capacity = 50000
- target network update every 10 eps.
- epsilon decay = 0.99999

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
def visualize_features_live(tensor):
    # Get first batch item's feature maps
    cv2.namedWindow('Feature Maps', cv2.WINDOW_NORMAL)

    features = tensor[0].detach().cpu().numpy()
    
    # Normalize to 0-255 range
    features = ((features - features.min()) / (features.max() - features.min() + 1e-8) * 255).astype(np.uint8)
    
    
    # Show live window
    cv2.imshow('Feature Maps', features)
    cv2.waitKey(1)

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
        
        self.fc_layers = nn.Sequential(
            nn.Linear(1728, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 2) # 2 actions: flap or not flap
        )
    
    def forward(self, x):
        # x should be [batch_size, 1, height, width]
        x = self.conv_layers(x)               # => [batch_size, 64, H', W']
        # visualize_features_live(x)
        x = x.view(x.size(0), -1)             # => [batch_size, 64*H'*W']
        return self.fc_layers(x)              # => [batch_size, 2]


##########################################################################
# 3) Frame Preprocessing (Sobel + Thresholding + Pooling)
##########################################################################
def preprocess_frame(frame):
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

    # Convert to PyTorch tensor as [1, 1, H, W] (batch=1, channels=1)
    tensor_image = torch.FloatTensor(display_image).unsqueeze(0).unsqueeze(0)  # => [1, 1, height, width]

    # # Create visualization windows
    # cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
    
    # # Show processed frame
    # cv2.imshow('Processed Frame', display_image)
    
    # cv2.waitKey(1)

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

    def sample(self, batch_size, sequence_length=4):

        normal_reward = [t for t in self.buffer]

        # Function to sample sequences
        def sample_sequence(buffer, count):
            sequences = []
            for _ in range(count):
                idx = random.randint(0, len(buffer) - sequence_length)
                sequences.append(buffer[idx : idx + sequence_length])
            return sequences

        normal_samples = sample_sequence(normal_reward, batch_size // sequence_length)

        # Flatten sequences into a single batch
        batch = [item for seq in normal_samples for item in seq]
        random.shuffle(batch)

        # Unpack batch as before
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states, dim=0),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.cat(next_states, dim=0),
            torch.tensor(dones, dtype=torch.bool),
        )

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
    batch_size = 100
    device = torch.device('mps')
    print(device)

    model = FlappyBirdCNN().to(device)
    target_model = FlappyBirdCNN().to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.00025)
    
    # Hyperparameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99999
    gamma = 0.9
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=50000)

    # If a model path is provided, load it
    if path_to_model is not None and os.path.exists(path_to_model):
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
                print("The bird got through a pipe at epoch " + str(epoch))
                reward = reward * 10

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

        # Save a video if this epoch was recorded
        if should_record and frames:
            # Create videos directory if it doesn't exist
            os.makedirs('./videos_v3', exist_ok=True)
            
            video_path = f'./videos_v3/flappy_bird_epoch_{epoch}.mp4'
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
    random.seed(time.time())
    # Make sure you have the correct version of flappy_bird_gymnasium
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    # Provide path_to_model to resume training from a saved checkpoint
    path_to_model = './models/flappy_bird_model_epoch_38000.pth'
    train_model(env, start_epoch=0, path_to_model=path_to_model)
    env.close()

if __name__ == "__main__":
    main()

```

## Results:
```txt
Saved video for epoch 0 to ./videos_v3/flappy_bird_epoch_0.mp4
Epoch 20, Total Reward: -7.499999999999998, Epsilon: 0.9998
Saved video for epoch 20 to ./videos_v3/flappy_bird_epoch_20.mp4
Epoch 40, Total Reward: -5.099999999999998, Epsilon: 0.9996
Saved video for epoch 40 to ./videos_v3/flappy_bird_epoch_40.mp4
Epoch 60, Total Reward: -7.499999999999998, Epsilon: 0.9994
Saved video for epoch 60 to ./videos_v3/flappy_bird_epoch_60.mp4
Epoch 80, Total Reward: -8.7, Epsilon: 0.9992
Saved video for epoch 80 to ./videos_v3/flappy_bird_epoch_80.mp4
Epoch 100, Total Reward: -6.899999999999999, Epsilon: 0.9990
Saved video for epoch 100 to ./videos_v3/flappy_bird_epoch_100.mp4
Epoch 120, Total Reward: -7.499999999999998, Epsilon: 0.9988
Saved video for epoch 120 to ./videos_v3/flappy_bird_epoch_120.mp4
Epoch 140, Total Reward: -8.099999999999998, Epsilon: 0.9986
Saved video for epoch 140 to ./videos_v3/flappy_bird_epoch_140.mp4
Epoch 160, Total Reward: -8.099999999999998, Epsilon: 0.9984
Saved video for epoch 160 to ./videos_v3/flappy_bird_epoch_160.mp4
Epoch 180, Total Reward: -8.099999999999998, Epsilon: 0.9982
Saved video for epoch 180 to ./videos_v3/flappy_bird_epoch_180.mp4
Epoch 200, Total Reward: -8.099999999999998, Epsilon: 0.9980
Saved video for epoch 200 to ./videos_v3/flappy_bird_epoch_200.mp4
Epoch 220, Total Reward: -3.299999999999998, Epsilon: 0.9978
Saved video for epoch 220 to ./videos_v3/flappy_bird_epoch_220.mp4
Epoch 240, Total Reward: -8.7, Epsilon: 0.9976
Saved video for epoch 240 to ./videos_v3/flappy_bird_epoch_240.mp4
Epoch 260, Total Reward: -6.299999999999999, Epsilon: 0.9974
Saved video for epoch 260 to ./videos_v3/flappy_bird_epoch_260.mp4
Epoch 280, Total Reward: -7.499999999999998, Epsilon: 0.9972
Saved video for epoch 280 to ./videos_v3/flappy_bird_epoch_280.mp4
Epoch 300, Total Reward: -7.499999999999998, Epsilon: 0.9970
Saved video for epoch 300 to ./videos_v3/flappy_bird_epoch_300.mp4
Epoch 320, Total Reward: -8.099999999999998, Epsilon: 0.9968
Saved video for epoch 320 to ./videos_v3/flappy_bird_epoch_320.mp4
Epoch 340, Total Reward: -8.099999999999998, Epsilon: 0.9966
Saved video for epoch 340 to ./videos_v3/flappy_bird_epoch_340.mp4
Epoch 360, Total Reward: -6.899999999999999, Epsilon: 0.9964
Saved video for epoch 360 to ./videos_v3/flappy_bird_epoch_360.mp4
Epoch 380, Total Reward: -5.699999999999998, Epsilon: 0.9962
Saved video for epoch 380 to ./videos_v3/flappy_bird_epoch_380.mp4
Epoch 400, Total Reward: -6.299999999999999, Epsilon: 0.9960
Saved video for epoch 400 to ./videos_v3/flappy_bird_epoch_400.mp4
Epoch 420, Total Reward: -6.899999999999999, Epsilon: 0.9958
Saved video for epoch 420 to ./videos_v3/flappy_bird_epoch_420.mp4
Epoch 440, Total Reward: -7.499999999999998, Epsilon: 0.9956
Saved video for epoch 440 to ./videos_v3/flappy_bird_epoch_440.mp4
Epoch 460, Total Reward: -7.499999999999998, Epsilon: 0.9954
Saved video for epoch 460 to ./videos_v3/flappy_bird_epoch_460.mp4
Epoch 480, Total Reward: -8.099999999999998, Epsilon: 0.9952
Saved video for epoch 480 to ./videos_v3/flappy_bird_epoch_480.mp4
Epoch 500, Total Reward: -6.299999999999999, Epsilon: 0.9950
Saved video for epoch 500 to ./videos_v3/flappy_bird_epoch_500.mp4
Epoch 520, Total Reward: -8.099999999999998, Epsilon: 0.9948
Saved video for epoch 520 to ./videos_v3/flappy_bird_epoch_520.mp4
Epoch 540, Total Reward: -8.7, Epsilon: 0.9946
Saved video for epoch 540 to ./videos_v3/flappy_bird_epoch_540.mp4
Epoch 560, Total Reward: -8.099999999999998, Epsilon: 0.9944
Saved video for epoch 560 to ./videos_v3/flappy_bird_epoch_560.mp4
Epoch 580, Total Reward: -6.899999999999999, Epsilon: 0.9942
Saved video for epoch 580 to ./videos_v3/flappy_bird_epoch_580.mp4
Epoch 600, Total Reward: -7.499999999999998, Epsilon: 0.9940
Saved video for epoch 600 to ./videos_v3/flappy_bird_epoch_600.mp4
Epoch 620, Total Reward: -8.099999999999998, Epsilon: 0.9938
Saved video for epoch 620 to ./videos_v3/flappy_bird_epoch_620.mp4
Epoch 640, Total Reward: -6.899999999999999, Epsilon: 0.9936
Saved video for epoch 640 to ./videos_v3/flappy_bird_epoch_640.mp4
Epoch 660, Total Reward: -7.499999999999998, Epsilon: 0.9934
Saved video for epoch 660 to ./videos_v3/flappy_bird_epoch_660.mp4
Epoch 680, Total Reward: -6.899999999999999, Epsilon: 0.9932
Saved video for epoch 680 to ./videos_v3/flappy_bird_epoch_680.mp4
Epoch 700, Total Reward: -8.7, Epsilon: 0.9930
Saved video for epoch 700 to ./videos_v3/flappy_bird_epoch_700.mp4
Epoch 720, Total Reward: -8.099999999999998, Epsilon: 0.9928
Saved video for epoch 720 to ./videos_v3/flappy_bird_epoch_720.mp4
Epoch 740, Total Reward: -8.099999999999998, Epsilon: 0.9926
Saved video for epoch 740 to ./videos_v3/flappy_bird_epoch_740.mp4
Epoch 760, Total Reward: -7.499999999999998, Epsilon: 0.9924
Saved video for epoch 760 to ./videos_v3/flappy_bird_epoch_760.mp4
Epoch 780, Total Reward: -7.499999999999998, Epsilon: 0.9922
Saved video for epoch 780 to ./videos_v3/flappy_bird_epoch_780.mp4
Epoch 800, Total Reward: -6.299999999999999, Epsilon: 0.9920
Saved video for epoch 800 to ./videos_v3/flappy_bird_epoch_800.mp4
Epoch 820, Total Reward: -6.299999999999999, Epsilon: 0.9918
Saved video for epoch 820 to ./videos_v3/flappy_bird_epoch_820.mp4
Epoch 840, Total Reward: -5.699999999999998, Epsilon: 0.9916
Saved video for epoch 840 to ./videos_v3/flappy_bird_epoch_840.mp4
Epoch 860, Total Reward: -7.499999999999998, Epsilon: 0.9914
Saved video for epoch 860 to ./videos_v3/flappy_bird_epoch_860.mp4
Epoch 880, Total Reward: -7.499999999999998, Epsilon: 0.9912
Saved video for epoch 880 to ./videos_v3/flappy_bird_epoch_880.mp4
Epoch 900, Total Reward: -7.499999999999998, Epsilon: 0.9910
Saved video for epoch 900 to ./videos_v3/flappy_bird_epoch_900.mp4
Epoch 920, Total Reward: -8.099999999999998, Epsilon: 0.9908
Saved video for epoch 920 to ./videos_v3/flappy_bird_epoch_920.mp4
Epoch 940, Total Reward: -7.499999999999998, Epsilon: 0.9906
Saved video for epoch 940 to ./videos_v3/flappy_bird_epoch_940.mp4
Epoch 960, Total Reward: -8.7, Epsilon: 0.9904
Saved video for epoch 960 to ./videos_v3/flappy_bird_epoch_960.mp4
Epoch 980, Total Reward: -8.099999999999998, Epsilon: 0.9902
Saved video for epoch 980 to ./videos_v3/flappy_bird_epoch_980.mp4
Epoch 1000, Total Reward: -8.099999999999998, Epsilon: 0.9900
Saved video for epoch 1000 to ./videos_v3/flappy_bird_epoch_1000.mp4
Epoch 1020, Total Reward: -8.099999999999998, Epsilon: 0.9899
Saved video for epoch 1020 to ./videos_v3/flappy_bird_epoch_1020.mp4
Epoch 1040, Total Reward: -8.7, Epsilon: 0.9897
Saved video for epoch 1040 to ./videos_v3/flappy_bird_epoch_1040.mp4
Epoch 1060, Total Reward: -7.499999999999998, Epsilon: 0.9895
Saved video for epoch 1060 to ./videos_v3/flappy_bird_epoch_1060.mp4
Epoch 1080, Total Reward: -7.499999999999998, Epsilon: 0.9893
Saved video for epoch 1080 to ./videos_v3/flappy_bird_epoch_1080.mp4
Epoch 1100, Total Reward: -8.099999999999998, Epsilon: 0.9891
Saved video for epoch 1100 to ./videos_v3/flappy_bird_epoch_1100.mp4
Epoch 1120, Total Reward: -7.499999999999998, Epsilon: 0.9889
Saved video for epoch 1120 to ./videos_v3/flappy_bird_epoch_1120.mp4
Epoch 1140, Total Reward: -7.499999999999998, Epsilon: 0.9887
Saved video for epoch 1140 to ./videos_v3/flappy_bird_epoch_1140.mp4
Epoch 1160, Total Reward: -8.099999999999998, Epsilon: 0.9885
Saved video for epoch 1160 to ./videos_v3/flappy_bird_epoch_1160.mp4
Epoch 1180, Total Reward: -8.099999999999998, Epsilon: 0.9883
Saved video for epoch 1180 to ./videos_v3/flappy_bird_epoch_1180.mp4
Epoch 1200, Total Reward: -7.499999999999998, Epsilon: 0.9881
Saved video for epoch 1200 to ./videos_v3/flappy_bird_epoch_1200.mp4
Epoch 1220, Total Reward: -5.699999999999998, Epsilon: 0.9879
Saved video for epoch 1220 to ./videos_v3/flappy_bird_epoch_1220.mp4
Epoch 1240, Total Reward: -5.699999999999998, Epsilon: 0.9877
Saved video for epoch 1240 to ./videos_v3/flappy_bird_epoch_1240.mp4
Epoch 1260, Total Reward: -8.099999999999998, Epsilon: 0.9875
Saved video for epoch 1260 to ./videos_v3/flappy_bird_epoch_1260.mp4
Epoch 1280, Total Reward: -7.499999999999998, Epsilon: 0.9873
Saved video for epoch 1280 to ./videos_v3/flappy_bird_epoch_1280.mp4
Epoch 1300, Total Reward: -7.499999999999998, Epsilon: 0.9871
Saved video for epoch 1300 to ./videos_v3/flappy_bird_epoch_1300.mp4
Epoch 1320, Total Reward: -8.099999999999998, Epsilon: 0.9869
Saved video for epoch 1320 to ./videos_v3/flappy_bird_epoch_1320.mp4
Epoch 1340, Total Reward: -8.7, Epsilon: 0.9867
Saved video for epoch 1340 to ./videos_v3/flappy_bird_epoch_1340.mp4
Epoch 1360, Total Reward: -8.099999999999998, Epsilon: 0.9865
Saved video for epoch 1360 to ./videos_v3/flappy_bird_epoch_1360.mp4
Epoch 1380, Total Reward: -8.099999999999998, Epsilon: 0.9863
Saved video for epoch 1380 to ./videos_v3/flappy_bird_epoch_1380.mp4
Epoch 1400, Total Reward: -6.899999999999999, Epsilon: 0.9861
Saved video for epoch 1400 to ./videos_v3/flappy_bird_epoch_1400.mp4
Epoch 1420, Total Reward: -8.099999999999998, Epsilon: 0.9859
Saved video for epoch 1420 to ./videos_v3/flappy_bird_epoch_1420.mp4
Epoch 1440, Total Reward: -8.099999999999998, Epsilon: 0.9857
Saved video for epoch 1440 to ./videos_v3/flappy_bird_epoch_1440.mp4
Epoch 1460, Total Reward: -8.099999999999998, Epsilon: 0.9855
Saved video for epoch 1460 to ./videos_v3/flappy_bird_epoch_1460.mp4
Epoch 1480, Total Reward: -8.099999999999998, Epsilon: 0.9853
Saved video for epoch 1480 to ./videos_v3/flappy_bird_epoch_1480.mp4
Epoch 1500, Total Reward: -6.299999999999999, Epsilon: 0.9851
Saved video for epoch 1500 to ./videos_v3/flappy_bird_epoch_1500.mp4
Epoch 1520, Total Reward: -7.499999999999998, Epsilon: 0.9849
Saved video for epoch 1520 to ./videos_v3/flappy_bird_epoch_1520.mp4
Epoch 1540, Total Reward: -5.699999999999998, Epsilon: 0.9847
Saved video for epoch 1540 to ./videos_v3/flappy_bird_epoch_1540.mp4
Epoch 1560, Total Reward: -7.499999999999998, Epsilon: 0.9845
Saved video for epoch 1560 to ./videos_v3/flappy_bird_epoch_1560.mp4
Epoch 1580, Total Reward: -8.099999999999998, Epsilon: 0.9843
Saved video for epoch 1580 to ./videos_v3/flappy_bird_epoch_1580.mp4
Epoch 1600, Total Reward: -7.499999999999998, Epsilon: 0.9841
Saved video for epoch 1600 to ./videos_v3/flappy_bird_epoch_1600.mp4
Epoch 1620, Total Reward: -8.7, Epsilon: 0.9839
Saved video for epoch 1620 to ./videos_v3/flappy_bird_epoch_1620.mp4
Epoch 1640, Total Reward: -5.699999999999998, Epsilon: 0.9837
Saved video for epoch 1640 to ./videos_v3/flappy_bird_epoch_1640.mp4
Epoch 1660, Total Reward: -8.7, Epsilon: 0.9835
Saved video for epoch 1660 to ./videos_v3/flappy_bird_epoch_1660.mp4
Epoch 1680, Total Reward: -8.099999999999998, Epsilon: 0.9833
Saved video for epoch 1680 to ./videos_v3/flappy_bird_epoch_1680.mp4
Epoch 1700, Total Reward: -6.899999999999999, Epsilon: 0.9831
Saved video for epoch 1700 to ./videos_v3/flappy_bird_epoch_1700.mp4
Epoch 1720, Total Reward: -8.099999999999998, Epsilon: 0.9829
Saved video for epoch 1720 to ./videos_v3/flappy_bird_epoch_1720.mp4
Epoch 1740, Total Reward: -8.099999999999998, Epsilon: 0.9828
Saved video for epoch 1740 to ./videos_v3/flappy_bird_epoch_1740.mp4
Epoch 1760, Total Reward: -8.099999999999998, Epsilon: 0.9826
Saved video for epoch 1760 to ./videos_v3/flappy_bird_epoch_1760.mp4
Epoch 1780, Total Reward: -6.299999999999999, Epsilon: 0.9824
Saved video for epoch 1780 to ./videos_v3/flappy_bird_epoch_1780.mp4
Epoch 1800, Total Reward: -7.499999999999998, Epsilon: 0.9822
Saved video for epoch 1800 to ./videos_v3/flappy_bird_epoch_1800.mp4
Epoch 1820, Total Reward: -7.499999999999998, Epsilon: 0.9820
Saved video for epoch 1820 to ./videos_v3/flappy_bird_epoch_1820.mp4
Epoch 1840, Total Reward: -6.299999999999999, Epsilon: 0.9818
Saved video for epoch 1840 to ./videos_v3/flappy_bird_epoch_1840.mp4
Epoch 1860, Total Reward: -8.099999999999998, Epsilon: 0.9816
Saved video for epoch 1860 to ./videos_v3/flappy_bird_epoch_1860.mp4
Epoch 1880, Total Reward: -7.499999999999998, Epsilon: 0.9814
Saved video for epoch 1880 to ./videos_v3/flappy_bird_epoch_1880.mp4
Epoch 1900, Total Reward: -7.499999999999998, Epsilon: 0.9812
Saved video for epoch 1900 to ./videos_v3/flappy_bird_epoch_1900.mp4
Epoch 1920, Total Reward: -8.7, Epsilon: 0.9810
Saved video for epoch 1920 to ./videos_v3/flappy_bird_epoch_1920.mp4
Epoch 1940, Total Reward: -8.099999999999998, Epsilon: 0.9808
Saved video for epoch 1940 to ./videos_v3/flappy_bird_epoch_1940.mp4
Epoch 1960, Total Reward: -7.499999999999998, Epsilon: 0.9806
Saved video for epoch 1960 to ./videos_v3/flappy_bird_epoch_1960.mp4
Epoch 1980, Total Reward: -7.499999999999998, Epsilon: 0.9804
Saved video for epoch 1980 to ./videos_v3/flappy_bird_epoch_1980.mp4
Epoch 2000, Total Reward: -6.899999999999999, Epsilon: 0.9802
Saved video for epoch 2000 to ./videos_v3/flappy_bird_epoch_2000.mp4
Epoch 2020, Total Reward: -6.299999999999999, Epsilon: 0.9800
Saved video for epoch 2020 to ./videos_v3/flappy_bird_epoch_2020.mp4
Epoch 2040, Total Reward: -8.099999999999998, Epsilon: 0.9798
Saved video for epoch 2040 to ./videos_v3/flappy_bird_epoch_2040.mp4
Epoch 2060, Total Reward: -8.7, Epsilon: 0.9796
Saved video for epoch 2060 to ./videos_v3/flappy_bird_epoch_2060.mp4
Epoch 2080, Total Reward: -6.299999999999999, Epsilon: 0.9794
Saved video for epoch 2080 to ./videos_v3/flappy_bird_epoch_2080.mp4
Epoch 2100, Total Reward: -6.299999999999999, Epsilon: 0.9792
Saved video for epoch 2100 to ./videos_v3/flappy_bird_epoch_2100.mp4
Epoch 2120, Total Reward: -8.099999999999998, Epsilon: 0.9790
Saved video for epoch 2120 to ./videos_v3/flappy_bird_epoch_2120.mp4
Epoch 2140, Total Reward: -5.099999999999998, Epsilon: 0.9788
Saved video for epoch 2140 to ./videos_v3/flappy_bird_epoch_2140.mp4
Epoch 2160, Total Reward: -8.099999999999998, Epsilon: 0.9786
Saved video for epoch 2160 to ./videos_v3/flappy_bird_epoch_2160.mp4
Epoch 2180, Total Reward: -7.499999999999998, Epsilon: 0.9784
Saved video for epoch 2180 to ./videos_v3/flappy_bird_epoch_2180.mp4
Epoch 2200, Total Reward: -8.7, Epsilon: 0.9782
Saved video for epoch 2200 to ./videos_v3/flappy_bird_epoch_2200.mp4
Epoch 2220, Total Reward: -8.7, Epsilon: 0.9780
Saved video for epoch 2220 to ./videos_v3/flappy_bird_epoch_2220.mp4
Epoch 2240, Total Reward: -7.499999999999998, Epsilon: 0.9778
Saved video for epoch 2240 to ./videos_v3/flappy_bird_epoch_2240.mp4
Epoch 2260, Total Reward: -8.099999999999998, Epsilon: 0.9777
Saved video for epoch 2260 to ./videos_v3/flappy_bird_epoch_2260.mp4
Epoch 2280, Total Reward: -8.099999999999998, Epsilon: 0.9775
Saved video for epoch 2280 to ./videos_v3/flappy_bird_epoch_2280.mp4
Epoch 2300, Total Reward: -8.7, Epsilon: 0.9773
Saved video for epoch 2300 to ./videos_v3/flappy_bird_epoch_2300.mp4
Epoch 2320, Total Reward: -6.899999999999999, Epsilon: 0.9771
Saved video for epoch 2320 to ./videos_v3/flappy_bird_epoch_2320.mp4
Epoch 2340, Total Reward: -8.099999999999998, Epsilon: 0.9769
Saved video for epoch 2340 to ./videos_v3/flappy_bird_epoch_2340.mp4
Epoch 2360, Total Reward: -5.699999999999998, Epsilon: 0.9767
Saved video for epoch 2360 to ./videos_v3/flappy_bird_epoch_2360.mp4
Epoch 2380, Total Reward: -8.099999999999998, Epsilon: 0.9765
Saved video for epoch 2380 to ./videos_v3/flappy_bird_epoch_2380.mp4
Epoch 2400, Total Reward: -7.499999999999998, Epsilon: 0.9763
Saved video for epoch 2400 to ./videos_v3/flappy_bird_epoch_2400.mp4
Epoch 2420, Total Reward: -6.299999999999999, Epsilon: 0.9761
Saved video for epoch 2420 to ./videos_v3/flappy_bird_epoch_2420.mp4
Epoch 2440, Total Reward: -6.299999999999999, Epsilon: 0.9759
Saved video for epoch 2440 to ./videos_v3/flappy_bird_epoch_2440.mp4
Epoch 2460, Total Reward: -8.099999999999998, Epsilon: 0.9757
Saved video for epoch 2460 to ./videos_v3/flappy_bird_epoch_2460.mp4
Epoch 2480, Total Reward: -8.099999999999998, Epsilon: 0.9755
Saved video for epoch 2480 to ./videos_v3/flappy_bird_epoch_2480.mp4
Epoch 2500, Total Reward: -6.899999999999999, Epsilon: 0.9753
Saved video for epoch 2500 to ./videos_v3/flappy_bird_epoch_2500.mp4
Epoch 2520, Total Reward: -8.7, Epsilon: 0.9751
Saved video for epoch 2520 to ./videos_v3/flappy_bird_epoch_2520.mp4
Epoch 2540, Total Reward: -2.099999999999998, Epsilon: 0.9749
Saved video for epoch 2540 to ./videos_v3/flappy_bird_epoch_2540.mp4
Epoch 2560, Total Reward: -8.7, Epsilon: 0.9747
Saved video for epoch 2560 to ./videos_v3/flappy_bird_epoch_2560.mp4
Epoch 2580, Total Reward: -7.499999999999998, Epsilon: 0.9745
Saved video for epoch 2580 to ./videos_v3/flappy_bird_epoch_2580.mp4
Epoch 2600, Total Reward: -6.899999999999999, Epsilon: 0.9743
Saved video for epoch 2600 to ./videos_v3/flappy_bird_epoch_2600.mp4
Epoch 2620, Total Reward: -6.899999999999999, Epsilon: 0.9741
Saved video for epoch 2620 to ./videos_v3/flappy_bird_epoch_2620.mp4
Epoch 2640, Total Reward: -8.099999999999998, Epsilon: 0.9739
Saved video for epoch 2640 to ./videos_v3/flappy_bird_epoch_2640.mp4
Epoch 2660, Total Reward: -7.499999999999998, Epsilon: 0.9738
Saved video for epoch 2660 to ./videos_v3/flappy_bird_epoch_2660.mp4
Epoch 2680, Total Reward: -5.699999999999998, Epsilon: 0.9736
Saved video for epoch 2680 to ./videos_v3/flappy_bird_epoch_2680.mp4
Epoch 2700, Total Reward: -8.099999999999998, Epsilon: 0.9734
Saved video for epoch 2700 to ./videos_v3/flappy_bird_epoch_2700.mp4
Epoch 2720, Total Reward: -6.299999999999999, Epsilon: 0.9732
Saved video for epoch 2720 to ./videos_v3/flappy_bird_epoch_2720.mp4
Epoch 2740, Total Reward: -6.899999999999999, Epsilon: 0.9730
Saved video for epoch 2740 to ./videos_v3/flappy_bird_epoch_2740.mp4
Epoch 2760, Total Reward: -8.099999999999998, Epsilon: 0.9728
Saved video for epoch 2760 to ./videos_v3/flappy_bird_epoch_2760.mp4
Epoch 2780, Total Reward: -6.299999999999999, Epsilon: 0.9726
Saved video for epoch 2780 to ./videos_v3/flappy_bird_epoch_2780.mp4
Epoch 2800, Total Reward: -7.499999999999998, Epsilon: 0.9724
Saved video for epoch 2800 to ./videos_v3/flappy_bird_epoch_2800.mp4
Epoch 2820, Total Reward: -7.499999999999998, Epsilon: 0.9722
Saved video for epoch 2820 to ./videos_v3/flappy_bird_epoch_2820.mp4
Epoch 2840, Total Reward: -8.099999999999998, Epsilon: 0.9720
Saved video for epoch 2840 to ./videos_v3/flappy_bird_epoch_2840.mp4
Epoch 2860, Total Reward: -7.499999999999998, Epsilon: 0.9718
Saved video for epoch 2860 to ./videos_v3/flappy_bird_epoch_2860.mp4
Epoch 2880, Total Reward: -7.499999999999998, Epsilon: 0.9716
Saved video for epoch 2880 to ./videos_v3/flappy_bird_epoch_2880.mp4
Epoch 2900, Total Reward: -7.499999999999998, Epsilon: 0.9714
Saved video for epoch 2900 to ./videos_v3/flappy_bird_epoch_2900.mp4
Epoch 2920, Total Reward: -8.7, Epsilon: 0.9712
Saved video for epoch 2920 to ./videos_v3/flappy_bird_epoch_2920.mp4
Epoch 2940, Total Reward: -6.899999999999999, Epsilon: 0.9710
Saved video for epoch 2940 to ./videos_v3/flappy_bird_epoch_2940.mp4
Epoch 2960, Total Reward: -8.099999999999998, Epsilon: 0.9708
Saved video for epoch 2960 to ./videos_v3/flappy_bird_epoch_2960.mp4
Epoch 2980, Total Reward: -7.499999999999998, Epsilon: 0.9706
Saved video for epoch 2980 to ./videos_v3/flappy_bird_epoch_2980.mp4
Epoch 3000, Total Reward: -8.7, Epsilon: 0.9704
Saved video for epoch 3000 to ./videos_v3/flappy_bird_epoch_3000.mp4
Epoch 3020, Total Reward: -8.7, Epsilon: 0.9703
Saved video for epoch 3020 to ./videos_v3/flappy_bird_epoch_3020.mp4
Epoch 3040, Total Reward: -6.299999999999999, Epsilon: 0.9701
Saved video for epoch 3040 to ./videos_v3/flappy_bird_epoch_3040.mp4
Epoch 3060, Total Reward: -8.7, Epsilon: 0.9699
Saved video for epoch 3060 to ./videos_v3/flappy_bird_epoch_3060.mp4
Epoch 3080, Total Reward: -6.299999999999999, Epsilon: 0.9697
Saved video for epoch 3080 to ./videos_v3/flappy_bird_epoch_3080.mp4
Epoch 3100, Total Reward: -8.099999999999998, Epsilon: 0.9695
Saved video for epoch 3100 to ./videos_v3/flappy_bird_epoch_3100.mp4
Epoch 3120, Total Reward: -6.899999999999999, Epsilon: 0.9693
Saved video for epoch 3120 to ./videos_v3/flappy_bird_epoch_3120.mp4
Epoch 3140, Total Reward: -8.099999999999998, Epsilon: 0.9691
Saved video for epoch 3140 to ./videos_v3/flappy_bird_epoch_3140.mp4
Epoch 3160, Total Reward: -8.099999999999998, Epsilon: 0.9689
Saved video for epoch 3160 to ./videos_v3/flappy_bird_epoch_3160.mp4
Epoch 3180, Total Reward: -7.499999999999998, Epsilon: 0.9687
Saved video for epoch 3180 to ./videos_v3/flappy_bird_epoch_3180.mp4
Epoch 3200, Total Reward: -8.099999999999998, Epsilon: 0.9685
Saved video for epoch 3200 to ./videos_v3/flappy_bird_epoch_3200.mp4
Epoch 3220, Total Reward: -8.7, Epsilon: 0.9683
Saved video for epoch 3220 to ./videos_v3/flappy_bird_epoch_3220.mp4
Epoch 3240, Total Reward: -8.099999999999998, Epsilon: 0.9681
Saved video for epoch 3240 to ./videos_v3/flappy_bird_epoch_3240.mp4
Epoch 3260, Total Reward: -7.499999999999998, Epsilon: 0.9679
Saved video for epoch 3260 to ./videos_v3/flappy_bird_epoch_3260.mp4
Epoch 3280, Total Reward: -6.899999999999999, Epsilon: 0.9677
Saved video for epoch 3280 to ./videos_v3/flappy_bird_epoch_3280.mp4
Epoch 3300, Total Reward: -8.099999999999998, Epsilon: 0.9675
Saved video for epoch 3300 to ./videos_v3/flappy_bird_epoch_3300.mp4
Epoch 3320, Total Reward: -7.499999999999998, Epsilon: 0.9673
Saved video for epoch 3320 to ./videos_v3/flappy_bird_epoch_3320.mp4
Epoch 3340, Total Reward: -7.499999999999998, Epsilon: 0.9672
Saved video for epoch 3340 to ./videos_v3/flappy_bird_epoch_3340.mp4
Epoch 3360, Total Reward: -8.099999999999998, Epsilon: 0.9670
Saved video for epoch 3360 to ./videos_v3/flappy_bird_epoch_3360.mp4
Epoch 3380, Total Reward: -6.899999999999999, Epsilon: 0.9668
Saved video for epoch 3380 to ./videos_v3/flappy_bird_epoch_3380.mp4
Epoch 3400, Total Reward: -8.099999999999998, Epsilon: 0.9666
Saved video for epoch 3400 to ./videos_v3/flappy_bird_epoch_3400.mp4
Epoch 3420, Total Reward: -5.099999999999998, Epsilon: 0.9664
Saved video for epoch 3420 to ./videos_v3/flappy_bird_epoch_3420.mp4
Epoch 3440, Total Reward: -8.099999999999998, Epsilon: 0.9662
Saved video for epoch 3440 to ./videos_v3/flappy_bird_epoch_3440.mp4
Epoch 3460, Total Reward: -6.299999999999999, Epsilon: 0.9660
Saved video for epoch 3460 to ./videos_v3/flappy_bird_epoch_3460.mp4
Epoch 3480, Total Reward: -7.499999999999998, Epsilon: 0.9658
Saved video for epoch 3480 to ./videos_v3/flappy_bird_epoch_3480.mp4
Epoch 3500, Total Reward: -6.299999999999999, Epsilon: 0.9656
Saved video for epoch 3500 to ./videos_v3/flappy_bird_epoch_3500.mp4
Epoch 3520, Total Reward: -7.499999999999998, Epsilon: 0.9654
Saved video for epoch 3520 to ./videos_v3/flappy_bird_epoch_3520.mp4
Epoch 3540, Total Reward: -8.7, Epsilon: 0.9652
Saved video for epoch 3540 to ./videos_v3/flappy_bird_epoch_3540.mp4
Epoch 3560, Total Reward: -7.499999999999998, Epsilon: 0.9650
Saved video for epoch 3560 to ./videos_v3/flappy_bird_epoch_3560.mp4
Epoch 3580, Total Reward: -8.099999999999998, Epsilon: 0.9648
Saved video for epoch 3580 to ./videos_v3/flappy_bird_epoch_3580.mp4
Epoch 3600, Total Reward: -8.7, Epsilon: 0.9646
Saved video for epoch 3600 to ./videos_v3/flappy_bird_epoch_3600.mp4
Epoch 3620, Total Reward: -8.099999999999998, Epsilon: 0.9644
Saved video for epoch 3620 to ./videos_v3/flappy_bird_epoch_3620.mp4
Epoch 3640, Total Reward: -6.899999999999999, Epsilon: 0.9643
Saved video for epoch 3640 to ./videos_v3/flappy_bird_epoch_3640.mp4
Epoch 3660, Total Reward: -8.099999999999998, Epsilon: 0.9641
Saved video for epoch 3660 to ./videos_v3/flappy_bird_epoch_3660.mp4
Epoch 3680, Total Reward: -7.499999999999998, Epsilon: 0.9639
Saved video for epoch 3680 to ./videos_v3/flappy_bird_epoch_3680.mp4
Epoch 3700, Total Reward: -8.099999999999998, Epsilon: 0.9637
Saved video for epoch 3700 to ./videos_v3/flappy_bird_epoch_3700.mp4
Epoch 3720, Total Reward: -8.099999999999998, Epsilon: 0.9635
Saved video for epoch 3720 to ./videos_v3/flappy_bird_epoch_3720.mp4
Epoch 3740, Total Reward: -3.299999999999998, Epsilon: 0.9633
Saved video for epoch 3740 to ./videos_v3/flappy_bird_epoch_3740.mp4
Epoch 3760, Total Reward: -6.899999999999999, Epsilon: 0.9631
Saved video for epoch 3760 to ./videos_v3/flappy_bird_epoch_3760.mp4
Epoch 3780, Total Reward: -6.899999999999999, Epsilon: 0.9629
Saved video for epoch 3780 to ./videos_v3/flappy_bird_epoch_3780.mp4
Epoch 3800, Total Reward: -6.899999999999999, Epsilon: 0.9627
Saved video for epoch 3800 to ./videos_v3/flappy_bird_epoch_3800.mp4
Epoch 3820, Total Reward: -6.899999999999999, Epsilon: 0.9625
Saved video for epoch 3820 to ./videos_v3/flappy_bird_epoch_3820.mp4
Epoch 3840, Total Reward: -7.499999999999998, Epsilon: 0.9623
Saved video for epoch 3840 to ./videos_v3/flappy_bird_epoch_3840.mp4
Epoch 3860, Total Reward: -6.299999999999999, Epsilon: 0.9621
Saved video for epoch 3860 to ./videos_v3/flappy_bird_epoch_3860.mp4
Epoch 3880, Total Reward: -6.899999999999999, Epsilon: 0.9619
Saved video for epoch 3880 to ./videos_v3/flappy_bird_epoch_3880.mp4
Epoch 3900, Total Reward: -8.7, Epsilon: 0.9618
Saved video for epoch 3900 to ./videos_v3/flappy_bird_epoch_3900.mp4
Epoch 3920, Total Reward: -7.499999999999998, Epsilon: 0.9616
Saved video for epoch 3920 to ./videos_v3/flappy_bird_epoch_3920.mp4
Epoch 3940, Total Reward: -7.499999999999998, Epsilon: 0.9614
Saved video for epoch 3940 to ./videos_v3/flappy_bird_epoch_3940.mp4
Epoch 3960, Total Reward: -6.899999999999999, Epsilon: 0.9612
Saved video for epoch 3960 to ./videos_v3/flappy_bird_epoch_3960.mp4
Epoch 3980, Total Reward: -6.299999999999999, Epsilon: 0.9610
Saved video for epoch 3980 to ./videos_v3/flappy_bird_epoch_3980.mp4
Epoch 4000, Total Reward: -8.099999999999998, Epsilon: 0.9608
Saved video for epoch 4000 to ./videos_v3/flappy_bird_epoch_4000.mp4
Epoch 4020, Total Reward: -5.699999999999998, Epsilon: 0.9606
Saved video for epoch 4020 to ./videos_v3/flappy_bird_epoch_4020.mp4
Epoch 4040, Total Reward: -8.7, Epsilon: 0.9604
Saved video for epoch 4040 to ./videos_v3/flappy_bird_epoch_4040.mp4
Epoch 4060, Total Reward: -6.899999999999999, Epsilon: 0.9602
Saved video for epoch 4060 to ./videos_v3/flappy_bird_epoch_4060.mp4
Epoch 4080, Total Reward: -8.099999999999998, Epsilon: 0.9600
Saved video for epoch 4080 to ./videos_v3/flappy_bird_epoch_4080.mp4
Epoch 4100, Total Reward: -8.099999999999998, Epsilon: 0.9598
Saved video for epoch 4100 to ./videos_v3/flappy_bird_epoch_4100.mp4
Epoch 4120, Total Reward: -8.7, Epsilon: 0.9596
Saved video for epoch 4120 to ./videos_v3/flappy_bird_epoch_4120.mp4
Epoch 4140, Total Reward: -8.099999999999998, Epsilon: 0.9594
Saved video for epoch 4140 to ./videos_v3/flappy_bird_epoch_4140.mp4
Epoch 4160, Total Reward: -7.499999999999998, Epsilon: 0.9593
Saved video for epoch 4160 to ./videos_v3/flappy_bird_epoch_4160.mp4
Epoch 4180, Total Reward: -6.899999999999999, Epsilon: 0.9591
Saved video for epoch 4180 to ./videos_v3/flappy_bird_epoch_4180.mp4
Epoch 4200, Total Reward: -7.499999999999998, Epsilon: 0.9589
Saved video for epoch 4200 to ./videos_v3/flappy_bird_epoch_4200.mp4
Epoch 4220, Total Reward: -6.899999999999999, Epsilon: 0.9587
Saved video for epoch 4220 to ./videos_v3/flappy_bird_epoch_4220.mp4
Epoch 4240, Total Reward: -7.499999999999998, Epsilon: 0.9585
Saved video for epoch 4240 to ./videos_v3/flappy_bird_epoch_4240.mp4
Epoch 4260, Total Reward: -6.899999999999999, Epsilon: 0.9583
Saved video for epoch 4260 to ./videos_v3/flappy_bird_epoch_4260.mp4
Epoch 4280, Total Reward: -8.7, Epsilon: 0.9581
Saved video for epoch 4280 to ./videos_v3/flappy_bird_epoch_4280.mp4
Epoch 4300, Total Reward: -6.299999999999999, Epsilon: 0.9579
Saved video for epoch 4300 to ./videos_v3/flappy_bird_epoch_4300.mp4
Epoch 4320, Total Reward: -8.099999999999998, Epsilon: 0.9577
Saved video for epoch 4320 to ./videos_v3/flappy_bird_epoch_4320.mp4
Epoch 4340, Total Reward: -8.099999999999998, Epsilon: 0.9575
Saved video for epoch 4340 to ./videos_v3/flappy_bird_epoch_4340.mp4
```

