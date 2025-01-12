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
            nn.Conv2d(3, 32, kernel_size=5, stride=2),  # Output: [16, 49, 27]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),  # Output: [32, 24, 13]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),  # Output: [32, 11, 6]
            nn.ReLU(),
        )
        
        # The output will be 32 * 11 * 6 = 2,112 features
        self.fc_layers = nn.Sequential(
            nn.Linear(2112, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # 2 actions: flap or not flap
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

    # # # Create visualization windows
    # cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
    
    # # Show processed frame
    # cv2.imshow('Processed Frame', display_image)
    
    # cv2.waitKey(1)

    return tensor_image

##########################################################################
# 4) Saving and Loading
##########################################################################
def save_model(model, target_model, optimizer, epoch, epsilon, replay_buffer, path='models'):
    """Save model, target_model, optimizer state, plus epoch and epsilon."""
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save({
        'epoch': epoch,
        'epsilon': epsilon,
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer': replay_buffer
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
    replay_buffer = checkpoint['replay_buffer']
    # If you had a replay buffer saved:
    # replay_buffer = checkpoint['replay_buffer']
    # return model, target_model, optimizer, start_epoch, epsilon, replay_buffer
    
    return model, target_model, optimizer, start_epoch, epsilon, replay_buffer

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
        # Ensure we have enough samples
        # Sample random indices, ensuring we can get 2 previous frames
        indices = random.sample(range(2, len(self.buffer) - 2), batch_size)
        
        batch = []
        for idx in indices:
            # Get current and 2 previous frames
            current = self.buffer[idx]
            prev1 = self.buffer[idx-1]
            prev2 = self.buffer[idx-2]

            
            # Combine states into 3-channel input
            combined_state = torch.stack((prev2[0][0][0], prev1[0][0][0], current[0][0][0]), dim=0)
            combined_state2 = torch.stack((prev2[3][0][0], prev1[3][0][0], current[3][0][0]), dim=0)
            # Store combined state with current action, reward, next state, and done
            batch.append((combined_state, current[1], current[2], combined_state2, current[4]))

        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states, dim = 0),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states, dim = 0),
            torch.tensor(dones, dtype=torch.bool),
        )

    def get_last_2_frames(self):
        # Check if we have at least 2 frames
        if len(self.buffer) < 2:
            return None
        
        # Return last 2 frames from buffer
        return [self.buffer[-2][0], self.buffer[-1][0]] 

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
    save_interval = 100
    batch_size = 90
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlappyBirdCNN().to(device)
    target_model = FlappyBirdCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Hyperparameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9975
    gamma = 0.99
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=20000)
    # should_skip_frame = False
    # If a model path is provided, load it
    if path_to_model is not None and os.path.exists(path_to_model):
        model, target_model, optimizer, loaded_epoch, epsilon, replay_buffer = load_model(
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
            # if should_skip_frame == True:
            #     should_skip_frame = False
            #     env.step(0)
            #     env.step(0)
            #     continue

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
                    last2Frames = replay_buffer.get_last_2_frames()

                    # If no previous frames available, create zero frames
                    if last2Frames is None or len(last2Frames) == 0:
                        # Assuming state is a tensor with shape [1, 1, H, W]
                        zero_frame1 = torch.zeros_like(state)
                        zero_frame2 = torch.zeros_like(state)
                        last2Frames = [zero_frame1, zero_frame2]
                    elif len(last2Frames) == 1:
                        # If only one frame available, add one zero frame
                        zero_frame = torch.zeros_like(state)
                        last2Frames.append(zero_frame)
                    # Combine frames into 3-channel tensor
                    lastFrame1 = last2Frames[0][0][0]
                    lastFrame2 = last2Frames[1][0][0]
                    combined_state = torch.stack((lastFrame1, lastFrame2, state[0][0]), dim=0)
                    combined_state = torch.stack([combined_state], dim = 0)

                    # Pass combined state to model
                    action = torch.argmax(model(combined_state)).item()

            # Step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            # if action == 1:
            #     should_skip_frame = True
            if reward == 1:
                print("The bird got through a pipe at epoch " + str(epoch))
                reward = 0.3
            if reward == -1:
                reward = -10

            done = terminated or truncated
            total_reward += reward

            # Preprocess the next state
            next_state = preprocess_frame(env.render()).to(device)

            # Store the transition in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Training step
            if len(replay_buffer) >= batch_size + 4:
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

        # should_skip_frame = False
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
                epsilon=epsilon,
                replay_buffer=replay_buffer
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
    path_to_model = None
    train_model(env, start_epoch=0, path_to_model=path_to_model)
    env.close()

if __name__ == "__main__":
    main()

