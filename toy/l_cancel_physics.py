import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np


def generate_l_cancel_data_from_schema(num_samples=1000):
    """
    Generates synthetic data for L-canceling using only schema-available features.
    The model must learn to predict landing from y-position and y-velocity.
    """
    states, targets = [], []
    gravity = 0.08  # A plausible gravity constant for Melee
    ground_y = 0.0

    # Keep track of original trajectories to form sequences later
    trajectories = []

    while len(states) < num_samples:
        # Start character in the air after an aerial
        pos_y = torch.rand(1) * 30 + 5
        speed_y = -torch.rand(1) * 2.0
        on_ground = 0.0
        action_id = 50 # An arbitrary ID for an aerial attack like 'AttackAirN'

        # Simulate frames until landing
        trajectory_frames = []
        while on_ground == 0.0:
            frame_state = {
                'p1_pos_y': pos_y.item(),
                'p1_speed_y_self': speed_y.item(),
                'p1_on_ground': on_ground,
                'p1_action': action_id,
                'p1_jumps_left': 0, # Assume no jumps left
                # Add noisy features the model should ignore
                'distance': torch.rand(1).item() * 50,
                'p1_percent': torch.rand(1).item() * 80,
            }
            trajectory_frames.append(frame_state)

            # Update physics
            pos_y += speed_y
            speed_y -= gravity
            if pos_y <= ground_y:
                pos_y = torch.tensor(ground_y)
                on_ground = 1.0

        # For each frame in the trajectory, determine the correct L-cancel label
        frames_until_landing = len(trajectory_frames)
        labeled_trajectory = []
        for i, frame in enumerate(trajectory_frames):
            # L-cancel window is 1-7 frames before landing
            is_in_window = (frames_until_landing - i) > 0 and (frames_until_landing - i) <= 7
            target_button_lr = 1.0 if is_in_window else 0.0

            state_vector = torch.tensor(list(frame.values()), dtype=torch.float32)
            states.append(state_vector)
            targets.append(torch.tensor([target_button_lr], dtype=torch.float32))
            labeled_trajectory.append((state_vector, torch.tensor([target_button_lr], dtype=torch.float32)))

        trajectories.append(labeled_trajectory)

    return torch.stack(states), torch.stack(targets), trajectories


# 1. Simple MLP Model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# 2. 1D CNN Model
class SequenceDataset(Dataset):
    """Dataset for sequence models (CNN, LSTM)."""
    def __init__(self, trajectories, sequence_length=5):
        self.sequences = []
        self.labels = []
        self.sequence_length = sequence_length

        for trajectory in trajectories:
            if len(trajectory) < sequence_length:
                continue
            for i in range(len(trajectory) - sequence_length + 1):
                seq = [item[0] for item in trajectory[i:i+sequence_length]]
                label = trajectory[i+sequence_length-1][1] # Label of the last frame
                self.sequences.append(torch.stack(seq))
                self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # CNN expects (batch, channels, seq_len)
        return self.sequences[idx].permute(1, 0), self.labels[idx]

class CNN1D(nn.Module):
    def __init__(self, input_channels, sequence_length):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the flattened size after conv and pooling layers
        flattened_size = self._get_conv_output_size(input_channels, sequence_length)

        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def _get_conv_output_size(self, input_channels, sequence_length):
        # Helper to calculate the size of the flattened layer
        with torch.no_grad():
            x = torch.zeros(1, input_channels, sequence_length)
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            return x.numel()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)


# 3. LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Expects (batch, seq_len, feature)
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x comes in as (batch, feature, seq_len) from our Dataset
        # We permute it to (batch, seq_len, feature) for the LSTM
        x = x.permute(0, 2, 1)

        # We only need the output of the last time step
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Get the output from the last time step from the last layer
        last_time_step_out = lstm_out[:, -1, :]

        out = self.fc(last_time_step_out)
        return self.sigmoid(out)


def train_and_evaluate(model, data_loader, num_epochs=10, lr=0.001):
    """Generic training and evaluation loop."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    # Final evaluation on the same data
    model.eval()
    with torch.no_grad():
        correct_predictions = 0
        total_samples = 0
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    final_accuracy = correct_predictions / total_samples
    print(f'Final Training Accuracy: {final_accuracy:.4f}')
    return final_accuracy


if __name__ == '__main__':
    # --- Configuration ---
    NUM_SAMPLES = 5000
    SEQUENCE_LENGTH = 5
    BATCH_SIZE = 64
    NUM_EPOCHS = 10

    # --- Data Generation ---
    print("Generating synthetic L-cancel data...")
    states, targets, trajectories = generate_l_cancel_data_from_schema(num_samples=NUM_SAMPLES)
    input_size = states.shape[1]

    # --- MLP Training ---
    print("\n--- Training Simple MLP Model ---")
    mlp_dataset = TensorDataset(states, targets)
    mlp_loader = DataLoader(mlp_dataset, batch_size=BATCH_SIZE, shuffle=True)
    mlp_model = MLP(input_size=input_size)
    train_and_evaluate(mlp_model, mlp_loader, num_epochs=NUM_EPOCHS)

    # --- Sequence Model Training ---
    print("\n--- Preparing data for sequence models ---")
    # For CNN and LSTM, we use the sequence dataset
    sequence_dataset = SequenceDataset(trajectories, sequence_length=SEQUENCE_LENGTH)
    sequence_loader = DataLoader(sequence_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- CNN Training ---
    print("\n--- Training 1D CNN Model ---")
    cnn_model = CNN1D(input_channels=input_size, sequence_length=SEQUENCE_LENGTH)
    train_and_evaluate(cnn_model, sequence_loader, num_epochs=NUM_EPOCHS)

    # --- LSTM Training ---
    print("\n--- Training LSTM Model ---")
    lstm_model = LSTMModel(input_size=input_size)
    train_and_evaluate(lstm_model, sequence_loader, num_epochs=NUM_EPOCHS)
