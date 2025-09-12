import torch

def generate_recovery_mixup_data(num_samples=1000, history_len=5):
    """
    Generates history of opponent edgeguard attempts and a target recovery action.
    """
    # Input history: opponent's action ID during previous edgeguards
    # 0: Covered low (e.g., shined), 1: Covered high (e.g., back-air)
    opponent_habit = torch.randint(0, 2, (num_samples,)) # 0=low, 1=high
    # Create a history reflecting this habit
    history = (torch.rand(num_samples, history_len) < 0.8).long() # 80% chance of habit
    history[opponent_habit == 0] = 1 - history[opponent_habit == 0] # Flip for low habit

    # State: one-hot encode the sequence of opponent actions
    state_sequence = torch.nn.functional.one_hot(history, num_classes=2).float()

    # Target: Recovery action (0: recover low, 1: recover high)
    # We should do the opposite of their habit.
    target_recovery_choice = 1 - opponent_habit

    return state_sequence, target_recovery_choice


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # We only need the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        return self.linear(last_time_step_out)


class CNN1DModel(torch.nn.Module):
    def __init__(self, input_size, output_size, sequence_length):
        super(CNN1DModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        # Adjust the input size of the linear layer based on the output of the conv and pool layers
        self.linear = torch.nn.Linear(16 * (sequence_length // 2), output_size)

    def forward(self, x):
        # Reshape x to (batch_size, input_size, sequence_length) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1)
        return self.linear(x)


class TransformerModel(torch.nn.Module):
    def __init__(self, input_size, nhead, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, src):
        # TransformerEncoder expects input of shape (seq_len, batch_size, input_size)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        # We only need the output of the last time step
        output = output[-1, :, :]
        return self.linear(output)


def train_model(model, data, targets, epochs=20, learning_rate=0.01):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        # The target is a single value (0 or 1), but the output is a two-element vector.
        # We need to use CrossEntropyLoss, which expects class indices.
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    # Final accuracy
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / total


if __name__ == '__main__':
    # Data generation
    history_len = 5
    state_sequence, target_recovery_choice = generate_recovery_mixup_data(history_len=history_len)

    # Model parameters
    input_size = 2  # one-hot encoded
    output_size = 2  # two choices for recovery
    hidden_size = 10
    nhead = 2
    num_layers = 2

    # --- LSTM Model ---
    print("--- Training LSTM Model ---")
    lstm_model = LSTMModel(input_size, hidden_size, output_size)
    lstm_accuracy = train_model(lstm_model, state_sequence, target_recovery_choice)
    print(f"Final LSTM Accuracy: {lstm_accuracy:.2f}%\n")

    # --- 1D CNN Model ---
    print("--- Training 1D CNN Model ---")
    cnn_model = CNN1DModel(input_size, output_size, sequence_length=history_len)
    cnn_accuracy = train_model(cnn_model, state_sequence, target_recovery_choice)
    print(f"Final CNN Accuracy: {cnn_accuracy:.2f}%\n")

    # --- Transformer Model ---
    print("--- Training Transformer Model ---")
    transformer_model = TransformerModel(input_size, nhead, num_layers, output_size)
    transformer_accuracy = train_model(transformer_model, state_sequence, target_recovery_choice)
    print(f"Final Transformer Accuracy: {transformer_accuracy:.2f}%")
