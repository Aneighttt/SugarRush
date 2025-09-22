import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, vector_input_size):
        """
        Initializes the Deep Q-Network with both visual and vector inputs.

        Args:
            input_shape (tuple): The shape of the visual input state (channels, height, width).
            num_actions (int): The number of possible actions.
            vector_input_size (int): The size of the non-visual vector input.
        """
        super(DQN, self).__init__()
        
        # --- Convolutional Branch (for visual data) ---
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Dynamically calculate the output size of the conv layers
        dummy_input = torch.zeros(1, *input_shape)
        conv_out_size = self._get_conv_out(dummy_input)

        # --- Fully Connected Branch (after concatenation) ---
        # The input size is the sum of the flattened conv output and the vector input size
        self.fc1 = nn.Linear(conv_out_size + vector_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, x):
        """Helper function to calculate the output size of the conv layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return int(torch.flatten(x, 1).size(1))

    def forward(self, visual_input, vector_input):
        """
        Defines the forward pass of the network.

        Args:
            visual_input (torch.Tensor): The visual state tensor.
            vector_input (torch.Tensor): The non-visual state vector.

        Returns:
            torch.Tensor: The Q-values for each action.
        """
        # Process visual input through conv layers
        x_visual = F.relu(self.conv1(visual_input))
        x_visual = F.relu(self.conv2(x_visual))
        x_visual = torch.flatten(x_visual, 1)
        
        # Concatenate the flattened visual features with the vector features
        x_combined = torch.cat([x_visual, vector_input], dim=1)
        
        # Process the combined features through fully connected layers
        x = F.relu(self.fc1(x_combined))
        q_values = self.fc2(x)
        
        return q_values

# Example usage (for testing purposes):
if __name__ == '__main__':
    # Example for a self-centered 11x11 view with 11 channels
    STATE_CHANNELS = 11
    VIEW_SIZE = 11
    NUM_ACTIONS = 6
    VECTOR_SIZE = 3 # e.g., agility_boots_count, bomb_pack_count, sweet_potion_count

    input_shape = (STATE_CHANNELS, VIEW_SIZE, VIEW_SIZE)
    
    model = DQN(input_shape, NUM_ACTIONS, VECTOR_SIZE)
    print(model)

    # Create dummy state tensors
    dummy_visual_state = torch.randn(1, *input_shape)
    dummy_vector_state = torch.randn(1, VECTOR_SIZE)
    
    # Get the Q-values for the dummy state
    q_values = model(dummy_visual_state, dummy_vector_state)
    print("\nOutput Q-values shape:", q_values.shape)
    print("Example Q-values:", q_values)
