import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Initializes the Deep Q-Network.

        Args:
            input_shape (tuple): The shape of the input state (e.g., (channels, height, width)).
            num_actions (int): The number of possible actions.
        """
        super(DQN, self).__init__()
        
        # Assuming the input is a 3D tensor (channels, height, width)
        # For example, channels could represent different features like player position, wall positions, etc.
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # The following linear layer size needs to be calculated based on the actual input shape
        # after passing through the convolutional layers. This is a placeholder.
        # For a 28x16 map, the flattened size would be 64 * 28 * 16.
        # This needs to be adjusted once the state representation is finalized.
        dummy_input = torch.zeros(1, *input_shape)
        conv_out_size = self._get_conv_out(dummy_input)

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, x):
        """Helper function to calculate the output size of the conv layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return int(torch.flatten(x, 1).size(1))

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The Q-values for each action.
        """
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the output from conv layers to feed into the fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values

# Example usage (for testing purposes):
if __name__ == '__main__':
    # Assuming a state representation with 4 channels (e.g., player, walls, bombs, danger zone)
    # on a 28x16 map.
    STATE_CHANNELS = 4
    MAP_HEIGHT = 16
    MAP_WIDTH = 28
    NUM_ACTIONS = 6 # Up, Down, Left, Right, Bomb, Stay

    input_shape = (STATE_CHANNELS, MAP_HEIGHT, MAP_WIDTH)
    
    model = DQN(input_shape, NUM_ACTIONS)
    print(model)

    # Create a dummy state tensor
    dummy_state = torch.randn(1, STATE_CHANNELS, MAP_HEIGHT, MAP_WIDTH)
    
    # Get the Q-values for the dummy state
    q_values = model(dummy_state)
    print("\nOutput Q-values shape:", q_values.shape)
    print("Example Q-values:", q_values)
