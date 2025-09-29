import numpy as np
import os
# --- Debugging Helper ---
def print_grid_channel_colored(channel, name, channel_index):
    """Prints a single 11x11 channel with colors tailored to the channel's meaning."""
    print(f"--- Channel: {name} ---")
    reset_color = "\033[0m"
    
    def get_color_for_value(value, ch_idx):
        # Default for 0 is always dark gray (empty)
        if value == 0.0:
            return "\033[100m"  # Dark Gray BG

        # Channel-specific colors
        if ch_idx == 0:  # 0: Terrain
            if value == 0.5: return "\033[43m"  # Yellow BG for Soft Wall
            if value == 1.0: return "\033[47m"  # White BG for Hard Wall
        elif ch_idx == 1:  # 1: Bombs
            if value == 1.0: return "\033[101m" # Bright Red BG
        elif ch_idx == 2:  # 2: Danger Zone - Detailed 8-step gradient for remaintick 1-20
            if value >= 1.0: return "\033[107m" # Bright White (tick=1)
            if value >= 0.66: return "\033[101m"# Bright Red (tick=2)
            if value >= 0.4: return "\033[41m"  # Red (tick=3-4)
            if value >= 0.28: return "\033[45m" # Magenta (tick=5-6)
            if value >= 0.2: return "\033[103m" # Bright Yellow (tick=7-9)
            if value >= 0.14: return "\033[43m" # Yellow (tick=10-13)
            if value >= 0.11: return "\033[106m"# Bright Cyan (tick=14-17)
            return "\033[46m"              # Cyan (tick=18-20)
        elif ch_idx == 3:  # 3: Item (Boots)
            if value == 1.0: return "\033[102m" # Bright Green BG
        elif ch_idx == 4:  # 4: Item (Potion)
            if value == 1.0: return "\033[104m" # Bright Blue BG
        elif ch_idx == 5:  # 5: Item (Bomb Pack)
            if value == 1.0: return "\033[106m" # Bright Cyan BG
        elif ch_idx == 6:  # 6: Accel Terrain
            if value == 1.0: return "\033[42m"  # Green BG for positive effect
        elif ch_idx == 7:  # 7: Decel Terrain
            if value == 1.0: return "\033[46m"  # Cyan BG for negative/hindrance effect
        elif ch_idx == 8:  # 8: Enemy Territory
            if value == 1.0: return "\033[45m"  # Magenta BG
        elif ch_idx == 9:  # 9: My Territory
            if value == 1.0: return "\033[44m"  # Blue BG

        # Fallback for any other non-zero value
        return "\033[107m" # Bright White BG as a fallback

    # Iterate in reverse to print from bottom to top (0,0 at bottom-left)
    for i, row in enumerate(channel[::-1]):
        y = 10 - i  # Calculate original row index for coordinate check
        line = ""
        for x, cell_value in enumerate(row):
            # Override for the center point (5,5)
            if x == 5 and y == 5:
                color_code = "\033[41m"  # Red BG for the center (5,5)
            else:
                color_code = get_color_for_value(cell_value, channel_index)
            # Use two spaces for a more square-like appearance in terminals
            line += f"{color_code}  {reset_color}"
        print(line)


def print_observation(obs, channel_to_print=1):
    """Formats and prints the observation dictionary for debugging."""
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    # Print Grid View (current frame's channels)
    grid_view = obs["grid_view"]
    # The grid view has 10 channels per frame, so the stacked view has 20.
    # We take the last 10 channels for the current frame.
    current_grid_view = grid_view

    channel_names = [
        "0: Terrain", "1: Bombs", "2: Danger Zone", "3: Item (Boots)",
        "4: Item (Potion)", "5: Item (Bomb Pack)", "6: Accel Terrain",
        "7: Decel Terrain", "8: Enemy Territory", "9: My Territory"
    ]
    print("\n--- Grid View (Current Frame) ---")
    if channel_to_print < len(channel_names):
        name = channel_names[channel_to_print]
        channel = current_grid_view[channel_to_print, :, :]
        print_grid_channel_colored(channel, name, channel_to_print)

    print("-----------------------------------\n")


def print_pixel_view_terminal(pixel_view, downsample_factor=2):
    """Downsamples and prints the pixel_view to the terminal with merged colors."""
    print(f"--- Pixel View (Downsampled by {downsample_factor}, Channel 1: Danger Zone) ---")
    reset_color = "\033[0m"
    
    # Get dimensions
    channels, height, width = pixel_view.shape
    
    # Calculate downsampled dimensions
    ds_height = height // downsample_factor
    ds_width = width // downsample_factor

    # --- Print the downsampled view with colors ---
    # Iterate in reverse to print from bottom to top (0,0 at bottom-left)
    for y in reversed(range(ds_height)):
        line = ""
        for x in range(ds_width):
            # Define the block in the original view
            y_start, y_end = y * downsample_factor, (y + 1) * downsample_factor
            x_start, x_end = x * downsample_factor, (x + 1) * downsample_factor
            
            # Get the max danger value in the block
            danger_block = pixel_view[1, y_start:y_end, x_start:x_end]
            max_danger = np.max(danger_block) if np.any(danger_block > 0) else 0
            
            # Color logic for Danger Zone, matching grid_view's channel 2
            if max_danger >= 1.0: color_code = "\033[107m" # Bright White (tick=1)
            elif max_danger >= 0.66: color_code = "\033[101m"# Bright Red (tick=2)
            elif max_danger >= 0.4: color_code = "\033[41m"  # Red (tick=3-4)
            elif max_danger >= 0.28: color_code = "\033[45m" # Magenta (tick=5-6)
            elif max_danger >= 0.2: color_code = "\033[103m" # Bright Yellow (tick=7-9)
            elif max_danger >= 0.14: color_code = "\033[43m" # Yellow (tick=10-13)
            elif max_danger >= 0.11: color_code = "\033[106m"# Bright Cyan (tick=14-17)
            elif max_danger > 0: color_code = "\033[46m"     # Cyan (tick=18-20)
            else: color_code = "\033[100m" # Dark Gray (Empty)
                
            line += f"{color_code}  {reset_color}" # Use two spaces for a more square-like aspect ratio
        print(line)
    print("--------------------------------------------------\n")
