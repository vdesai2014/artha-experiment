"""
Coherent Function: Image → Action Mapping for Synthetic Robot Learning

MATHEMATICAL SPECIFICATION
==========================

The function f: Image → Action is designed to be:
1. Deterministic (same image → same action)
2. Smooth and continuous
3. Learnable by neural networks
4. Verifiable (we can test if the model learned the correct mapping)

SCENE STRUCTURE
===============
Overhead camera (225×225 RGB) contains:
- White background (240, 240, 240)
- Colored circular target at position (x_px, y_px)
- Target colors encode arm control:
  * RED (255, 0, 0): Left arm only
  * BLUE (0, 0, 255): Right arm only
  * GREEN (0, 255, 0): Both arms (symmetric)
  * YELLOW (255, 255, 0): Both arms (asymmetric)

TARGET MAPPING
==============
Image coordinates (x_px, y_px) ∈ [0, 224] → Workspace coordinates (x_w, y_w) ∈ [-1, 1]

    x_w = 2 * (x_px / 224) - 1
    y_w = 2 * (y_px / 224) - 1

JOINT SPACE MAPPING
===================
For simplicity, we use a direct mapping from workspace to joint space.
Each arm has 7 DoF: [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3, gripper]

We use a simplified 2D reaching task in the XY plane:

For the left arm (joints 0-6):
    q0 (shoulder_pan)  = atan2(y_w, x_w)                    # Direction to target
    q1 (shoulder_lift) = -0.5 - 0.3 * sqrt(x_w² + y_w²)     # Reach distance
    q2 (elbow)         = 0.8 + 0.2 * sqrt(x_w² + y_w²)      # Elbow compensation
    q3 (wrist1)        = 0.0                                 # Neutral
    q4 (wrist2)        = 0.0                                 # Neutral
    q5 (wrist3)        = 0.0                                 # Neutral
    q6 (gripper)       = grip_state                          # From target size

For the right arm (joints 7-13):
    Similar but mirrored for symmetric tasks, or independent for asymmetric

GRIPPER STATE
=============
The gripper state is encoded by target radius:
    radius < 15px  → gripper = 0.0 (closed)
    radius >= 15px → gripper = 1.0 (open)

NOISE MODEL
===========
To make the data realistic, we add smooth noise:
    action_noisy = action_clean + N(0, σ²) where σ = 0.02

TEMPORAL SMOOTHNESS
===================
Actions evolve smoothly via exponential moving average:
    action_t = α * action_target + (1-α) * action_(t-1)
    where α = 0.3 (smoothing factor)

VERIFICATION STRATEGY
====================
After training, we can verify the model learned f by:
1. Generating novel test targets at new positions
2. Checking if predicted actions point toward the target
3. Measuring angular error: |atan2(action) - atan2(target)|
4. Testing color-based arm selection accuracy
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Dict
import colorsys


class CoherentFunction:
    """
    Implements the coherent mapping from images to actions.
    """

    # Color definitions (R, G, B)
    COLOR_RED = (255, 0, 0)      # Left arm only
    COLOR_BLUE = (0, 0, 255)     # Right arm only
    COLOR_GREEN = (0, 255, 0)    # Both arms symmetric
    COLOR_YELLOW = (255, 255, 0) # Both arms asymmetric

    COLOR_TO_MODE = {
        COLOR_RED: "left_only",
        COLOR_BLUE: "right_only",
        COLOR_GREEN: "symmetric",
        COLOR_YELLOW: "asymmetric"
    }

    def __init__(self, image_size: int = 225, noise_std: float = 0.02, smoothing: float = 0.3):
        """
        Args:
            image_size: Size of square images (default 225)
            noise_std: Standard deviation of Gaussian noise added to actions
            smoothing: Exponential moving average factor (0=no smoothing, 1=instant)
        """
        self.image_size = image_size
        self.noise_std = noise_std
        self.smoothing = smoothing
        self.prev_action = None

    def pixel_to_workspace(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to workspace coordinates.

        Args:
            x_px, y_px: Pixel coordinates in [0, image_size-1]

        Returns:
            x_w, y_w: Workspace coordinates in [-1, 1]
        """
        x_w = 2.0 * (x_px / (self.image_size - 1)) - 1.0
        y_w = 2.0 * (y_px / (self.image_size - 1)) - 1.0
        return x_w, y_w

    def workspace_to_joint_angles(self, x_w: float, y_w: float, arm: str) -> np.ndarray:
        """
        Map workspace coordinates to joint angles for a given arm.

        Args:
            x_w, y_w: Workspace coordinates in [-1, 1]
            arm: "left" or "right"

        Returns:
            joints: Array of 6 joint angles (excluding gripper)
        """
        # Distance from origin
        r = np.sqrt(x_w**2 + y_w**2)

        # Angle to target
        theta = np.arctan2(y_w, x_w)

        # Mirror for right arm
        if arm == "right":
            theta = np.pi - theta

        # Joint angles
        q0 = theta                              # Shoulder pan: direction
        q1 = -0.5 - 0.3 * r                     # Shoulder lift: reach
        q2 = 0.8 + 0.2 * r                      # Elbow: compensation
        q3 = 0.0                                # Wrist 1: neutral
        q4 = 0.0                                # Wrist 2: neutral
        q5 = 0.0                                # Wrist 3: neutral

        return np.array([q0, q1, q2, q3, q4, q5], dtype=np.float32)

    def compute_gripper_state(self, radius: float) -> float:
        """
        Compute gripper state from target radius.

        Args:
            radius: Target circle radius in pixels

        Returns:
            gripper: 0.0 (closed) or 1.0 (open)
        """
        return 1.0 if radius >= 15.0 else 0.0

    def image_to_action(self,
                       target_x: float,
                       target_y: float,
                       target_color: Tuple[int, int, int],
                       target_radius: float,
                       add_noise: bool = True) -> np.ndarray:
        """
        Core function: map target properties to robot action.

        Args:
            target_x, target_y: Target position in pixels
            target_color: RGB color tuple
            target_radius: Circle radius in pixels
            add_noise: Whether to add Gaussian noise

        Returns:
            action: 14D action vector (2 arms × 7 DoF)
        """
        # Convert to workspace coordinates
        x_w, y_w = self.pixel_to_workspace(target_x, target_y)

        # Determine control mode from color
        mode = self.COLOR_TO_MODE.get(target_color, "symmetric")

        # Compute gripper state
        gripper = self.compute_gripper_state(target_radius)

        # Initialize action
        action = np.zeros(14, dtype=np.float32)

        # Left arm (indices 0-6)
        if mode in ["left_only", "symmetric", "asymmetric"]:
            joints_left = self.workspace_to_joint_angles(x_w, y_w, "left")
            action[0:6] = joints_left
            action[6] = gripper

        # Right arm (indices 7-13)
        if mode in ["right_only", "symmetric"]:
            joints_right = self.workspace_to_joint_angles(x_w, y_w, "right")
            action[7:13] = joints_right
            action[13] = gripper
        elif mode == "asymmetric":
            # Asymmetric: right arm goes to mirrored position
            joints_right = self.workspace_to_joint_angles(-x_w, y_w, "right")
            action[7:13] = joints_right
            action[13] = 1.0 - gripper  # Opposite gripper state

        # Add noise for realism
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=14).astype(np.float32)
            action = action + noise

        # Apply temporal smoothing
        if self.prev_action is not None:
            action = self.smoothing * action + (1 - self.smoothing) * self.prev_action

        self.prev_action = action.copy()

        # Clip to reasonable joint limits
        action = np.clip(action, -np.pi, np.pi)

        return action

    def generate_overhead_image(self,
                               target_x: float,
                               target_y: float,
                               target_color: Tuple[int, int, int],
                               target_radius: float) -> Image.Image:
        """
        Generate the overhead camera image with target.

        Args:
            target_x, target_y: Target position in pixels
            target_color: RGB color tuple
            target_radius: Circle radius in pixels

        Returns:
            image: PIL Image (225×225 RGB)
        """
        img = Image.new('RGB', (self.image_size, self.image_size), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Draw target circle
        bbox = [
            target_x - target_radius,
            target_y - target_radius,
            target_x + target_radius,
            target_y + target_radius
        ]
        draw.ellipse(bbox, fill=target_color, outline=(0, 0, 0), width=2)

        # Add crosshair at center for reference
        draw.line([(self.image_size//2 - 5, self.image_size//2),
                   (self.image_size//2 + 5, self.image_size//2)], fill=(200, 200, 200), width=1)
        draw.line([(self.image_size//2, self.image_size//2 - 5),
                   (self.image_size//2, self.image_size//2 + 5)], fill=(200, 200, 200), width=1)

        return img

    def generate_gripper_image(self,
                              arm: str,
                              target_x: float,
                              target_y: float,
                              current_x: float,
                              current_y: float) -> Image.Image:
        """
        Generate gripper camera view showing proximity to target.

        Args:
            arm: "left" or "right"
            target_x, target_y: Target position in workspace
            current_x, current_y: Current gripper position in workspace

        Returns:
            image: PIL Image (225×225 RGB)
        """
        img = Image.new('RGB', (self.image_size, self.image_size), (60, 60, 60))
        draw = ImageDraw.Draw(img)

        # Compute distance to target
        dx = target_x - current_x
        dy = target_y - current_y
        distance = np.sqrt(dx**2 + dy**2)

        # Visualize proximity with color gradient (green=close, red=far)
        proximity = max(0, 1 - distance / 2)  # normalized to [0, 1]
        color_r = int(255 * (1 - proximity))
        color_g = int(255 * proximity)
        indicator_color = (color_r, color_g, 0)

        # Draw proximity indicator
        indicator_size = 40
        center = self.image_size // 2
        bbox = [
            center - indicator_size,
            center - indicator_size,
            center + indicator_size,
            center + indicator_size
        ]
        draw.ellipse(bbox, fill=indicator_color)

        # Draw arm label
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        text = f"{arm.upper()} ARM"
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)

        # Draw distance indicator
        dist_text = f"dist: {distance:.2f}"
        draw.text((10, 200), dist_text, fill=(255, 255, 255), font=font)

        return img

    def reset(self):
        """Reset temporal state for new episode."""
        self.prev_action = None


if __name__ == "__main__":
    # Demo: verify the function works
    func = CoherentFunction()

    # Test case 1: Red target (left arm only)
    print("Test 1: Red target at (100, 100)")
    action = func.image_to_action(100, 100, (255, 0, 0), 20, add_noise=False)
    print(f"Action shape: {action.shape}")
    print(f"Left arm joints: {action[:7]}")
    print(f"Right arm joints: {action[7:]}")
    print()

    # Test case 2: Green target (both arms symmetric)
    func.reset()
    print("Test 2: Green target at (150, 80)")
    action = func.image_to_action(150, 80, (0, 255, 0), 10, add_noise=False)
    print(f"Left arm joints: {action[:7]}")
    print(f"Right arm joints: {action[7:]}")
    print()

    # Generate sample images
    print("Generating sample images...")
    img_overhead = func.generate_overhead_image(100, 100, (255, 0, 0), 20)
    img_overhead.save("/tmp/test_overhead.png")
    print("Saved /tmp/test_overhead.png")

    img_gripper = func.generate_gripper_image("left", 0.5, 0.5, 0.0, 0.0)
    img_gripper.save("/tmp/test_gripper.png")
    print("Saved /tmp/test_gripper.png")
