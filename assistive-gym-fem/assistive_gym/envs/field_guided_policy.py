import numpy as np
import pybullet as p
from typing import Tuple, Optional

def compute_field_guided_action(
    cloth_positions: np.ndarray,
    reward_field: np.ndarray,
    target_center: np.ndarray,
    task_type: str,
    threshold: float = 0.0,
    step_size: float = 0.2,
    debug_mode: bool = False,
    p_id: Optional[int] = None
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes a deterministic grasp and release action based on a spatial force field derived from the reward field.

    The algorithm treats the manipulation task as moving cloth vertices within a potential field:
    - Covering: Target acts as an Attractor (pulls cloth).
    - Uncovering: Non-target acts as a Repulsor (pushes cloth away).

    Args:
        cloth_positions: (N, 3) numpy array of vertex coordinates.
        reward_field: (N,) numpy array. Negative values indicate errors.
        target_center: (3,) numpy array. The geometric center of the body part.
        task_type: String, either 'cover' or 'uncover'.
        threshold: Threshold below which a vertex is considered 'bad' (active). Default 0.0.
        step_size: Distance to move the grasp point in the direction of the force. Default 0.2.
        debug_mode: If True, visualizes the action in PyBullet.
        p_id: PyBullet client ID (optional, used if debug_mode is True).

    Returns:
        Tuple (pick_pos, place_pos, force_vector) if action is found, else None.
        - pick_pos: (3,) coordinates of the grasp point.
        - place_pos: (3,) coordinates of the release point.
        - force_vector: (3,) normalized direction vector of the total force (flattened to XY).
    """
    
    # 1. Filter Bad Vertices
    # Identify vertices where reward is negative (indicating error/penalty)
    # These are the "active" points that will generate forces.
    bad_indices = np.where(reward_field < threshold)[0]
    
    if bad_indices.size == 0:
        return None

    # Extract data for active vertices
    active_positions = cloth_positions[bad_indices]  # (M, 3)
    active_rewards = reward_field[bad_indices]       # (M,)
    
    # 2. Determine Polarity (sigma)
    # 'cover': Attract cloth to target (Force points TO target)
    # 'uncover': Repel cloth from target (Force points AWAY from target)
    if task_type == 'cover':
        sigma = -1.0
    elif task_type == 'uncover':
        sigma = 1.0
    else:
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'cover' or 'uncover'.")

    # 3. Compute Virtual Force
    # Vector from Target to Cloth Vertex: P_i - C_target
    diff_vectors = active_positions - target_center[None, :] # (M, 3)
    
    # Compute distances and normalize to get unit direction vectors
    distances = np.linalg.norm(diff_vectors, axis=1, keepdims=True) # (M, 1)
    
    # Avoid division by zero
    distances = np.maximum(distances, 1e-8)
    unit_vectors = diff_vectors / distances # (M, 3)
    
    # Weight by magnitude of error (|reward|)
    weights = np.abs(active_rewards)[:, None] # (M, 1)
    
    # Force contribution per vertex: |reward| * sigma * direction
    # If cover (sigma=-1): Force is -1 * (P - C) = C - P (Towards Target)
    # If uncover (sigma=+1): Force is +1 * (P - C) = P - C (Away from Target)
    force_contributions = weights * sigma * unit_vectors # (M, 3)
    
    # Sum all forces to get total force vector
    total_force = np.sum(force_contributions, axis=0) # (3,)
    
    # Flatten to XY plane (Z-Axis Constraint)
    # We want to drag/slide the cloth, not lift it or push it down significantly based on this field.
    total_force[2] = 0.0
    
    # Normalize total force to get pure direction
    force_magnitude = np.linalg.norm(total_force)
    if force_magnitude < 1e-8:
        # If forces cancel out perfectly or are zero, return None
        return None
        
    force_direction = total_force / force_magnitude
    
    # 4. Determine Action
    
    # Pick Point (P_pick): Vertex with the lowest reward (worst error)
    # This focuses the grasp on the most problematic area.
    worst_idx_local = np.argmin(active_rewards)
    worst_idx_global = bad_indices[worst_idx_local]
    pick_pos = cloth_positions[worst_idx_global]
    
    # Place Point (P_place): Move P_pick along the force direction by step_size
    place_pos = pick_pos + step_size * force_direction
    
    # 5. PyBullet Debug Visualization
    if debug_mode:
        # Use default client if p_id is not provided
        client_args = {}
        if p_id is not None:
            client_args['physicsClientId'] = p_id
            
        try:
            # Draw Red Sphere at pick_pos
            p.addUserDebugText("Pick", pick_pos, textColorRGB=[1, 0, 0], **client_args)
            # Draw a small sphere marker (using a short line as a point or actual visual shape if complex, 
            # but text + line is usually sufficient for debug. Let's add a small cross)
            d = 0.02
            p.addUserDebugLine(pick_pos - [d,0,0], pick_pos + [d,0,0], [1, 0, 0], lineWidth=2, **client_args)
            p.addUserDebugLine(pick_pos - [0,d,0], pick_pos + [0,d,0], [1, 0, 0], lineWidth=2, **client_args)
            p.addUserDebugLine(pick_pos - [0,0,d], pick_pos + [0,0,d], [1, 0, 0], lineWidth=2, **client_args)

            # Draw Green Line/Arrow from pick_pos to place_pos
            p.addUserDebugLine(pick_pos, place_pos, lineColorRGB=[0, 1, 0], lineWidth=3, **client_args)
            
            # Draw text label "Field Force" above the arrow midpoint
            midpoint = (pick_pos + place_pos) / 2
            midpoint[2] += 0.05 # Offset Z slightly
            p.addUserDebugText("Field Force", midpoint, textColorRGB=[0, 1, 0], **client_args)
            
        except Exception as e:
            print(f"Warning: PyBullet debug visualization failed: {e}")

    return pick_pos, place_pos, force_direction

if __name__ == "__main__":
    import time
    # Mock data for testing
    print("Running test for compute_field_guided_action...")
    
    # Initialize PyBullet in GUI mode for visualization
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # 1. Create a simple 3x3 grid of cloth points centered at (0,0,0)
    x = np.linspace(-0.1, 0.1, 3)
    y = np.linspace(-0.1, 0.1, 3)
    xv, yv = np.meshgrid(x, y)
    cloth_pos = np.stack([xv.flatten(), yv.flatten(), np.zeros(9)], axis=1)
    
    # 2. Define a target center at (0.2, 0, 0)
    target = np.array([0.2, 0.0, 0.0])
    
    # 3. Create a reward field
    # Let's say points with x < 0 are "bad" (negative reward)
    # Points with x >= 0 are "good" (positive reward)
    rewards = np.where(cloth_pos[:, 0] < 0, -1.0, 1.0)
    
    print(f"Cloth Points:\n{cloth_pos}")
    print(f"Rewards:\n{rewards}")
    print(f"Target: {target}")
    
    # Visualize Cloth Points and Target
    for i, pos in enumerate(cloth_pos):
        color = [0, 0, 1] if rewards[i] >= 0 else [1, 0, 1] # Blue for good, Magenta for bad
        p.addUserDebugText(f"P{i}", pos, textColorRGB=color)
        p.addUserDebugLine(pos, pos + [0, 0, 0.01], color, lineWidth=2)
        
    p.addUserDebugText("TARGET", target, textColorRGB=[1, 0, 0], textSize=1.5)
    p.addUserDebugLine(target - [0.02,0,0], target + [0.02,0,0], [1, 0, 0], lineWidth=3)
    p.addUserDebugLine(target - [0,0.02,0], target + [0,0.02,0], [1, 0, 0], lineWidth=3)
    
    # 4. Test 'cover' task (Attraction)
    print("\n--- Testing 'cover' task (Should pull towards target) ---")
    result_cover = compute_field_guided_action(
        cloth_positions=cloth_pos,
        reward_field=rewards,
        target_center=target,
        task_type='cover',
        step_size=0.1,
        debug_mode=True
    )
    
    if result_cover:
        pick, place, force = result_cover
        print(f"Pick Pos: {pick}")
        print(f"Place Pos: {place}")
        print(f"Force Vector: {force}")
        
        # Verification: Force should point roughly towards +X (target is at +0.2)
        if force[0] > 0:
            print("SUCCESS: Force points towards target.")
        else:
            print("FAILURE: Force does not point towards target.")
    else:
        print("No action found.")

    # 5. Test 'uncover' task (Repulsion)
    # Note: In a real scenario, we wouldn't run both visualizations on top of each other immediately,
    # but for this test we will just print the result. To visualize 'uncover', we can clear lines or just add it.
    # Let's add a small offset to the text or just run it.
    
    print("\n--- Testing 'uncover' task (Should push away from target) ---")
    # We will use a different debug color or just let it overlap for now, 
    # but let's pause to let the user see the first one.
    print("Visualizing 'cover' task. Sleeping for 2 seconds...")
    time.sleep(2)
    p.removeAllUserDebugItems()
    
    # Redraw context
    for i, pos in enumerate(cloth_pos):
        color = [0, 0, 1] if rewards[i] >= 0 else [1, 0, 1]
        p.addUserDebugText(f"P{i}", pos, textColorRGB=color)
    p.addUserDebugText("TARGET", target, textColorRGB=[1, 0, 0], textSize=1.5)

    result_uncover = compute_field_guided_action(
        cloth_positions=cloth_pos,
        reward_field=rewards,
        target_center=target,
        task_type='uncover',
        step_size=0.1,
        debug_mode=True
    )
    
    if result_uncover:
        pick, place, force = result_uncover
        print(f"Pick Pos: {pick}")
        print(f"Place Pos: {place}")
        print(f"Force Vector: {force}")
        
        # Verification: Force should point roughly towards -X (away from target at +0.2)
        if force[0] < 0:
            print("SUCCESS: Force points away from target.")
        else:
            print("FAILURE: Force does not point away from target.")
    else:
        print("No action found.")
        
    print("\nVisualization active. Press Ctrl+C to exit.")
    while True:
        p.stepSimulation()
        time.sleep(0.1)
