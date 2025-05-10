import numpy as np
import cv2

print(f"DEBUG: Starting execution of models/flow_regularization.py ({__file__})")

def regularize_flow_field(initial_flow, lambda_smoothness, num_iterations, step_size):
    """
    Regularizes an optical flow field using Tikhonov regularization (quadratic smoothness).
    Minimizes E(u) = 0.5 * ||u - u_initial||^2 + 0.5 * lambda_smoothness * ||∇u||^2
    (where ||∇u||^2 is related to -uΔu after integration by parts)
    using gradient descent. The gradient of this energy is (u - u_initial) - lambda_smoothness * Δu.

    Args:
        initial_flow (np.ndarray): The initial flow field (H, W, 2).
        lambda_smoothness (float): Weight for the smoothness term.
        num_iterations (int): Number of gradient descent iterations.
        step_size (float): Step size (learning rate) for gradient descent.

    Returns:
        np.ndarray: The regularized flow field, or None if initial_flow is None.
    """
    print(f"DEBUG: regularize_flow_field called with lambda={lambda_smoothness}, iterations={num_iterations}, step_size={step_size}")
    if initial_flow is None:
        print("DEBUG: initial_flow is None in regularize_flow_field. Returning None.")
        return None
    if not isinstance(initial_flow, np.ndarray) or initial_flow.ndim != 3 or initial_flow.shape[2] != 2:
        raise ValueError("Initial flow must be an HxWx2 NumPy array.")

    # Ensure flow is float64 for precision with Laplacian and updates
    current_flow_x = initial_flow[:, :, 0].astype(np.float64)
    current_flow_y = initial_flow[:, :, 1].astype(np.float64)

    # Keep a copy of the original initial flow for the data term
    initial_flow_x_const = initial_flow[:, :, 0].astype(np.float64)
    initial_flow_y_const = initial_flow[:, :, 1].astype(np.float64)

    for i in range(num_iterations):
        # Compute Laplacian for x and y components of the current flow
        # cv2.Laplacian computes src(x+1,y) + src(x-1,y) + src(x,y+1) + src(x,y-1) - 4*src(x,y)
        laplacian_flow_x = cv2.Laplacian(current_flow_x, cv2.CV_64F, ksize=1)
        laplacian_flow_y = cv2.Laplacian(current_flow_y, cv2.CV_64F, ksize=1)

        # Gradient of the energy function components:
        # Data term gradient: (current_flow_component - initial_flow_component)
        # Smoothness term gradient: -lambda_smoothness * laplacian_flow_component
        grad_x = (current_flow_x - initial_flow_x_const) - lambda_smoothness * laplacian_flow_x
        grad_y = (current_flow_y - initial_flow_y_const) - lambda_smoothness * laplacian_flow_y
        
        # Gradient descent update
        current_flow_x -= step_size * grad_x
        current_flow_y -= step_size * grad_y
        
        if (i + 1) % (num_iterations // 10 if num_iterations >= 10 else 1) == 0: # Optional: print progress
             print(f"DEBUG: Regularization iteration {i+1}/{num_iterations}")

    regularized_flow = np.stack((current_flow_x, current_flow_y), axis=-1)
    print("DEBUG: Flow regularization complete.")
    return regularized_flow.astype(initial_flow.dtype) # Cast back to original dtype

print("DEBUG: Successfully defined regularize_flow_field in models/flow_regularization.py (at end of file)")