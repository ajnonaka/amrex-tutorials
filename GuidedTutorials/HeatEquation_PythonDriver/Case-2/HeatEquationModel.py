#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import amrex.space3d as amr
import numpy as np


def postprocess_function(state_new, geom):
    # Get the real center of the domain from geometry
    prob_lo = geom.data().ProbLo()
    prob_hi = geom.data().ProbHi()

    # Calculate the actual center coordinates
    center_x = 0.5 * (prob_lo[0] + prob_hi[0])
    center_y = 0.5 * (prob_lo[1] + prob_hi[1])
    center_z = 0.5 * (prob_lo[2] + prob_hi[2])

    # Now use these real center coordinates
    dx = geom.data().CellSize()

    # Convert to index coordinates
    i_center = int((center_x - prob_lo[0]) / dx[0])
    j_center = int((center_y - prob_lo[1]) / dx[1]) 
    k_center = int((center_z - prob_lo[2]) / dx[2])

    center_iv = amr.IntVect3D(i_center, j_center, k_center)

    center_val = None
    for mfi in state_new:
        bx = mfi.validbox()
        if bx.contains(center_iv):
            state_arr = xp.array(state_new.array(mfi), copy=False)
            local_i = i_center - bx.small_end[0] + ngx
            local_j = j_center - bx.small_end[1] + ngy
            local_k = k_center - bx.small_end[2] + ngz
            center_val = float(state_arr[0, local_k, local_j, local_i])
            if xp.__name__ == 'cupy':
                center_val = float(center_val)
                break

    if center_val is None:
        center_val = 0.0d

    # Compute output metrics from final state using PyAMReX built-ins
    max_val = state_new.max(comp=0, local=False)
    sum_val = state_new.sum(comp=0, local=False)

    # Get total number of valid cells (excluding ghost zones)
    total_cells = state_new.box_array().numPts
    mean_val = sum_val / total_cells

    # Use L2 norm for standard deviation calculation
    l2_norm = state_new.norm2(0)
    sum_sq = l2_norm**2
    variance = (sum_sq / total_cells) - mean_val**2
    std_val = np.sqrt(max(0, variance))

    integral = sum_val * dx[0] * dx[1] * dx[2]

    return np.array([
        max_val,
        mean_val,
        std_val,
        integral,
        center_val
    ])

class SimulationModel:
    """Simple wrapper to make pyamrex simulations callable with parameter arrays."""

    def __init__(self, n_cell=32, max_grid_size=16, nsteps=1000, plot_int=100, dt=1e-5, use_parmparse=False):
        # Conditionally initialize AMReX
        if not amr.initialized():
            amr.initialize([])

    def __call__(self, params):
        """
        Run simulation for each parameter set.

        Parameters:
        -----------
        params : numpy.ndarray of shape (n_samples, n_params)
            (Use get_pnames() to get these names programmatically)

        Returns:
        --------
        numpy.ndarray of shape (n_samples, n_outputs)
            (Use get_outnames() to get these names programmatically)
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        n_samples = params.shape[0]
        outputs = np.zeros((n_samples, 5))

        for i in range(n_samples):
            result = step()
            outputs[i, :] = postprocess_function(results)
            )

        return outputs

    def get_pnames(self):
        """
        Get parameter names for the heat equation model.

        Returns:
        --------
        list : Parameter names corresponding to the input dimensions
        """
        return []

    def get_outnames(self):
        """
        Get output names for the heat equation model.

        Returns:
        --------
        list : Output names corresponding to the computed quantities
        """
        return []

if __name__ == "__main__":

    # Initialize AMReX
    amr.initialize([])

    # Create model using ParmParse to read from inputs file
    model = HeatEquationModel(use_parmparse=True)

    print(f"Heat equation model initialized with:")
    print(f"  n_cell = {model.n_cell}")
    print(f"  max_grid_size = {model.max_grid_size}")
    print(f"  nsteps = {model.nsteps}")
    print(f"  plot_int = {model.plot_int}")
    print(f"  dt = {model.dt}")

    # Test with random parameters
    test_params = np.array([
        [1.0, 1.0, 0.01],   # baseline
        [2.0, 1.5, 0.02],   # higher diffusion, higher amplitude
        [0.5, 2.0, 0.005]   # lower diffusion, higher amplitude, narrower
    ])

    print("\nRunning heat equation with parameters:")
    print("  [diffusion, amplitude, width]")
    print(test_params)

    outputs = model(test_params)

    print("\nResults [max, mean, std, integral, center]:")
    print(outputs)

    # Finalize AMReX
    amr.finalize()

