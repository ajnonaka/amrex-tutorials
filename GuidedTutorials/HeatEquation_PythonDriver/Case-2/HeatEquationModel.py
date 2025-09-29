#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional, Union
import amrex.space3d as amr


def load_cupy():
    """Load appropriate array library (CuPy for GPU, NumPy for CPU)."""
    if amr.Config.have_gpu:
        try:
            import cupy as cp
            amr.Print("Note: found and will use cupy")
            return cp
        except ImportError:
            amr.Print("Warning: GPU found but cupy not available! Using numpy...")
            import numpy as np
            return np
        if amr.Config.gpu_backend == "SYCL":
            amr.Print("Warning: SYCL GPU backend not yet implemented for Python")
            import numpy as np
            return np
    else:
        import numpy as np
        amr.Print("Note: found and will use numpy")
        return np


def postprocess_function(multifab: amr.MultiFab, geom: amr.Geometry) -> np.ndarray:
    """
    Geometry-aware postprocessing function.
    
    Parameters:
    -----------
    multifab : amr.MultiFab
        Simulation data
    geom : amr.Geometry
        Domain geometry
        
    Returns:
    --------
    np.ndarray
        Processed outputs [max, mean, std, integral/sum, center_value]
    """
    xp = load_cupy()
    
    # Get basic statistics
    max_val = multifab.max(comp=0, local=False)
    sum_val = multifab.sum(comp=0, local=False)
    total_cells = multifab.box_array().numPts
    mean_val = sum_val / total_cells
    
    # Calculate standard deviation
    l2_norm = multifab.norm2(0)
    sum_sq = l2_norm**2
    variance = (sum_sq / total_cells) - mean_val**2
    std_val = np.sqrt(max(0, variance))
    
    # Get value at center
    dx = geom.data().CellSize()
    prob_lo = geom.data().ProbLo()
    prob_hi = geom.data().ProbHi()
    
    # Calculate center coordinates
    center_coords = [(prob_lo[i] + prob_hi[i]) / 2.0 for i in range(geom.Dim())]
    
    # Find the cell index closest to center
    center_indices = []
    for i in range(geom.Dim()):
        idx = int((center_coords[i] - prob_lo[i]) / dx[i])
        center_indices.append(idx)
    
    # Get value at center (default to 0 if can't access)
    center_val = 0.0
    try:
        for mfi in multifab:
            bx = mfi.validbox()
            if (center_indices[0] >= bx.small_end[0] and center_indices[0] <= bx.big_end[0] and
                center_indices[1] >= bx.small_end[1] and center_indices[1] <= bx.big_end[1] and
                center_indices[2] >= bx.small_end[2] and center_indices[2] <= bx.big_end[2]):
                
                state_arr = xp.array(multifab.array(mfi), copy=False)
                # Convert global to local indices
                local_i = center_indices[0] - bx.small_end[0]
                local_j = center_indices[1] - bx.small_end[1] 
                local_k = center_indices[2] - bx.small_end[2]
                center_val = float(state_arr[0, local_k, local_j, local_i])
                break
    except (IndexError, AttributeError):
        # Fall back to (0,0,0) if center calculation fails
        for mfi in multifab:
            state_arr = xp.array(multifab.array(mfi), copy=False)
            center_val = float(state_arr[0, 0, 0, 0])
            break
    
    return np.array([max_val, mean_val, std_val, sum_val, center_val])


class SimulationModel(ABC):
    """Base class defining the simulation interface."""

    def __init__(self, **kwargs):
        """Initialize AMReX if needed."""
        if not amr.initialized():
            amr.initialize([])

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """
        Run simulation for each parameter set.

        Parameters:
        -----------
        params : np.ndarray of shape (n_samples, n_params) or (n_params,)
            Parameter sets to run

        Returns:
        --------
        np.ndarray of shape (n_samples, n_outputs)
            Simulation outputs
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        n_samples = params.shape[0]
        n_outputs = len(self.get_outnames())
        outputs = np.zeros((n_samples, n_outputs))

        for i in range(n_samples):
            try:
                multifab, varnames, geom = self.evolve(params[i, :])
                outputs[i, :] = self.postprocess(multifab, varnames, geom)
            except Exception as e:
                amr.Print(f"Warning: Simulation failed for parameter set {i}: {e}")
                outputs[i, :] = np.nan

        return outputs

    @abstractmethod
    def evolve(self, param_set: np.ndarray) -> Tuple[amr.MultiFab, Optional[amr.Vector_string], Optional[amr.Geometry]]:
        """
        Run a single simulation step/evolution.
        
        Parameters:
        -----------
        param_set : np.ndarray
            Single parameter set
            
        Returns:
        --------
        tuple : (multifab, varnames, geom)
            - multifab: simulation data
            - varnames: variable names or None
            - geom: domain geometry or None
        """
        pass

    def postprocess(self, multifab: amr.MultiFab, varnames: Optional[amr.Vector_string], 
                   geom: Optional[amr.Geometry]) -> np.ndarray:
        """
        Postprocessing function with geometry awareness.
        
        Parameters:
        -----------
        multifab : amr.MultiFab
            Simulation data
        varnames : amr.Vector_string or None
            Variable names
        geom : amr.Geometry or None
            Domain geometry
            
        Returns:
        --------
        np.ndarray
            Processed outputs [max, mean, std, integral/sum, center_value]
        """
        if geom is not None:
            return postprocess_function(multifab, geom)
        else:
            return self._postprocess_no_geom(multifab)

    def _postprocess_no_geom(self, multifab: amr.MultiFab) -> np.ndarray:
        """
        Postprocessing when geometry is not available.
        Uses (0,0,0,0) cell instead of center.
        """
        xp = load_cupy()
        
        # Get value at (0,0,0,0) cell
        cell_val = 0.0
        try:
            for mfi in multifab:
                state_arr = xp.array(multifab.array(mfi), copy=False)
                cell_val = float(state_arr[0, 0, 0, 0])  # component 0, cell (0,0,0)
                break
        except (IndexError, AttributeError):
            amr.Print("Warning: Could not access cell (0,0,0,0)")

        # Compute statistics
        max_val = multifab.max(comp=0, local=False)
        sum_val = multifab.sum(comp=0, local=False)
        total_cells = multifab.box_array().numPts
        mean_val = sum_val / total_cells
        
        l2_norm = multifab.norm2(0)
        sum_sq = l2_norm**2
        variance = (sum_sq / total_cells) - mean_val**2
        std_val = np.sqrt(max(0, variance))

        return np.array([max_val, mean_val, std_val, sum_val, cell_val])

    @abstractmethod
    def get_pnames(self) -> List[str]:
        """Get parameter names."""
        pass

    @abstractmethod
    def get_outnames(self) -> List[str]:
        """Get output names."""
        pass


class HeatEquationModel(SimulationModel):
    """Heat equation simulation model."""

    def __init__(self, n_cell: int = 32, max_grid_size: int = 16, nsteps: int = 1000, 
                 plot_int: int = 100, dt: float = 1e-5, use_parmparse: bool = False):
        super().__init__()
        
        if use_parmparse:
            from main import parse_inputs  # Import here to avoid circular imports
            params = parse_inputs()
            self.n_cell = params['n_cell']
            self.max_grid_size = params['max_grid_size']
            self.nsteps = params['nsteps']
            self.plot_int = params['plot_int']
            self.dt = params['dt']
        else:
            self.n_cell = n_cell
            self.max_grid_size = max_grid_size
            self.nsteps = nsteps
            self.plot_int = plot_int
            self.dt = dt

    def evolve(self, param_set: np.ndarray) -> Tuple[amr.MultiFab, amr.Vector_string, amr.Geometry]:
        """
        Run heat equation simulation.
        
        Parameters:
        -----------
        param_set : np.ndarray
            [diffusion_coeff, init_amplitude, init_width]
        
        Returns:
        --------
        tuple : (phi_new, varnames, geom)
            Ready to pass to write_single_level_plotfile
        """
        from main import main  # Import here to avoid circular imports
        
        if len(param_set) != 3:
            raise ValueError(f"Expected 3 parameters, got {len(param_set)}")
        
        phi_new, geom = main(
            diffusion_coeff=float(param_set[0]),
            init_amplitude=float(param_set[1]),
            init_width=float(param_set[2]),
            n_cell=self.n_cell,
            max_grid_size=self.max_grid_size,
            nsteps=self.nsteps,
            plot_int=self.plot_int,
            dt=self.dt,
            plot_files_output=False,
            verbose=0
        )
        
        varnames = amr.Vector_string(['phi'])
        return phi_new, varnames, geom

    def get_pnames(self) -> List[str]:
        return ["diffusion_coefficient", "initial_amplitude", "initial_width"]

    def get_outnames(self) -> List[str]:
        return ["max", "mean", "std", "integral", "center"]
