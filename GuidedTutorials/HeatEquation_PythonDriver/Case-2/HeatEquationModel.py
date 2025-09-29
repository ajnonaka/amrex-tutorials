#!/bin/python3
import numpy as np
from typing import Tuple, List, Optional
import amrex.space3d as amr
from BaseModel import SimulationModel, load_cupy


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

    def postprocess(self, multifab: amr.MultiFab, varnames: Optional[amr.Vector_string], 
                   geom: Optional[amr.Geometry]) -> np.ndarray:
        """
        Heat equation specific postprocessing with geometry awareness.
        
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
        
        # Get value at center (if geometry available)
        center_val = 0.0
        if geom is not None:
            dx = geom.data().CellSize()
            prob_lo = geom.data().ProbLo()
            prob_hi = geom.data().ProbHi()
            
            # Calculate center coordinates
            center_coords = [(prob_lo[i] + prob_hi[i]) / 2.0 for i in range(3)]
            
            # Find the cell index closest to center
            center_indices = []
            for i in range(3):
                idx = int((center_coords[i] - prob_lo[i]) / dx[i])
                center_indices.append(idx)
            
            # Get value at center (default to 0 if can't access)
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
        else:
            # No geometry available, use (0,0,0,0) cell
            try:
                for mfi in multifab:
                    state_arr = xp.array(multifab.array(mfi), copy=False)
                    center_val = float(state_arr[0, 0, 0, 0])
                    break
            except (IndexError, AttributeError):
                amr.Print("Warning: Could not access cell (0,0,0,0)")
        
        return np.array([max_val, mean_val, std_val, sum_val, center_val])

    def get_pnames(self) -> List[str]:
        return ["diffusion_coefficient", "initial_amplitude", "initial_width"]

    def get_outnames(self) -> List[str]:
        return ["max", "mean", "std", "integral", "center"]
