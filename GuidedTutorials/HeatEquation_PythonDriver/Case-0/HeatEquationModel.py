#!/usr/bin/env python3
import numpy as np
from typing import Tuple, List, Optional
import amrex.space3d as amr
from AMReXBaseModel import AMReXBaseModel, load_cupy


class HeatEquationModel(AMReXBaseModel):
    """
    Heat equation simulation model with full statistical properties
    and AMReX integration.
    """

    # Define parameter fields with proper naming convention
    _param_fields = [
        ('param', 'diffusion_coefficient'),
        ('param', 'initial_amplitude'),
        ('param', 'initial_width'),
    ]

    # Define output fields
    _output_fields = [
        ('output', 'max_temperature'),
        ('output', 'mean_temperature'),
        ('output', 'std_temperature'),
        ('output', 'total_energy'),
    ]

    # Spatial domain bounds (3D heat equation domain)
    _spatial_domain_bounds = [
        np.array([0.0, 0.0, 0.0]),    # left edge
        np.array([1.0, 1.0, 1.0]),    # right edge
        np.array([32, 32, 32])         # default grid dimensions
    ]

    def __init__(self, n_cell: int = 32, max_grid_size: int = 16,
                 nsteps: int = 1000, plot_int: int = 100,
                 dt: float = 1e-5, use_parmparse: bool = False, **kwargs):
        """
        Initialize heat equation model.

        Parameters:
        -----------
        n_cell : int
            Number of grid cells in each dimension
        max_grid_size : int
            Maximum grid size for AMR
        nsteps : int
            Number of time steps
        plot_int : int
            Plot interval (for output control)
        dt : float
            Time step size
        use_parmparse : bool
            Whether to use AMReX ParmParse for input
        """
        # Store simulation parameters
        if use_parmparse:
            from main import parse_inputs
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

        # Update spatial domain dimensions based on n_cell
        if self.n_cell != 32:
            self._spatial_domain_bounds[2] = np.array([self.n_cell, self.n_cell, self.n_cell])

        # Initialize base class
        super().__init__(**kwargs)

    def _get_field_info(self, field_tuple):
        """
        Provide field metadata with statistical properties.
        Only includes fields that are actually used by the framework.
        """
        field_type, field_name = field_tuple

        field_info_dict = {
            # Parameters - match param_margpc.txt values
            ('param', 'diffusion_coefficient'): {
                'mean': 1.0,
                'std': 0.25,
                'bounds': [0.1, 5.0],
                'distribution': 'normal',
                'units': 'm**2/s',
                'display_name': 'Thermal Diffusivity',
            },
            ('param', 'initial_amplitude'): {
                'mean': 1.0,
                'std': 0.25,
                'bounds': [0.1, 3.0],
                'distribution': 'normal',
                'units': 'K',
                'display_name': 'Initial Temperature',
            },
            ('param', 'initial_width'): {
                'mean': 0.01,
                'std': 0.0025,
                'bounds': [0.001, 0.1],
                'distribution': 'normal',
                'units': 'm',
                'display_name': 'Initial Width',
            },

            # Outputs - just units and display names
            ('output', 'max_temperature'): {
                'units': 'K',
                'display_name': 'Maximum Temperature',
            },
            ('output', 'mean_temperature'): {
                'units': 'K',
                'display_name': 'Mean Temperature',
            },
            ('output', 'std_temperature'): {
                'units': 'K',
                'display_name': 'Temperature Std Dev',
            },
            ('output', 'total_energy'): {
                'units': 'J',
                'display_name': 'Total Energy',
            },
        }

        return field_info_dict.get(field_tuple, {})

    def evolve(self, param_set: np.ndarray):
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

    def postprocess(self, sim_state) -> np.ndarray:
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
            Processed outputs [max, mean, std, integral/sum]
        """
        xp = load_cupy()
        [multifab, varnames, geom] = sim_state

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

        return np.array([
            float(max_val),
            float(mean_val),
            float(std_val),
            float(sum_val)
        ])

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
