#!/usr/bin/env python3
import numpy as np
from typing import Tuple, Optional
import amrex.space3d as amr
from AMReXModelBase import AMReXModelBase, load_cupy


class HeatEquationModel(AMReXModelBase):
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
        ('output', 'center_temperature'),
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

        # Store physical constants
        self.thermal_conductivity = 1.0  # W/(m·K)
        self.specific_heat = 1.0         # J/(kg·K)
        self.density = 1.0                # kg/m³

    def _get_field_info(self, field_tuple):
        """
        Provide detailed field metadata with statistical properties.
        """
        field_type, field_name = field_tuple

        field_info_dict = {
            ('param', 'diffusion_coefficient'): {
                'units': 'm**2/s',
                'display_name': 'Thermal Diffusivity',
                'description': 'α = k/(ρ·cp) - thermal diffusivity coefficient',
                'mean': 1.0,
                'std': 0.3,
                'bounds': [0.1, 5.0],
                'distribution': 'lognormal',  # Physical parameter, must be positive
                'log_scale': True,
                'default': 1.0,
                'physical_range': [1e-7, 1e-3],  # Realistic range for materials (m²/s)
                'material_examples': {
                    'air': 2.2e-5,
                    'water': 1.4e-7,
                    'steel': 1.2e-5,
                    'copper': 1.1e-4,
                }
            },
            ('param', 'initial_amplitude'): {
                'units': 'K',
                'display_name': 'Initial Temperature Amplitude',
                'description': 'Peak temperature of initial Gaussian distribution',
                'mean': 1.0,
                'std': 0.2,
                'bounds': [0.1, 3.0],
                'distribution': 'normal',
                'log_scale': False,
                'default': 1.0,
                'normalized': True,  # Values are normalized to reference temperature
            },
            ('param', 'initial_width'): {
                'units': 'm',
                'display_name': 'Initial Distribution Width',
                'description': 'Standard deviation of initial Gaussian temperature field',
                'mean': 0.01,
                'std': 0.003,
                'bounds': [0.001, 0.1],
                'distribution': 'lognormal',  # Width must be positive
                'log_scale': True,
                'default': 0.01,
                'relative_to_domain': True,  # As fraction of domain size
            },
            ('output', 'max_temperature'): {
                'units': 'K',
                'display_name': 'Maximum Temperature',
                'description': 'Maximum temperature in the domain',
                'valid_range': [0.0, np.inf],
                'expected_range': [0.1, 3.0],  # Typical range for normalized runs
                'monotonic': 'decreasing',  # Decreases with time due to diffusion
            },
            ('output', 'mean_temperature'): {
                'units': 'K',
                'display_name': 'Mean Temperature',
                'description': 'Spatial average of temperature field',
                'valid_range': [0.0, np.inf],
                'expected_range': [0.0, 1.0],
                'conserved': True,  # Should be conserved in isolated system
            },
            ('output', 'std_temperature'): {
                'units': 'K',
                'display_name': 'Temperature Standard Deviation',
                'description': 'Spatial standard deviation of temperature',
                'valid_range': [0.0, np.inf],
                'expected_range': [0.0, 1.0],
                'monotonic': 'decreasing',  # Decreases as field homogenizes
            },
            ('output', 'total_energy'): {
                'units': 'J',
                'display_name': 'Total Thermal Energy',
                'description': 'Integrated thermal energy in domain',
                'valid_range': [0.0, np.inf],
                'conserved': True,  # Conserved quantity
                'scaling': 'extensive',  # Scales with system size
            },
            ('output', 'center_temperature'): {
                'units': 'K',
                'display_name': 'Center Temperature',
                'description': 'Temperature at domain center',
                'valid_range': [0.0, np.inf],
                'expected_range': [0.0, 3.0],
                'probe_location': [0.5, 0.5, 0.5],  # Normalized coordinates
            },
        }

        return field_info_dict.get(field_tuple, {})

    def evolve(self, param_set: np.ndarray) -> Tuple[amr.MultiFab, amr.Vector_string, amr.Geometry]:
        """
        Run heat equation simulation with given parameters.

        Parameters:
        -----------
        param_set : np.ndarray
            [diffusion_coefficient, initial_amplitude, initial_width]

        Returns:
        --------
        tuple : (phi_new, varnames, geom)
            MultiFab with solution, variable names, and geometry
        """
        from main import main  # Import simulation driver

        if len(param_set) != len(self.param_names):
            raise ValueError(f"Expected {len(self.param_names)} parameters, got {len(param_set)}")

        # Extract parameters with proper names
        diffusion_coeff = float(param_set[0])
        init_amplitude = float(param_set[1])
        init_width = float(param_set[2])

        # Validate parameters against bounds
        for i, (param_val, param_name) in enumerate(zip(param_set, self.param_names)):
            info = self.get_param_info(param_name)
            if 'bounds' in info:
                if param_val < info['bounds'][0] or param_val > info['bounds'][1]:
                    print(f"Warning: {param_name}={param_val} outside bounds {info['bounds']}")

        # Run simulation
        phi_new, geom = main(
            diffusion_coeff=diffusion_coeff,
            init_amplitude=init_amplitude,
            init_width=init_width,
            n_cell=self.n_cell,
            max_grid_size=self.max_grid_size,
            nsteps=self.nsteps,
            plot_int=self.plot_int,
            dt=self.dt,
            plot_files_output=False,
            verbose=0
        )

        varnames = amr.Vector_string(['temperature'])
        return phi_new, varnames, geom

    def postprocess(self, multifab: amr.MultiFab,
                   varnames: Optional[amr.Vector_string] = None,
                   geom: Optional[amr.Geometry] = None) -> np.ndarray:
        """
        Extract output quantities from simulation results.

        Parameters:
        -----------
        multifab : amr.MultiFab
            Simulation data (temperature field)
        varnames : amr.Vector_string or None
            Variable names
        geom : amr.Geometry or None
            Domain geometry

        Returns:
        --------
        np.ndarray
            Output quantities [max, mean, std, total_energy, center_value]
        """
        xp = load_cupy()

        # Get basic statistics
        max_val = multifab.max(comp=0, local=False)
        sum_val = multifab.sum(comp=0, local=False)
        total_cells = multifab.box_array().numPts
        mean_val = sum_val / total_cells

        # Calculate standard deviation using L2 norm
        l2_norm = multifab.norm2(0)
        sum_sq = l2_norm**2
        variance = (sum_sq / total_cells) - mean_val**2
        std_val = np.sqrt(max(0, variance))

        # Calculate total energy (normalized by cell volume if geometry available)
        if geom is not None:
            dx = geom.data().CellSize()
            cell_volume = dx[0] * dx[1] * dx[2]
            total_energy = sum_val * cell_volume * self.density * self.specific_heat
        else:
            total_energy = sum_val  # Dimensionless if no geometry

        # Get temperature at domain center
        center_val = 0.0
        if geom is not None:
            dx = geom.data().CellSize()
            prob_lo = geom.data().ProbLo()
            prob_hi = geom.data().ProbHi()

            # Calculate center coordinates
            center_coords = [(prob_lo[i] + prob_hi[i]) / 2.0 for i in range(3)]

            # Find the cell index closest to center
            center_indices = [int((center_coords[i] - prob_lo[i]) / dx[i]) for i in range(3)]

            # Get value at center
            try:
                for mfi in multifab:
                    bx = mfi.validbox()
                    if all(center_indices[i] >= bx.small_end[i] and
                           center_indices[i] <= bx.big_end[i] for i in range(3)):

                        state_arr = xp.array(multifab.array(mfi), copy=False)
                        # Convert global to local indices
                        local_indices = [center_indices[i] - bx.small_end[i] for i in range(3)]
                        center_val = float(state_arr[0, local_indices[2],
                                                     local_indices[1], local_indices[0]])
                        break
            except (IndexError, AttributeError) as e:
                print(f"Warning: Could not access center cell: {e}")
                center_val = mean_val  # Fall back to mean
        else:
            center_val = mean_val  # No geometry, use mean as proxy

        return np.array([max_val, mean_val, std_val, total_energy, center_val])

    def validate_outputs(self, outputs: np.ndarray) -> bool:
        """
        Validate that outputs are physically reasonable.

        Parameters:
        -----------
        outputs : np.ndarray
            Output values to validate

        Returns:
        --------
        bool
            True if outputs are valid
        """
        if outputs.shape[-1] != len(self.output_names):
            return False

        # Check each output against its valid range
        for i, output_name in enumerate(self.output_names):
            info = self.get_output_info(output_name)
            if 'valid_range' in info:
                min_val, max_val = info['valid_range']
                if outputs[..., i].min() < min_val or outputs[..., i].max() > max_val:
                    print(f"Warning: {output_name} outside valid range {info['valid_range']}")
                    return False

        # Physical consistency checks
        max_temp = outputs[..., 0]
        mean_temp = outputs[..., 1]

        # Maximum should be >= mean
        if np.any(max_temp < mean_temp):
            print("Warning: Maximum temperature less than mean")
            return False

        return True


if __name__ == "__main__":
    # Initialize AMReX
    amr.initialize([])

    # Create model with default parameters
    model = HeatEquationModel(n_cell=32, nsteps=100, dt=1e-5)

    print("Heat Equation Model Configuration:")
    print("=" * 50)
    print(f"Parameter fields: {model.param_names}")
    print(f"Output fields: {model.output_names}")
    print(f"Domain: {model.domain_left_edge} to {model.domain_right_edge}")
    print(f"Grid: {model.domain_dimensions}")

    # Show statistical properties
    print("\nParameter Statistics:")
    print("-" * 50)
    for i, param_name in enumerate(model.param_names):
        info = model.get_param_info(param_name)
        print(f"{param_name}:")
        print(f"  Mean: {model.modelpar['mean'][i]:.3f} {info.get('units', '')}")
        print(f"  Std:  {model.modelpar['std'][i]:.3f}")
        print(f"  Bounds: {info.get('bounds', 'Not specified')}")
        print(f"  Distribution: {info.get('distribution', 'normal')}")

    # Test with different parameter sets
    test_params = np.array([
        [1.0, 1.0, 0.01],   # baseline
        [2.0, 1.5, 0.02],   # higher diffusion, higher amplitude
        [0.5, 2.0, 0.005]   # lower diffusion, higher amplitude, narrower
    ])

    print("\nRunning simulations...")
    print("-" * 50)
    outputs = model(test_params)

    print("\nResults:")
    print("-" * 50)
    for i, params in enumerate(test_params):
        print(f"Run {i+1}: D={params[0]:.2f}, A={params[1]:.2f}, W={params[2]:.3f}")
        for j, output_name in enumerate(model.output_names):
            info = model.get_output_info(output_name)
            units = info.get('units', '')
            print(f"  {info.get('display_name', output_name)}: {outputs[i,j]:.4f} {units}")

    # Validate outputs
    if model.validate_outputs(outputs):
        print("\n✓ All outputs are physically valid")
    else:
        print("\n✗ Some outputs failed validation")

    # Write parameter marginals for UQ analysis
    model.write_param_marginals('heat_equation_marginals.txt')

    # Finalize AMReX
    amr.finalize()
