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
        ('param', 'diffusion_coeff'),
        ('param', 'init_amplitude'),
        ('param', 'init_width'),
    ]

    # Define output fields
    _output_fields = [
        ('output', 'max_temp'),
        ('output', 'mean_temp'),
        ('output', 'std_temp'),
        ('output', 'total_energy'),
    ]

    # Spatial domain bounds (3D heat equation domain)
    _spatial_domain_bounds = [
        np.array([0.0, 0.0, 0.0]),    # left edge
        np.array([1.0, 1.0, 1.0]),    # right edge
        np.array([32, 32, 32])         # default grid dimensions
    ]

    # Enable subprocess mode
    _use_subprocess = True
    _model_script = './model.x'

    def _get_field_info(self, field_tuple):
        """
        Provide field metadata with statistical properties.
        Only includes fields that are actually used by the framework.
        """
        field_type, field_name = field_tuple

        field_info_dict = {
            # Parameters - match param_margpc.txt values
            ('param', 'diffusion_coeff'): {
                'mean': 1.0,
                'std': 0.25,
                'bounds': [0.1, 5.0],
                'distribution': 'normal',
                'units': 'm**2/s',
                'display_name': 'Thermal Diffusivity',
            },
            ('param', 'init_amplitude'): {
                'mean': 1.0,
                'std': 0.25,
                'bounds': [0.1, 3.0],
                'distribution': 'normal',
                'units': 'K',
                'display_name': 'Initial Temperature',
            },
            ('param', 'init_width'): {
                'mean': 0.01,
                'std': 0.0025,
                'bounds': [0.001, 0.1],
                'distribution': 'normal',
                'units': 'm',
                'display_name': 'Initial Width',
            },

            # Outputs - just units and display names
            ('output', 'max_temp'): {
                'units': 'K',
                'display_name': 'Maximum Temperature',
            },
            ('output', 'mean_temp'): {
                'units': 'K',
                'display_name': 'Mean Temperature',
            },
            ('output', 'std_temp'): {
                'units': 'K',
                'display_name': 'Temperature Std Dev',
            },
            ('output', 'total_energy'): {
                'units': 'J',
                'display_name': 'Total Energy',
            },
        }

        return field_info_dict.get(field_tuple, {})
