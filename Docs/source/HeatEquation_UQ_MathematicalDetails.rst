.. _pytuq_mathematical_details:

Parameter Sensitivity via Polynomial Chaos
-------------------------------------------

In this example, PyTUQ constructs polynomial chaos expansions to analyze how uncertain physical parameters affect simulation outputs in a heat diffusion problem.

The Heat Equation
^^^^^^^^^^^^^^^^^

The governing equation for this tutorial is the heat diffusion equation:

.. math::

   \frac{\partial T}{\partial t} = D \nabla^2 T + S(x,y,z)

where:

- :math:`T` is the temperature field
- :math:`D` is the diffusion coefficient (``diffusion_coeff``)
- :math:`S(x,y,z)` is an optional source term (not used in this example)

The initial temperature profile is a Gaussian centered at (0.5, 0.5, 0.5):

.. math::

   T(x,y,z,t=0) = 1 + A \exp\left(-\frac{r^2}{w^2}\right)

where:

- :math:`A` is the initial amplitude (``init_amplitude``)
- :math:`w^2` is the initial width parameter (``init_width``)
- :math:`r^2 = (x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2`

Uncertain Input Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The three uncertain parameters in this analysis are:

1. **diffusion_coeff** (:math:`D`): Controls how fast heat spreads through the domain

   - Mean: 1.0 m²/s
   - Standard deviation: 0.25 m²/s
   - Range: [0.25, 1.75] m²/s

2. **init_amplitude** (:math:`A`): Peak temperature above baseline

   - Mean: 1.0 K
   - Standard deviation: 0.25 K
   - Range: [0.25, 1.75] K

3. **init_width** (:math:`w^2`): Controls spread of initial temperature profile

   - Mean: 0.01 m²
   - Standard deviation: 0.0025 m²
   - Range: [0.0025, 0.0175] m²

These parameters are specified in the AMReX inputs file and read using ``ParmParse::query()`` (see ``main.cpp`` lines 100-111).

Quantities of Interest (Outputs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simulation extracts four statistical quantities at the final timestep:

1. **max_temp**: Maximum temperature in the domain
2. **mean_temp**: Average temperature across all cells
3. **std_temp**: Standard deviation of temperature
4. **total_energy**: Sum of temperature values (proportional to total thermal energy)

These outputs are computed in ``main.cpp`` (lines 293-299) and written to the datalog file.

PyTUQ Workflow
^^^^^^^^^^^^^^

PyTUQ uses polynomial chaos expansion to construct a surrogate model:

1. **Parameter sampling**: Generate sample points in the 3D parameter space based on the specified distributions
2. **Forward simulations**: Run the AMReX heat equation solver for each parameter set
3. **Surrogate construction**: Fit polynomial chaos coefficients to map inputs → outputs
4. **Sensitivity analysis**: Compute Sobol indices to identify which parameters most influence each output

The connection is:

- **Inputs**: ParmParse parameters (``diffusion_coeff``, ``init_amplitude``, ``init_width``) specified in ``inputs`` file or command line
- **Outputs**: Quantities of interest extracted from datalog files or direct Python access to MultiFabs
