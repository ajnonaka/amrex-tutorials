.. _guided_pytuq_integration:

.. _pytuq_quickstart:

PyTUQ
=====

.. admonition:: **Time to Complete**: 15-20 minutes
   :class: note

   **What you will learn**:
      - Install PyTUQ
      - Run AMReX + PyTUQ examples
      - Deploy on Perlmutter

Motivation
----------

AMReX simulations deliver high-fidelity, accurate results for complex physics problems, but parameter studies and uncertainty quantification can require hundreds or thousands of runs—making comprehensive analysis computationally prohibitive.

This tutorial demonstrates how to improve efficiency without sacrificing accuracy by using polynomial chaos expansions to build fast surrogate models that identify which input parameters truly matter.

PyTUQ (Python interface to the UQ Toolkit) provides specialized tools for surrogate construction and global sensitivity analysis, enabling rapid parameter space exploration and dimensionality reduction for scientific applications.

We demonstrate how to integrate PyTUQ with your AMReX application through three practical workflows: C++ executables with file-based I/O (Case-1), Python-wrapped C++ codes (Case-2), and native PyAMReX applications (Case-3).

Located in ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ``, this example analyzes a heat equation solver to illustrate the complete forward UQ workflow from parameter sampling through sensitivity analysis.

Installation
------------

.. code-block:: bash
   :caption: Quick install

   git clone --recursive --branch v1.0.0z https://github.com/sandialabs/pytuq
   cd pytuq
   echo "dill" >> requirements.txt
   pip install -r requirements.txt
   pip install .

Examples
--------

C++ AMReX + PyTUQ (BASH driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Build and Run

   **Prerequisites**: AMReX compiled with MPI support

   .. code-block:: bash
      :caption: Build C++ example

      cd GuidedTutorials/HeatEquation_UQ/Case-1
      make -j4

   .. code-block:: bash
      :caption: Run with bash script

      ./wf_uqpc.x

C++ AMReX + PyTUQ (python driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Build and Run

   **Prerequisites**: AMReX compiled with MPI support

   .. code-block:: bash
      :caption: Build C++ example

      cd GuidedTutorials/HeatEquation_UQ/Case-2
      make -j4

   .. note::

      **Adapting the PyTUQ example script:**

      Copy the PyTUQ example script and replace the Ishigami test function with the HeatEquationModel wrapper. This allows PyTUQ to drive the C++ executable through Python subprocess calls.

   .. code-block:: bash
      :caption: Copy PyTUQ example script

      cp ../../../../pytuq/examples/ex_pcgsa.py .

   **Modify ex_pcgsa.py:**

   Add the import near the top of the file:

   .. code-block:: python

      from ModelXBaseModel import HeatEquationModel

   Replace the model definition:

   .. code-block:: python
      :caption: Original (around line 23)

      myfunc = Ishigami()

   .. code-block:: python
      :caption: Replacement

      myfunc = HeatEquationModel()
      myfunc._request_out_fields = [('output', 'max_temp')]

   .. code-block:: bash
      :caption: Run UQ analysis

      python ./ex_pcgsa.py

PyAMReX + PyTUQ
~~~~~~~~~~~~~~~

.. dropdown:: Setup and Run

   **Prerequisites**: pyAMReX installed

   .. code-block:: bash
      :caption: Navigate to Case-3 directory

      cd GuidedTutorials/HeatEquation_UQ/Case-3

   .. note::

      **Using PyAMReX with PyTUQ:**

      For native PyAMReX applications, copy the PyTUQ example script and replace the Ishigami function with the PyAMReX-based HeatEquationModel. This enables direct Python-to-Python integration without subprocess overhead.

   .. code-block:: bash
      :caption: Copy PyTUQ example script

      cp ../../../../pytuq/examples/ex_pcgsa.py .

   **Modify ex_pcgsa.py:**

   Add the import near the top of the file:

   .. code-block:: python

      from HeatEquationModel import HeatEquationModel

   Replace the model definition:

   .. code-block:: python
      :caption: Original (around line 23)

      myfunc = Ishigami()

   .. code-block:: python
      :caption: Replacement

      myfunc = HeatEquationModel()
      myfunc._request_out_fields = [('output', 'max_temp')]

   .. code-block:: bash
      :caption: Run UQ analysis

      python ./ex_pcgsa.py

Perlmutter Deployment
---------------------

.. note::

   Module setup required: ``module load python cuda``

C++ AMReX on Perlmutter
~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Perlmutter Setup

   .. code-block:: bash
      :caption: Virtual environment setup

      module load conda
      # For NERSC (see https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware):
      # conda create -y --prefix /global/common/software/myproject/$USER/pytuq_integration python=3.11
      conda create -y --name pytuq_integration python=3.11
      git clone --recursive --branch v1.0.0z https://github.com/sandialabs/pytuq
      cd pytuq
      echo "dill" >> requirements.txt
      pip install -r requirements.txt
      pip install .

   .. code-block:: bash
      :caption: perlmutter_build.sh

      module load PrgEnv-gnu cudatoolkit
      make USE_MPI=TRUE USE_CUDA=TRUE

   .. code-block:: bash
      :caption: Submit job

      # If stored in common software: conda activate /global/common/software/myproject/$USER/pytuq_integration
      conda activate pytuq_integration
      sbatch wk_uqpc.slurm

   .. note::

       For NERSC, consider placing your conda environment in ``/global/common/software``
       for better performance and persistence. See the `NERSC Python documentation
       <https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware>`_
       for details.

Mathematical Details
--------------------

Parameter Sensitivity via Polynomial Chaos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Extending this tutorial to other applications
---------------------------------------------

Identify / Add input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Implementation Steps:**

1. Add physics parameters to main.cpp declarations:

   .. code-block:: cpp

      amrex::Real diffusion_coeff;
      amrex::Real init_amplitude;
      amrex::Real init_width;

2. Read parameters from inputs file:

   .. code-block:: cpp

      diffusion_coeff = 1.0;
      pp.query("diffusion_coeff", diffusion_coeff);

      init_amplitude = 1.0;
      pp.query("init_amplitude", init_amplitude);

      init_width = 0.01;
      pp.query("init_width", init_width);

3. Use parameters in initial conditions and evolution:

   .. code-block:: cpp

      // Initial conditions
      amrex::Real rsquared = ((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5)) / init_width;
      phiOld(i,j,k) = 1.0 + init_amplitude * std::exp(-rsquared);

      // Evolution
      phiNew(i,j,k) = phiOld(i,j,k) + dt * diffusion_coeff * laplacian;

The simplest output extraction method is described here:

Output example: C++ Datalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: When to use
   :class: tip

   Choose when you have centralized output locations and want simple I/O.

**Implementation Steps:**

1. Add datalog configuration to C++ code:

   .. code-block:: cpp

      const int datwidth = 14;
      const int datprecision = 6;
      int datalog_int = -1;
      bool datalog_final = true;
      pp.query("datalog_int", datalog_int);

2. Write statistics to datalog.txt:

   .. code-block:: cpp

      // Check if we should write datalog
      bool write_datalog = false;
      if (datalog_final && step == nsteps) {
          write_datalog = true;  // Write final step
      } else if (datalog_int > 0 && step % datalog_int == 0) {
          write_datalog = true;  // Write every datalog_int steps
      }

      if (write_datalog && amrex::ParallelDescriptor::IOProcessor()) {
          std::ofstream datalog("datalog.txt", std::ios::app);

          amrex::Real mean_temp = phi_new.sum(0) / phi_new.boxArray().numPts();
          amrex::Real max_temperature = phi_new.max(0);
          amrex::Real variance = phi_new.norm2(0) / phi_new.boxArray().numPts() - mean_temp * mean_temp;
          amrex::Real std_temperature = (variance > 0.0) ? std::sqrt(variance) : 0.0;

          datalog << time << " " << max_temperature << " " << std_temperature << " " << step << std::endl;
      }

   .. note::

      Some AMReX codes already have datalog capabilities:

      - For existing datalogs, you may only need to ``tail -n 1 datalog`` for the final values
      - For AMR codes, use the built-in DataLog:

        .. code-block:: cpp

           // Assuming 'amr' is your Amr object
           if (amrex::ParallelDescriptor::IOProcessor()) {
               amr.DataLog(0) << "# time max_temperature mean_temp" << std::endl;
               amr.DataLog(0) << time << " " << max_temperature << " " << mean_temp << std::endl;
           }

Extending to PyAMReX Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For PyAMReX applications, adapt your existing ``main.py`` to enable UQ parameter sweeps:

**1. Modify main() function signature**

Add UQ parameters as function arguments:

.. code-block:: python
   :caption: Original

   def main(n_cell, max_grid_size, nsteps, plot_int, dt):

.. code-block:: python
   :caption: With UQ parameters

   def main(n_cell: int = 32, max_grid_size: int = 16, nsteps: int = 100,
            plot_int: int = 100, dt: float = 1e-5, plot_files_output: bool = False,
            verbose: int = 1, diffusion_coeff: float = 1.0, init_amplitude: float = 1.0,
            init_width: float = 0.01) -> Tuple[amr.MultiFab, amr.Geometry]:

**2. Use parameters in physics**

Replace hardcoded values with function parameters:

.. code-block:: python
   :caption: Original initial condition (main.py:117-118)

   rsquared = ((x[:, xp.newaxis, xp.newaxis] - 0.5)**2
             + (y[xp.newaxis, :, xp.newaxis] - 0.5)**2
             + (x[xp.newaxis, xp.newaxis, :] - 0.5)**2) / 0.01
   phiOld[:, ngz:-ngz, ngy:-ngy, ngx:-ngx] = 1. + xp.exp(-rsquared)

.. code-block:: python
   :caption: Parameterized

   rsquared = ((x[:, xp.newaxis, xp.newaxis] - 0.5)**2
             + (y[xp.newaxis, :, xp.newaxis] - 0.5)**2
             + (x[xp.newaxis, xp.newaxis, :] - 0.5)**2) / init_width
   phiOld[:, ngz:-ngz, ngy:-ngy, ngx:-ngx] = 1. + init_amplitude * xp.exp(-rsquared)

.. code-block:: python
   :caption: Original evolution (main.py:143-144)

   phiNew[:, ngz:-ngz,ngy:-ngy,ngx:-ngx] = (
       phiOld[:, ngz:-ngz,ngy:-ngy,ngx:-ngx]
       + dt*((phiOld[:, ngz:-ngz, ngy:-ngy, ngx+1:hix-ngx+1] - ... ) / dx[0]**2))

.. code-block:: python
   :caption: With diffusion coefficient

   phiNew[:, ngz:-ngz,ngy:-ngy,ngx:-ngx] = (
       phiOld[:, ngz:-ngz,ngy:-ngy,ngx:-ngx]
       + dt * diffusion_coeff * ((phiOld[:, ngz:-ngz, ngy:-ngy, ngx+1:hix-ngx+1] - ... ) / dx[0]**2))

**3. Return simulation state**

Modify the main function to return MultiFab and Geometry:

.. code-block:: python

   return phi_new, geom

**4. Create PyTUQ model wrapper**

Create a ``HeatEquationModel.py`` that inherits from ``AMReXBaseModel``:

.. code-block:: python

   from AMReXBaseModel import AMReXBaseModel
   import amrex.space3d as amr
   import numpy as np

   class HeatEquationModel(AMReXBaseModel):
       _param_fields = [
           ('param', 'diffusion_coeff'),
           ('param', 'init_amplitude'),
           ('param', 'init_width'),
       ]

       _output_fields = [
           ('output', 'max_temp'),
           ('output', 'mean_temp'),
           ('output', 'std_temp'),
           ('output', 'total_energy'),
       ]

       def evolve(self, param_set: np.ndarray):
           """Run simulation with given parameters."""
           from main import main
           phi_new, geom = main(
               diffusion_coeff=float(param_set[0]),
               init_amplitude=float(param_set[1]),
               init_width=float(param_set[2]),
               plot_files_output=False,
               verbose=0
           )
           varnames = amr.Vector_string(['phi'])
           return phi_new, varnames, geom

       def postprocess(self, sim_state) -> np.ndarray:
           """Extract quantities of interest."""
           multifab, varnames, geom = sim_state
           max_val = multifab.max(comp=0, local=False)
           sum_val = multifab.sum(comp=0, local=False)
           mean_val = sum_val / multifab.box_array().numPts
           # Calculate std dev...
           return np.array([max_val, mean_val, std_val, sum_val])

See ``Case-3/HeatEquationModel.py`` for the complete implementation.

Summary
-------

**Key Takeaways:**

* PyTUQ can use models with ``outputs = model(inputs)`` interface
* C++ codes can use wrapper scripts or gnu parallel; Python codes integrate directly
* Best practices for Perlmutter requires python environment modules

Additional Resources
--------------------

- `PyTUQ Documentation <https://sandialabs.github.io/pytuq>`_
- `PyTUQ Examples directory <https://github.com/sandialabs/pytuq/tree/main/examples>`_
- `AMReX Documentation <https://amrex-codes.github.io/amrex/docs_html/>`_
- `pyAMReX Documentation <https://pyamrex.readthedocs.io>`_
- :ref:`_guided_heat` - Base tutorial this builds upon

.. seealso::

   For complete working examples of the ``outputs = model(inputs)`` pattern, see:

   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-1/`` - C++ executable and python scripts called from a bash workflow script
   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-2/`` - C++ executable driven by python wrapping bash
   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-3/`` - PyAMReX native
