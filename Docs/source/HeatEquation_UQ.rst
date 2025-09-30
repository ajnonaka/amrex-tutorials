.. _guided_pytuq_integration:

PyTUQ Integration with AMReX Applications
==========================================

.. admonition:: **Time to Complete**: 30-45 minutes
   :class: note

   **Prerequisites**:
      - Basic knowledge of AMReX build system
      - Familiarity with Python/NumPy
      - Understanding of uncertainty quantification concepts
   
   **What you will learn**:
      - How to wrap AMReX simulations with PyTUQ's generic model interface
      - Converting simulation codes to ``outputs = model(inputs)`` pattern
      - Processing simulation outputs for UQ analysis

Goals
-----

This tutorial demonstrates how to integrate PyTUQ (Python Uncertainty Quantification Toolkit) with AMReX-based applications. PyTUQ expects models to follow a simple interface:

.. code-block:: python

   # Generic PyTUQ model interface
   outputs = model(inputs)  # Both are numpy arrays
   # inputs shape: [n_samples, n_parameters]
   # outputs shape: [n_samples, n_outputs]

You will learn to:

1. Wrap AMReX simulations to provide this interface
2. Extract and format simulation outputs as numpy arrays
3. Choose the appropriate integration approach for your workflow
4. Run sensitivity analysis and inverse modeling using PyTUQ

Prerequisites and Setup
-----------------------

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

Install the following components in order:

.. code-block:: bash
   :caption: Complete installation script

   #!/bin/bash
   
   # For NERSC: module load conda
   
   # 1. Clone repositories
   git clone --recursive --branch v1.0.0z https://github.com/sandialabs/pytuq
   git clone --branch 25.10 https://github.com/amrex-codes/pyamrex
   git clone --branch 25.10 https://github.com/amrex-codes/amrex
   git clone --branch development https://github.com/amrex-codes/amrex-tutorials
   # Alternative: git clone --branch add_pybind_interface_test https://github.com/jmsexton03/amrex-tutorials
   
   # 2. Setup conda environment
   # Create conda environment (use -y for non-interactive)
   conda create -y --name pyamrex_pytuq python=3.11 --no-default-packages
   
   # For NERSC (see https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware):
   # conda create -y --prefix /global/common/software/myproject/$USER/pyamrex_pytuq python=3.11
   
   conda activate pyamrex_pytuq
   # For NERSC: conda activate /global/common/software/myproject/$USER/pyamrex_pytuq
   
   # 3. Build and install pyAMReX (developer install)
   cd pyamrex
   
   # Set environment variable for AMReX source
   export AMREX_SRC=$PWD/../amrex
   
   # Optional: Set compilers explicitly
   # export CC=$(which clang)
   # export CXX=$(which clang++)
   # For GPU support:
   # export CUDACXX=$(which nvcc)
   # export CUDAHOSTCXX=$(which clang++)
   
   # Install Python requirements
   python3 -m pip install -U -r requirements.txt
   python3 -m pip install -v --force-reinstall --no-deps .
   
   # Build with cmake (includes all dimensions)
   cmake -S . -B build -DAMReX_SPACEDIM="1;2;3" -DpyAMReX_amrex_src=$(pwd)/../amrex
   cmake --build build --target pip_install -j 8
   
   cd ../
   
   # 4. Install PyTUQ
   cd pytuq
   python -m pip install -r requirements.txt
   python -m pip install .
   conda install -y dill
   cd ../
   
   # 5. Setup workflow files (optional - for Case 1 examples)
   # mkdir rundir
   # cd rundir
   # tar -xf ~/workflow_uqpc.tar  # Obtain from PyTUQ examples
   # cd ../
   
   # 6. Verify installation
   conda list | grep pyamrex  # Should show pyamrex 25.10
   conda list | grep pytuq    # Should show pytuq 1.0.0z

.. note::

   For NERSC users, consider placing your conda environment in ``/global/common/software`` 
   for better performance and persistence. See the `NERSC Python documentation 
   <https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware>`_ 
   for details.

.. warning::

   Ensure version compatibility:
   
   - AMReX: 25.10
   - pyAMReX: 25.10 
   - PyTUQ: v1.0.0z
   - amrex-tutorials: development branch (or jmsexton03/add_pybind_interface_test for testing)

PyTUQ Model Interface Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTUQ supports various model types, all following the same interface pattern:

.. code-block:: python
   :caption: Example 1: Linear model from PyTUQ tests

   def linear_model(x, par):
       y = par['b'] + x @ par['W']
       return y
   
   # Usage with PyTUQ
   W = np.random.randn(pdim, npt)
   b = np.random.randn(npt)
   true_model_params = {'W': W, 'b': b}
   
   # Model wrapper for PyTUQ
   model = lambda x: linear_model(x, true_model_params)
   outputs = model(inputs)  # inputs: [n_samples, pdim]

.. code-block:: python
   :caption: Example 2: Simple function model

   # Direct lambda function
   model = lambda x: x[:,0]**4 - 2.*x[:,0]**3
   
   # Or as a class method
   class Ishigami:
       def __call__(self, x):
           # Ishigami function implementation
           return y
   
   model = Ishigami()
   outputs = model(inputs)

.. code-block:: python
   :caption: Example 3: External executable wrapper

   def model(x):
       # Write inputs to file
       np.savetxt('inputs.txt', x)
       
       # Run external simulation
       os.system('./model.x inputs.txt outputs.txt')
       
       # Load and return outputs
       y = np.loadtxt('outputs.txt').reshape(x.shape[0], -1)
       return y

Reference PyTUQ Workflow Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTUQ provides several workflow examples demonstrating the model interface:

.. list-table::
   :header-rows: 1

   * - Analysis Type
     - Example File
     - Model Interface Pattern
   * - Global Sensitivity
     - `ex_pcgsa.py <https://github.com/pytuq/examples/ex_pcgsa.py>`_
     - ``y = model(x)`` with polynomial chaos
   * - Inverse Modeling
     - `ex_mcmc_fitmodel.py <https://github.com/pytuq/examples/ex_mcmc_fitmodel.py>`_
     - ``y_pred = model(params)`` for likelihood evaluation
   * - Gaussian Process
     - `ex_gp.py <https://github.com/pytuq/examples/ex_gp.py>`_
     - Surrogate: ``y_approx = gp.predict(x)``
   * - Linear Regression
     - `ex_lreg_merr.py <https://github.com/pytuq/examples/ex_lreg_merr.py>`_
     - ``y = X @ beta + error``

Input/Output Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTUQ expects data as NumPy arrays with specific shapes:

.. code-block:: python

   # Standard interface for all models
   def model(inputs):
       """
       Args:
           inputs: np.ndarray of shape [n_samples, n_parameters]
       
       Returns:
           outputs: np.ndarray of shape [n_samples, n_outputs]
       """
       return outputs

**Heat Equation Input Parameters** (from AMReX inputs file):

.. code-block:: text
   :caption: Example inputs file with UQ parameters

   # Grid/Domain parameters
   n_cell = 32                 # number of cells on each side of domain
   max_grid_size = 16          # size of each box (or grid)
   
   # Time stepping parameters
   nsteps = 100                # total steps in simulation
   dt = 1.e-5                  # time step
   
   # Output control
   plot_int = -1               # how often to write plotfile (-1 = no plots)
   datalog_int = -1            # how often to write datalog (-1 = no regular output)
   
   # Physics parameters (these are what we vary for UQ)
   diffusion_coeff = 1.0       # diffusion coefficient for heat equation
   init_amplitude = 1.0        # amplitude of initial temperature profile
   init_width = 0.01           # width parameter (variance, not std dev)

**Heat Equation Output Format** (datalog):

.. code-block:: text
   :caption: Datalog output format

   #         time      max_temp      std_temp    final_step
             0.01       1.09628             0          1000

For UQ analysis, we typically vary the physics parameters (``diffusion_coeff``, ``init_amplitude``, ``init_width``) 
and extract the outputs (``max_temp``, ``std_temp``) at the final timestep.

.. warning::

   Ensure your outputs are well-correlated with the input parameters being varied. 
   Outputs unaffected by input changes will produce meaningless UQ results.

Choosing Your Integration Approach
-----------------------------------

All approaches ultimately provide the same ``outputs = model(inputs)`` interface:

.. code-block:: text

   Start: Need outputs = model(inputs) interface
   │
   ├─ Using Python already?
   │  ├─ Yes → Have pyAMReX?
   │  │  ├─ Yes → Case 2: Direct Python model
   │  │  └─ Using PICMI/WarpX? → Case 4: PICMI wrapper
   │  └─ No → Continue to C++ options
   │
   └─ C++ Application (need wrapper)
      ├─ Centralized outputs and recompling? → Case 1a: Datalog wrapper
      ├─ Bash configuration with no C++ code changes? → Case 1b: Bash wrapper
      ├─ Fextract or fcompare workflows? → Case 1c: Fextract wrapper
      └─ Want Python bindings? → Case 3: Pybind11 wrapper

Integration Cases
-----------------

Case 1: C++ Application Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three approaches create a Python wrapper providing the ``model(inputs)`` interface.

Common C++ Modifications for HeatEquation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All C++ HeatEquation cases (1a, 1b, 1c) share the same parameter modifications:

**Physics Parametrization:**

.. code-block:: cpp
   :caption: Add physics parameters (in main.cpp declarations)
   
   // diffusion coefficient for heat equation
   amrex::Real diffusion_coeff;
   
   // amplitude of initial temperature profile
   amrex::Real init_amplitude;
   
   // width parameter controlling spread of initial profile (variance, not std dev)
   amrex::Real init_width;

.. code-block:: cpp
   :caption: Read parameters from inputs file (in ParmParse section)
   
   diffusion_coeff = 1.0;
   pp.query("diffusion_coeff", diffusion_coeff);
   
   init_amplitude = 1.0;
   pp.query("init_amplitude", init_amplitude);
   
   init_width = 0.01;
   pp.query("init_width", init_width);

.. code-block:: cpp
   :caption: Use parameters in initial conditions
   
   amrex::Real rsquared = ((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5)) / init_width;
   phiOld(i,j,k) = 1.0 + init_amplitude * std::exp(-rsquared);

.. code-block:: cpp
   :caption: Use diffusion coefficient in evolution
   
   phiNew(i,j,k) = phiOld(i,j,k) + dt * diffusion_coeff * laplacian;

The cases differ only in how they extract outputs:

Case 1a: C++ with Datalog Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Additional Modifications for Datalog:**

.. code-block:: cpp
   :caption: Add datalog configuration
   
   const int datwidth = 14;
   const int datprecision = 6;
   int datalog_int = -1;
   bool datalog_final = true;

.. code-block:: cpp
   :caption: Write statistics to datalog.txt
   
   if (write_datalog && amrex::ParallelDescriptor::IOProcessor()) {
       std::ofstream datalog("datalog.txt", std::ios::app);
       
       amrex::Real mean_temp = phi_new.sum(0) / phi_new.boxArray().numPts();
       amrex::Real max_temperature = phi_new.max(0);
       amrex::Real variance = phi_new.norm2(0) / phi_new.boxArray().numPts() - mean_temp * mean_temp;
       amrex::Real std_temperature = (variance > 0.0) ? std::sqrt(variance) : 0.0;
       
       datalog << time << " " << max_temperature << " " << std_temperature << " " << step << std::endl;
   }

Case 1b: C++ with Plotfile/Bash Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses the same parametrized C++ code but with ``plot_int > 0`` to generate plotfiles, then extracts data via bash scripts.

Case 1c: C++ with Fextract Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses the same parametrized C++ code with plotfiles, but extracts data using AMReX fextract utilities.

Case 2: PyAMReX Direct Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: When to use
   :class: tip

   For Python-based AMReX applications - provides native model interface.

**Implementation:**

.. code-block:: python
   :caption: HeatEquationModel.py

   import numpy as np
   import pyamrex.amrex as amrex
   from BaseModel import BaseModel
   
   class HeatEquationModel(BaseModel):
       """PyAMReX model with PyTUQ interface"""
       
       def __init__(self, geom, default_params):
           super().__init__(geom)
           self.default_params = default_params
           
       def __call__(self, inputs):
           """
           Direct PyTUQ interface - no wrapper needed
           
           Args:
               inputs: np.ndarray [n_samples, n_params]
           
           Returns:
               outputs: np.ndarray [n_samples, n_outputs]
           """
           n_samples = inputs.shape[0]
           outputs = np.zeros((n_samples, 2))
           
           for i, params in enumerate(inputs):
               # Set parameters
               self.thermal_conductivity = params[0]
               self.heat_source = params[1]
               self.initial_temp = params[2]
               
               # Initialize and run
               self.initialize()
               self.advance(n_steps=100)
               
               # Extract outputs directly
               outputs[i, 0] = self.get_max_temperature()
               outputs[i, 1] = self.get_avg_temperature()
               
               # Reset for next run
               self.reset()
           
           return outputs
   
   # Direct usage with PyTUQ
   model = HeatEquationModel(geom, default_params)
   # model now provides the required interface

Case 3: C++ with Pybind11 Bindings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: When to use
   :class: tip

   When you want compiled performance with Python interface.

**Binding Implementation:**

.. code-block:: cpp
   :caption: bindings.cpp

   #include <pybind11/pybind11.h>
   #include <pybind11/numpy.h>
   #include "HeatEquation.H"
   
   namespace py = pybind11;
   
   py::array_t<double> heat_equation_model(py::array_t<double> inputs) {
       // Get input dimensions
       auto buf = inputs.request();
       double* in_ptr = static_cast<double*>(buf.ptr);
       size_t n_samples = buf.shape[0];
       size_t n_params = buf.shape[1];
       
       // Prepare output array
       py::array_t<double> outputs({n_samples, 2});
       auto out_buf = outputs.request();
       double* out_ptr = static_cast<double*>(out_buf.ptr);
       
       // Run simulations
       for (size_t i = 0; i < n_samples; ++i) {
           double thermal_cond = in_ptr[i * n_params + 0];
           double heat_source = in_ptr[i * n_params + 1];
           double init_temp = in_ptr[i * n_params + 2];
           
           HeatEquation sim(thermal_cond, heat_source, init_temp);
           sim.run();
           
           out_ptr[i * 2 + 0] = sim.get_max_temperature();
           out_ptr[i * 2 + 1] = sim.get_avg_temperature();
       }
       
       return outputs;
   }
   
   PYBIND11_MODULE(amrex_uq, m) {
       m.def("model", &heat_equation_model, 
             "Heat equation model with PyTUQ interface");
   }

**Python usage:**

.. code-block:: python

   import amrex_uq
   
   # Direct PyTUQ interface from C++
   model = amrex_uq.model
   outputs = model(inputs)  # Standard interface

Case 4: PICMI/WarpX Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: warpx_model.py

   from pywarpx import picmi
   import numpy as np
   
   def warpx_model(inputs):
       """
       WarpX model with PyTUQ interface
       
       Args:
           inputs: [n_samples, n_params]
                  params = [E_max, plasma_density, ...]
       """
       outputs = np.zeros((inputs.shape[0], 3))
       
       for i, params in enumerate(inputs):
           # Create simulation with parameters
           sim = picmi.Simulation(
               E_max=params[0],
               n_plasma=params[1],
               # ... other parameters
           )
           
           # Run simulation
           sim.step(max_steps=1000)
           
           # Extract outputs
           outputs[i, 0] = sim.get_field_energy()[-1]
           outputs[i, 1] = sim.get_particle_energy()[-1]
           outputs[i, 2] = sim.get_momentum_spread()[-1]
       
       return outputs
   
   # Ready for PyTUQ
   model = warpx_model

Validation and Testing
----------------------

Validate your model interface:

.. code-block:: python
   :caption: test_model_interface.py

   import numpy as np
   from model import heat_equation_model  # Your model
   
   # Test with random inputs
   n_samples = 10
   n_params = 3
   test_inputs = np.random.rand(n_samples, n_params)
   
   # Rescale to parameter ranges
   param_ranges = np.array([[0.5, 1.5], [1.0, 10.0], [273, 373]])
   for i in range(n_params):
       test_inputs[:, i] = (param_ranges[i, 1] - param_ranges[i, 0]) * \
                           test_inputs[:, i] + param_ranges[i, 0]
   
   # Test model interface
   outputs = heat_equation_model(test_inputs)
   
   # Validate output shape
   assert outputs.shape[0] == n_samples, "Wrong number of samples"
   assert outputs.ndim == 2, "Outputs should be 2D array"
   
   # Check outputs are reasonable
   assert np.all(np.isfinite(outputs)), "Outputs contain NaN or Inf"
   assert np.all(outputs > 0), "Temperature should be positive"
   
   print(f"✓ Model interface validated")
   print(f"  Input shape: {test_inputs.shape}")
   print(f"  Output shape: {outputs.shape}")
   print(f"  Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")

Running Complete UQ Workflows
------------------------------

Example workflow combining model with PyTUQ analysis:

.. code-block:: python
   :caption: complete_uq_workflow.py

   import numpy as np
   import pytuq
   from model import heat_equation_model
   
   # 1. Define parameter distributions
   param_dists = [
       pytuq.UniformDist(0.5, 1.5),    # thermal_conductivity
       pytuq.UniformDist(1.0, 10.0),   # heat_source
       pytuq.UniformDist(273, 373)     # initial_temperature
   ]
   
   # 2. Global Sensitivity Analysis (like ex_pcgsa.py)
   pce = pytuq.PCE(model=heat_equation_model,
                   distributions=param_dists,
                   order=3)
   pce.fit(n_samples=100)
   sobol_indices = pce.get_sobol_indices()
   
   # 3. Inverse Modeling (like ex_mcmc_fitmodel.py)
   observed_data = np.array([350.0, 325.0])  # Observed [max_temp, avg_temp]
   
   def likelihood(params):
       pred = heat_equation_model(params.reshape(1, -1))[0]
       return -0.5 * np.sum((pred - observed_data)**2 / sigma**2)
   
   mcmc = pytuq.MCMC(likelihood, param_dists)
   samples = mcmc.sample(n_samples=1000)
   
   # 4. Surrogate Modeling (like ex_gp.py)
   gp = pytuq.GaussianProcess(model=heat_equation_model,
                              param_ranges=param_ranges)
   gp.fit(n_samples=50)
   
   # Fast predictions with surrogate
   test_points = np.random.rand(1000, 3)
   fast_predictions = gp.predict(test_points)

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Issue
     - Solution
   * - Model returns wrong shape
     - Ensure output is ``[n_samples, n_outputs]``, use ``reshape(-1, n_outputs)``
   * - Model crashes on certain parameters
     - Add parameter validation and bounds checking in wrapper
   * - Slow execution for many samples
     - Consider parallel execution or surrogate models
   * - Outputs uncorrelated with inputs
     - Verify parameters are actually being used in simulation
   * - Memory issues with large runs
     - Process in batches, clean up temporary files

Additional Resources
--------------------

- `PyTUQ Documentation <https://pytuq.readthedocs.io>`_
- `PyTUQ Examples Repository <https://github.com/pytuq/examples>`_
- `AMReX Documentation <https://amrex-codes.github.io/amrex/docs_html/>`_
- `pyAMReX Documentation <https://pyamrex.readthedocs.io>`_
- :ref:`guided_heat_equation` - Base tutorial this builds upon

.. seealso::

   For complete working examples of the ``outputs = model(inputs)`` pattern, see:
   
   - ``amrex-tutorials/GuidedTutorials/UQ/Case1a/`` - Datalog wrapper
   - ``amrex-tutorials/GuidedTutorials/UQ/Case2/`` - PyAMReX native
   - ``amrex-tutorials/GuidedTutorials/UQ/Case3/`` - Pybind11 wrapper
