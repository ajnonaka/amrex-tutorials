.. _guided_pytuq_integration:

.. _pytuq_quickstart:

PyTUQ Quick Start Guide
=======================

.. admonition:: **Time to Complete**: 15-20 minutes
   :class: note

   **What you will learn**:
      - Install PyTUQ
      - Run AMReX + PyTUQ examples
      - Deploy on Perlmutter

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

C++ AMReX + PyTUQ
~~~~~~~~~~~~~~~~~

.. admonition:: Build and Run
   :class: dropdown

   **Prerequisites**: AMReX compiled with MPI support

   .. code-block:: bash
      :caption: Build C++ example

      cd GuidedTutorials/HeatEquation_UQ/Case-1
      make -j4

   .. code-block:: bash
      :caption: Copy ex_pcgsa.py and use the black box model

      cp ../../../../pytuq/examples/ex_pcgsa.py
      # replace Ishigami() with minimal python model

      16d15
      < from ModelXBaseModel import HeatEquationModel
      23,24c22
      < myfunc = HeatEquationModel()
      < myfunc._request_out_fields = [('output', 'max_temp')]
      ---
      > myfunc = Ishigami()

   .. code-block:: bash
      :caption: Run with Python wrapped bash

      python ./ex_pcgsa.py

PyAMReX + PyTUQ
~~~~~~~~~~~~~~~

.. admonition:: Setup and Run
   :class: dropdown

   **Prerequisites**: pyAMReX installed

    .. code-block:: bash
       :caption: Direct Python integration

       cp ../../../../pytuq/examples/ex_pcgsa.py
       # replace Ishigami() with pyamrex model

       16d15
       <       from HeatEquationModel import HeatEquationModel
       23,24c22
       < myfunc = HeatEquationModel()
       < myfunc._request_out_fields = [('output', 'max_temp')]
       ---
       > myfunc = Ishigami()

   .. code-block:: bash
      :caption: Run

      python ./ex_pcgsa.py

Perlmutter Deployment (not implemented)
---------------------------------------

.. note::

   Module setup required: ``module load python cuda``

C++ AMReX on Perlmutter
~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Perlmutter Setup
   :class: dropdown

   .. code-block:: bash
      :caption: perlmutter_build.sh

      module load PrgEnv-gnu cuda
      make USE_MPI=TRUE USE_CUDA=TRUE

   .. code-block:: bash
      :caption: Submit job

      sbatch run_amrex_uq.slurm

PyAMReX on Perlmutter
~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Perlmutter Setup
   :class: dropdown

   .. code-block:: bash
      :caption: Virtual environment setup

      module load conda
      # For NERSC (see https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware):
      conda create -y --prefix /global/common/software/myproject/$USER/pytuq_integration python=3.11


   .. code-block:: bash
      :caption: Run PyAMReX + PyTUQ

      srun -n 4 python run_pyamrex_uq.py

   .. note::

       For NERSC users, consider placing your conda environment in ``/global/common/software``
       for better performance and persistence. See the `NERSC Python documentation
       <https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware>`_
       for details.

Summary
-------

**Key Takeaways:**

* PyTUQ can use models with ``outputs = model(inputs)`` interface
* C++ codes need wrapper scripts; Python codes integrate directly
* Best practices for Perlmutter requires environment modules and MPI configuration

Additional Resources
--------------------

- `PyTUQ Documentation <https://pytuq.readthedocs.io>`_
- `PyTUQ Examples directory <https://github.com/sandialabs/pytuq/tree/main/examples>`_
- `AMReX Documentation <https://amrex-codes.github.io/amrex/docs_html/>`_
- `pyAMReX Documentation <https://pyamrex.readthedocs.io>`_
- :ref:`_guided_heat` - Base tutorial this builds upon

.. seealso::

   For complete working examples of the ``outputs = model(inputs)`` pattern, see:

   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-1/`` - C++ wrappers
   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-2/`` - PyAMReX native

PyTUQ Integration with AMReX Applications (First draft)
=======================================================

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

This tutorial demonstrates how to integrate PyTUQ (Python Uncertainty Quantification Toolkit) with AMReX-based applications.

You will learn to:

1. Configure wrappers for AMReX simulations to interface to PyTUQ
2. Use inputs and extract datalog outputs
3. Choose the appropriate integration approach for your workflow
4. Run sensitivity analysis and inverse modeling using PyTUQ

Prerequisites and Setup
-----------------------

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

Install pytuq as described in `pytuq/README.md <https://github.com/sandialabs/pytuq/blob/main/README.md>`_:

.. code-block:: bash
   :caption: Pytuq installation script

   #!/bin/bash

   # For NERSC: module load conda

   # 1. Clone repositories
   git clone --recursive --branch v1.0.0z https://github.com/sandialabs/pytuq

   # 2. Setup conda environment (optional, you can add to an existing env)
   # Create conda environment (use -y for non-interactive)
   conda create -y --name pytuq_integration python=3.11 --no-default-packages

   # For NERSC (see https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware):
   # conda create -y --prefix /global/common/software/myproject/$USER/pytuq_integration python=3.11

   conda activate pytuq_integration
   # For NERSC: conda activate /global/common/software/myproject/$USER/pytuq_integration

   # 3. Install PyTUQ
   cd pytuq
   python -m pip install -r requirements.txt
   python -m pip install .
   conda install -y dill
   cd ../

   # 4. Verify installation
   conda list | grep pytuq    # Should show pytuq 1.0.0z

For a full install including this tutorial, amrex, pyamrex, and pytuq see `example_detailed_install.sh <../../../GuidedTutorials/HeatEquation_UQ/example_detailed_install.sh>`_

.. note::

   For NERSC users, consider placing your conda environment in ``/global/common/software``
   for better performance and persistence. See the `NERSC Python documentation
   <https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware>`_
   for details.

Reference PyTUQ Workflow Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTUQ provides several workflow examples demonstrating the model interface:

.. list-table::
   :header-rows: 1

   * - Analysis Type
     - Example File
     - Model Interface Pattern
   * - Global Sensitivity
     - `ex_pcgsa.py <https://github.com/sandialabs/pytuq/blob/main/examples/ex_pcgsa.py>`_
     - ``myfunc = Ishigami(); ysam = myfunc(xsam)`` with polynomial chaos
   * - Inverse Modeling
     - `ex_mcmc_fitmodel.py <https://github.com/sandialabs/pytuq/blob/main/examples/ex_mcmc_fitmodel.py>`_
     - ``true_model, true_model_params = linear_model, {'W': W, 'b': b}; yd = true_model(true_model_input, true_model_params)`` for likelihood evaluation
   * - Gaussian Process
     - `ex_gp.py <https://github.com/sandialabs/pytuq/blob/main/examples/ex_gp.py>`_
     - Surrogate: ``true_model = sin4; y = true_model(x)+datastd*np.random.randn(ntrn)``
   * - Linear Regression
     - `ex_lreg_merr.py <https://github.com/sandialabs/pytuq/blob/main/examples/ex_lreg_merr.py>`_
     - ``true_model = lambda x: x[:,0]**4 - 2.*x[:,0]**3 #fcb.sin4; y = true_model(x)``

Integration Cases
-----------------

Case 1: C++ Application Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. note::

   The key to PyTUQ integration is creating a loop that maps input parameters to output quantities.
   This loop structure differs between cases:

   - **Case 1**: Bash workflow manages run directories, runs executable with input parameters, parses outputs
   - **Case 2**: Python loop directly calls pyAMReX functions

The simplest output extraction method is described here:

Case 1: C++ with Datalog Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

3. Configure bash wrapper (``model.x``):

   .. code-block:: bash
      :caption: Configuration section of model.x

      # Configuration
      EXE="main3d.gnu.ex"
      INPUTS="inputs"
      INCLUDE_HEADER=true
      POSTPROCESSOR="./postprocess_datalog.sh"

   The ``model.x`` script (based on PyTUQ workflow examples) handles:

   - Reading input parameters from file
   - Setting up AMReX inputs command line options with parameters
   - Running the executable
   - Calling postprocessor to extract outputs
   - Writing outputs to file

4. Set up files like pnames.txt
   diffusion_coeff
   init_amplitude
   init_width

5. Use the pytuq infrastructure for setting up the inputs, e.g.
   .. code-block:: bash
      :caption: Configuration for input PC coefficients

       ## (a) Given mean and standard deviation of each normal random parameter
       echo "1 0.1 " > param_margpc.txt
       echo "1 0.1" >> param_margpc.txt
       echo ".01 0.0025" >> param_margpc.txt
       PC_TYPE=HG # Hermite-Gaussian PC
       INPC_ORDER=1

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
   from AMReXBaseModel import AMReXBaseModel

   #Define inherited class with __call__ and outnames and pnames methods
   class HeatEquationModel(AMReXBaseModel)

   # Direct usage with PyTUQ
   model = HeatEquationModel()
   # model now provides the required interface

Running Complete UQ Workflows
------------------------------

Example workflow combining model with PyTUQ analysis:
[add example from pytuq examples]

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

   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-1/`` - C++ wrappers
   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-2/`` - PyAMReX native
