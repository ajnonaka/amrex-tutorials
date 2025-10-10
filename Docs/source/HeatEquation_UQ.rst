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

C++ AMReX + PyTUQ (BASH driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Build and Run
   :class: dropdown

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

.. admonition:: Build and Run
   :class: dropdown

   **Prerequisites**: AMReX compiled with MPI support

   .. code-block:: bash
      :caption: Build C++ example

      cd GuidedTutorials/HeatEquation_UQ/Case-2
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
      :caption: Use Case-2 directory

      cd GuidedTutorials/HeatEquation_UQ/Case-3

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

      module load PrgEnv-gnu cudatoolkit
      make USE_MPI=TRUE USE_CUDA=TRUE

   .. code-block:: bash
      :caption: Submit job

      sbatch wk_uqpc.slurm

.. PyAMReX on Perlmutter
.. ~~~~~~~~~~~~~~~~~~~~~

..
 admonition:: Perlmutter Setup
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
