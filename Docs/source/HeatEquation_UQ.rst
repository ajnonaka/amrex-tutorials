.. _guided_pytuq_integration:

.. _pytuq_quickstart:

AMReX-pytuq
===========

.. admonition:: **Time to Complete**: 15-20 minutes
   :class: note

   **What you will learn**:
      - Install PyTUQ
      - Run AMReX + PyTUQ examples
      - Deploy on Perlmutter

Overview
--------

AMReX simulations deliver high-fidelity, accurate results for complex physics problems, but parameter studies and uncertainty quantification can require hundreds or thousands of runs—making comprehensive analysis computationally prohibitive.

This tutorial demonstrates how to improve efficiency without sacrificing accuracy by using polynomial chaos expansions to build fast surrogate models that identify which input parameters truly matter.

PyTUQ (Python interface to the UQ Toolkit) provides specialized tools for surrogate construction and global sensitivity analysis, enabling rapid parameter space exploration and dimensionality reduction for scientific applications.

We demonstrate how to integrate PyTUQ with your AMReX application through three practical workflows: C++ executables managed with gnu parallel (Case-1), Python-driven C++ executables with bash run management (Case-2), and native PyAMReX applications (Case-3).

Located in ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ``, this example analyzes a heat equation solver to illustrate the complete forward UQ workflow from parameter sampling through sensitivity analysis. After running the provided examples, the Customizing the Workflow section explains the underlying problem setup and provides step-by-step guidance for adapting this workflow to your own AMReX application

Installation
------------

Install pytuq as described in `pytuq/README.md <https://github.com/sandialabs/pytuq/blob/main/README.md>`_:

.. note::

   For NERSC users, consider placing your conda environment in ``/global/common/software``
   for better performance and persistence. Also, remember to ``module load conda`` before creating environments and installing. See the `NERSC Python documentation
   <https://docs.nersc.gov/development/languages/python/nersc-python/#moving-your-conda-setup-to-globalcommonsoftware>`_
   for details.

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

.. note::

   **Comprehensive Installation:**

   For a detailed installation script that includes AMReX, pyAMReX, and PyTUQ setup in a conda environment, see ``GuidedTutorials/HeatEquation_UQ/example_detailed_install.sh`` `example_detailed_install.sh <../../../GuidedTutorials/HeatEquation_UQ/example_detailed_install.sh>`_


Examples
--------

C++ AMReX + PyTUQ (BASH driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Build and Run
   :open:

   **Prerequisites**:

   - AMReX and pytuq cloned at the same directory level as amrex-tutorials
   - GNU Parallel for task management: ``sudo apt-get install parallel``

   .. code-block:: bash
      :caption: Build C++ example

      cd amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-1
      make -j4

   .. code-block:: bash
      :caption: Run with bash script

      ./wf_uqpc.x

.. dropdown:: Understanding GNU Parallel Workflow Pattern

   First, this script generate

   After PyTUQ scripts generate an input parameter file (``psam.txt``), the Case-1 workflow uses GNU Parallel to run multiple simulations efficiently and collect outputs into a results file (``ysam.txt``) that PyTUQ can use for surrogate model fitting. Here's how the pattern works:

   .. code-block:: bash
      :caption: Example GNU Parallel command (``wf_uqpc.x``)

      parallel --jobs 4 --keep-order --colsep ' ' \
        './main3d.gnu.ex inputs diffusion_coeff={1} init_amplitude={2} init_width={3} \
        datalog=datalog_{#}.txt \
        > /dev/null 2>&1 \
        && tail -1 datalog_{#}.txt' \
      :::: psam.txt > ysam.txt

   **Understanding the placeholders:**

   - ``{1}``, ``{2}``, ``{3}`` - **Column numbers** from the input file (``psam.txt``)
   - ``{#}`` - **Line number** from the input file (used for unique file naming)

   **How it works:**

   1. **Input file (psam.txt)** - Space/tab-separated parameter samples (one row per simulation):

      .. code-block:: text

         1.252744433619e+00 1.189332992765e+00 1.107922825308e-02
         1.153626977108e+00 8.912357108037e-01 8.456030290466e-03
         3.265959848625e-01 7.285478940917e-01 7.687549687284e-03

   2. **For each row**, GNU Parallel:

      - Substitutes ``{1}`` with column 1
      - Substitutes ``{2}`` with column 2
      - Substitutes ``{3}`` with column 3
      - Substitutes ``{#}`` with the line number (e.g., ``1``) → ``datalog_1.txt``

   3. **After simulation**, extracts output from final line of simulation standard output with ``tail -1 datalog_{#}.txt``

   4. **Output file (ysam.txt)** - Collected results (one row per simulation):

      .. code-block:: text

         1.106358248808103    0.007723115686160915    0.0562709919848521       253.0710548041209
         0.8113473575862051   0.003858932516788893    0.03451729840856418      126.4495007101384
         0.6608518630828316   0.002734417468982227    0.02636512039600031      89.60139162360962
         1.082407360836961    0.01225809407946649     0.06904148772504816      401.6732267959578
         1.279158901768254    0.006549137494533053    0.05630882962684351      214.6021374208591

   **Key options:**

   - ``--jobs 4`` - Run 4 simulations in parallel
   - ``--colsep ' '`` - Columns separated by spaces (use ``'\t'`` for tabs)
   - ``-k`` - Keep output order matching input order (important for parallel jobs)
   - ``::::`` - Read input from file (``psam.txt``)

   **Output extraction alternatives:**

   Instead of ``tail -1 datalog_{#}.txt``, you can use other extraction methods (see :ref:`pytuq_extending_tutorial` for output examples):

   - **Plotfile tools**: ``fextrema.gnu.ex plt_{#} | tail -1 | awk '{print $3}'``
   - **Custom post-processing script**: ``./postprocess_run.sh {#}``

     A post-processing script can take the line number ``{#}`` as an argument to locate and process the corresponding simulation outputs (datalog, plotfile, etc.), then print the space/tab-separated row for that simulation.

   This pattern makes it easy to:

   - Take a parameter file with N rows (one per simulation)
   - Run simulations in parallel with unique output files
   - Collect results into an output file with N rows (one per simulation result)

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

   .. note::

      **Offline Workflow (similar to Case-1):**

      Case-2 can also be run in offline mode where you manually generate training data, then fit the surrogate model. This is useful for running simulations on HPC systems and post-processing locally:

      .. code-block:: bash
         :caption: Generate training data offline

         # Call model.x on parameter samples to generate outputs
         ./model.x ptrain.txt ytrain.txt

      The ``model.x`` wrapper script manages calling your C++ executable for each parameter set and collecting the outputs. After generating ``ytrain.txt``, use PyTUQ's ``pc_fit.py`` to construct the surrogate model.

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

.. note::

   **Advanced Workflow Management:**

   For more complex UQ workflows requiring sophisticated task management, consider these strategies:

   - **SLURM Job Arrays**: For running many similar parameter sweep jobs - `NERSC Job Arrays Documentation <https://docs.nersc.gov/jobs/examples/#job-arrays>`_
   - **Hydra Submitit Launcher**: For configuration-driven HPC job submission - `Hydra Submitit Usage <https://hydra.cc/docs/plugins/submitit_launcher/#usage>`_
   - **libEnsemble**: For dynamic ensemble management and adaptive sampling - `libEnsemble Platforms Guide <https://libensemble.readthedocs.io/en/main/platforms/platforms_index.html>`_

   See the `NERSC Workflow Documentation <https://docs.nersc.gov/jobs/workflow/>`_ for the latest recommendations on workflow management strategies.

Customizing the Workflow
------------------------

.. toctree::
   :maxdepth: 1

   HeatEquation_UQ_MathematicalDetails
   HeatEquation_UQ_ExtendingTutorial

Summary
-------

**Key Takeaways:**

* PyTUQ can use models with ``outputs = model(inputs)`` interface
* C++ codes can use wrapper scripts or gnu parallel; Python codes integrate directly
* Best practices for Perlmutter requires python environment modules

Additional Resources
--------------------

**PyTUQ Resources:**

- `PyTUQ Documentation <https://sandialabs.github.io/pytuq>`_
- `PyTUQ Examples directory <https://github.com/sandialabs/pytuq/tree/main/examples>`_
- eebaill, ksargsyan, & Bert Debusschere. (2025). sandialabs/pytuq: v1.0.0z (v1.0.0z). Zenodo. https://doi.org/10.5281/zenodo.17110054

**AMReX Resources:**

- `AMReX Documentation <https://amrex-codes.github.io/amrex/docs_html/>`_
- `pyAMReX Documentation <https://pyamrex.readthedocs.io>`_
- :ref:`_guided_heat` - Base tutorial this builds upon

**Uncertainty Quantification Theory:**

- Ghanem, Roger, David Higdon, and Houman Owhadi, eds. *Handbook of Uncertainty Quantification*. Vol. 6. New York: Springer, 2017. (For workflow, plotting, and analysis specifics)

.. seealso::

   For complete working examples of the ``outputs = model(inputs)`` pattern, see:

   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-1/`` - C++ executable and python scripts called from a bash workflow script
   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-2/`` - C++ executable driven by python wrapping bash
   - ``amrex-tutorials/GuidedTutorials/HeatEquation_UQ/Case-3/`` - PyAMReX native
