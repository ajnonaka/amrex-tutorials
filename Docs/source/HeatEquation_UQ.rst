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

.. note::

   **Comprehensive Installation:**

   For a detailed installation script that includes AMReX, pyAMReX, and PyTUQ setup in a conda environment, see ``GuidedTutorials/HeatEquation_UQ/example_detailed_install.sh``

Examples
--------

C++ AMReX + PyTUQ (BASH driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Build and Run
   :open:

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
