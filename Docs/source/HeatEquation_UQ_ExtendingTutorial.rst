.. _pytuq_extending_tutorial:

Extending this tutorial to other applications
==============================================

Identify / Add input parameters
--------------------------------

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
----------------------------

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
----------------------------------

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

Configure PyTUQ Parameters
---------------------------

PyTUQ requires configuration files specifying uncertain parameters and output quantities of interest. For BASH-driven workflows (Case-1), create these files directly:

**Parameter Configuration (param_margpc.txt)**

Specify mean and standard deviation for each uncertain parameter (one per line):

.. code-block:: bash

   echo "1 0.25 " > param_margpc.txt        # diffusion_coeff: mean=1.0, std=0.25
   echo "1 0.25" >> param_margpc.txt        # init_amplitude: mean=1.0, std=0.25
   echo "0.01 0.0025" >> param_margpc.txt   # init_width: mean=0.01, std=0.0025

**Parameter Names (pnames.txt)**

List parameter names matching your AMReX ParmParse inputs:

.. code-block:: bash

   echo "diffusion_coeff" > pnames.txt
   echo "init_amplitude" >> pnames.txt
   echo "init_width" >> pnames.txt

**Output Names (outnames.txt)**

Specify quantities of interest to extract from simulation outputs:

.. code-block:: bash

   echo "max_temp" > outnames.txt
   echo "mean_temp" >> outnames.txt
   echo "std_temp" >> outnames.txt
   echo "total_energy" >> outnames.txt

**Polynomial Chaos Configuration**

Set the polynomial chaos type and order:

.. code-block:: bash

   PCTYPE="HG"    # Hermite-Gaussian for normal distributions
   ORDER=1        # First-order polynomial chaos
   NSAM=111       # Number of samples (depends on dimensionality and order)

.. note::

   **Polynomial Chaos Types:**

   - ``HG`` (Hermite-Gaussian): For normal/Gaussian distributions
   - ``LU`` (Legendre-Uniform): For uniform distributions
   - ``LG`` (Laguerre-Gamma): For gamma distributions

   **Sample Count:** For ``d`` dimensions and order ``p``, you need at least ``(p+d)!/(p!*d!)`` samples. For 3D with order 1: minimum 4 samples, recommended 111+.

**PyTUQ Workflow Steps**

The workflow consists of three stages:

1. **Prepare PC basis** - Use ``pc_prep.py`` to initialize polynomial chaos basis functions based on parameter distributions

2. **Sample parameter space** - Use ``pc_sam.py`` to generate parameter samples (creates ``qsam.txt``)

3. **Fit surrogate model** - Use ``pc_fit.py`` to construct polynomial chaos expansion from simulation outputs

.. note::

   **Case-2 Flexibility:**

   Case-2 (Python-wrapped C++) supports both approaches:

   - **BASH file configuration** (like Case-1): Use text files (``param_margpc.txt``, ``pnames.txt``, ``outnames.txt``) with a ``model.x`` wrapper script that specifies which C++ executable and inputs file to use
   - **Python metadata**: Define parameter distributions in the model class's ``_get_field_info()`` method

   Case-3 (native PyAMReX) requires Python metadata since there's no separate executable to wrap.
