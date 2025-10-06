.. _guided_heat_python_driver:

Tutorial: Heat Equation - Python Driver
========================================

.. admonition:: **Time to Complete**: 10 mins
   :class: warning

   **PREREQUISITES:**
     - Complete :ref:`guided_heat_simple` tutorial first

   **GOALS:**
     - Create minimal Python interface to AMReX C++ code
     - Implement one-line simulation execution from Python
     - Access simulation results programmatically
     - Understand the function refactoring pattern


This tutorial demonstrates two complementary approaches for creating Python interfaces
to AMReX C++ simulation codes:

- **Case-1**: Minimal pybind11 interface with one-line simulation execution
- **Case-2**: Pure Python approach using pyamrex MultiFabs directly

Both approaches can handle complex simulations and MultiFab data structures.

The One-Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The goal is to enable simple Python usage patterns that work across different simulation types:

**Case-1 Pattern (pybind11 C++ interface):**

.. code-block:: python

   import amrex_sim

   # One-line simulation execution with structured results
   result = amrex_sim.run_simulation(["./YourSimulation", "inputs"])

   print(f"Max value: {result.max_value}")
   print(f"Final time: {result.final_time}")
   print(f"Success: {result.success}")

**Case-2 Pattern (Pure Python with pyamrex):**

.. code-block:: python

   from YourModel import SimulationModel

   # One-line execution with parameter arrays
   model = SimulationModel(use_parmparse=True)
   results = model(params)  # Returns numpy array of results

   print(f"Results: {results}")  # [max, mean, std, integral, center]

Both approaches provide direct access to simulation results without subprocess overhead
or complex callback systems.

Key Architecture Pattern
~~~~~~~~~~~~~~~~~~~~~~~~

Both approaches demonstrate the core principle of creating **simplified interfaces** to complex AMReX simulations. The key insight is to transform existing simulation logic into easily callable functions with structured return values:

**Case-1 Approach (C++ Refactoring):**
Refactor the C++ ``main()`` function into a reusable function that can be called from both command line and Python:

.. code-block:: cpp

   // Reusable simulation function
   SimulationResult simulation_main(int argc, char* argv[]) {
       amrex::Initialize(argc, argv);
       {
           // All simulation logic here
           // ...
           result.max_value = multifab.max(0);
           result.success = true;
       }
       amrex::Finalize();
       return result;
   }

**Case-2 Approach (Python Function Design):**
Create a callable class or function that encapsulates simulation parameters and execution:

.. code-block:: python

   class SimulationModel:
       def __call__(self, params):
           # All simulation logic using pyamrex
           # ...
           return np.array([max_val, mean_val, std_val, integral, center_val])

Both patterns enable the same core benefit: **one-line simulation execution** with structured results, regardless of whether the underlying implementation uses C++ bindings or pure Python.

Two Implementation Approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Case-1: Minimal pybind11 Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Location: :code:`amrex-tutorials/GuidedTutorials/HeatEquation_PythonDriver/Case-1/`

This approach creates C++ bindings using pybind11 for direct simulation execution:

.. code-block:: bash

   cd Case-1
   mkdir build && cd build
   cmake ..
   cmake --build . -j4

Creates both:
- ``HeatEquation_PythonDriver`` - C++ executable
- ``amrex_heat.cpython-*.so`` - Python module

.. note::
   The GNUmakefile in Case-1 is experimental and under development as an alternative build system.

Case-2: Pure Python with pyamrex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Location: :code:`amrex-tutorials/GuidedTutorials/HeatEquation_PythonDriver/Case-2/`

This approach uses pure Python with pyamrex to access MultiFabs and AMReX functionality directly:

.. code-block:: bash

   cd Case-2
   python HeatEquationModel.py

Case-1 Files (pybind11 Approach)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Case-1`` directory contains:

- ``main.cpp`` - Heat equation solver with ``heat_equation_main()`` function
- ``bindings.cpp`` - Minimal pybind11 interface exposing the one-liner
- ``CMakeLists.txt`` - Build configuration with pybind11 support
- ``GNUmakefile`` - Experimental GNU Make build system (under development)
- ``pybind11.cmake`` - Pybind11 infrastructure (copied from pyamrex)
- ``inputs`` - Simulation parameters (``n_cell``, ``dt``, etc.)
- ``test.py`` - Example Python usage script
- ``README.md`` - Documentation and usage instructions

Case-2 Files (Pure Python Approach)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Case-2`` directory contains:

- ``HeatEquationModel.py`` - Pure Python implementation using pyamrex MultiFabs
- ``inputs`` - Simulation parameters
- ``README.md`` - Documentation for the pure Python approach

Case-1 Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Result Structure
^^^^^^^^^^^^^^^^

The simulation returns a simple struct with essential information:

.. code-block:: cpp

   struct SimulationResult {
       double max_temperature;
       int final_step;
       double final_time;
       bool success;
   };

This struct is automatically exposed to Python through pybind11, allowing direct
access to all fields.

   Minimal Python Bindings
   ++++++++++++++++++++++++

The Python interface is implemented with minimal pybind11 code in ``bindings.cpp``.
The key components are:

1. **Forward declaration** of the ``SimulationResult`` struct from ``main.cpp``
2. **Function declaration** for ``heat_equation_main()``
3. **Pybind11 module** that exposes both the struct and function

.. code-block:: cpp

   // Forward declarations from main.cpp
   struct SimulationResult {
       double max_temperature;
       int final_step;
       double final_time;
       bool success;
   };

   SimulationResult heat_equation_main(int argc, char* argv[]);

   PYBIND11_MODULE(amrex_heat, m) {
       m.doc() = "Minimal AMReX Heat Equation Python Interface";

       // Expose SimulationResult struct
       py::class_<SimulationResult>(m, "SimulationResult")
           .def_readonly("max_temperature", &SimulationResult::max_temperature)
           .def_readonly("final_step", &SimulationResult::final_step)
           .def_readonly("final_time", &SimulationResult::final_time)
           .def_readonly("success", &SimulationResult::success);

       // Main simulation function - one-liner interface
       m.def("run_simulation", [](py::list args) {
           // Convert Python list to C++ argc/argv with proper lifetime management
           std::vector<std::string> args_str;
           for (auto item : args) {
               args_str.push_back(py::str(item));
           }

           std::vector<char*> args_cstr;
           for (auto& s : args_str) {
               args_cstr.push_back(const_cast<char*>(s.c_str()));
           }
           args_cstr.push_back(nullptr);  // Null terminate

           return heat_equation_main(static_cast<int>(args_cstr.size() - 1), args_cstr.data());
       }, "Run the heat equation simulation and return results");
   }

The argument conversion ensures proper lifetime management of the C++ strings and
null-terminates the argument array as expected by ``argc/argv`` conventions.

   CMake Integration
   +++++++++++++++++

The ``CMakeLists.txt`` integrates pybind11 using the pyamrex infrastructure. The key
elements are the pybind11 integration and building both targets from the same source:

.. code-block:: cmake

   # Use pyamrex pybind11 infrastructure
   include(pybind11.cmake)

   # Add the main executable
   add_executable(HeatEquation_PythonDriver main.cpp)

   # Add the pybind11 module including main simulation logic
   pybind11_add_module(amrex_heat bindings.cpp main.cpp)

   # Link AMReX to both targets
   target_link_libraries(HeatEquation_PythonDriver PRIVATE AMReX::amrex)
   target_link_libraries(amrex_heat PRIVATE AMReX::amrex)

The ``pybind11.cmake`` file is copied from the pyamrex repository and provides the necessary
pybind11 infrastructure without requiring a separate pyamrex installation.

Running the Examples
~~~~~~~~~~~~~~~~~~~~

Case-1: pybind11 Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^

**C++ Executable:**

.. code-block:: bash

   cd Case-1/build
   ./HeatEquation_PythonDriver inputs

**Python Interface:**

.. code-block:: bash

   cd Case-1/build
   python ../test.py

Case-2: Pure Python with pyamrex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Pure Python Execution:**

.. code-block:: bash

   cd Case-2
   python HeatEquationModel.py

The Case-1 Python script demonstrates accessing simulation results:

.. code-block:: python

   import amrex_heat

   print("Running simulation...")
   result = amrex_heat.run_simulation(["./HeatEquation_PythonDriver", "inputs"])

   print(f"Simulation Results:")
   print(f"  Success: {result.success}")
   print(f"  Final step: {result.final_step}")
   print(f"  Final time: {result.final_time:.6f}")
   print(f"  Max temperature: {result.max_temperature:.6f}")

Expected output:

.. code-block::

   Heat Equation Python Driver Test
   ========================================
   Running simulation...
   Advanced step 1
   Advanced step 2
   ...
   Advanced step 1000

   Simulation Results:
     Success: True
     Final step: 1000
     Final time: 0.010000
     Max temperature: 1.089070

   ✓ Simulation completed successfully!

Case-2 Example Output
^^^^^^^^^^^^^^^^^^^^^

The pure Python approach provides object-oriented access:

.. code-block:: python

   import numpy as np
   from HeatEquationModel import HeatEquationModel

   # Create model using inputs file
   model = HeatEquationModel(use_parmparse=True)

   # Run with parameter array [diffusion_coeff, init_amplitude, init_width]
   params = np.array([1.0, 1.0, 0.01])
   results = model(params)

   print(f"Results: {results}")
   # Output: [max_value, mean_value, std_dev, total_heat, center_value]

Choosing the Right Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use Case-1 (pybind11) when:**
- You want to create custom bindings for existing C++ code
- You need a one-line interface wrapping complex C++ simulation logic
- You want both command-line and Python interfaces from the same codebase
- You prefer minimal changes to existing C++ code

**Use Case-2 (Pure Python with pyamrex) when:**
- You want to write simulation logic directly in Python
- You prefer leveraging the full pyamrex ecosystem
- You want object-oriented simulation management
- You need rapid prototyping and development

Adapting These Patterns
^^^^^^^^^^^^^^^^^^^^^^^

**For Case-1 (pybind11 approach):**

1. **Refactor main()**: Extract simulation logic into reusable function
2. **Design result structure**: Choose what data to return to Python
3. **Create bindings**: Write ``bindings.cpp`` for your interface
4. **Update build**: Add pybind11 support to CMakeLists.txt

**For Case-2 (Pure Python approach):**

1. **Design class interface**: Define methods for initialization, execution, results
2. **Use pyamrex directly**: Leverage MultiFabs and AMReX functionality
3. **Implement simulation logic**: Write algorithm using pyamrex primitives
4. **Add data management**: Handle input/output and state management

Benefits Comparison
~~~~~~~~~~~~~~~~~~~

**Case-1 (pybind11) Benefits:**

- **Minimal C++ changes**: Preserves existing simulation logic
- **Performance**: Direct C++ calls with no overhead
- **Dual interface**: Same code for command line and Python
- **Integration**: Works with existing build systems

**Case-2 (Pure Python) Benefits:**

- **Development speed**: Rapid iteration and testing
- **Ecosystem access**: Full pyamrex functionality available
- **Readability**: Clear Python simulation logic
- **Flexibility**: Easy to modify and extend algorithms

**Both approaches:**

- Support complex MultiFab operations
- Enable sophisticated simulation workflows
- Integrate well with scientific Python ecosystem
- Provide foundation for advanced features

Use Cases for Each Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Case-1 (pybind11) is ideal for:**
- **Wrapping existing C++ codes**: Minimal changes to proven simulation codes
- **Performance-critical workflows**: Direct C++ execution with minimal overhead
- **One-line interfaces**: Simple Python access to complex simulations
- **Hybrid development**: Teams with both C++ and Python expertise

**Case-2 (Pure Python with pyamrex) is ideal for:**
- **Rapid prototyping**: Quick iteration on simulation algorithms
- **Educational purposes**: Clear, readable simulation logic
- **Python-first development**: Teams primarily working in Python
- **Leveraging pyamrex ecosystem**: Using existing pyamrex tools and patterns

**Both approaches support:**
- Parameter sweeps and optimization workflows
- Jupyter notebooks and interactive visualization
- Complex MultiFab operations and data analysis
- Machine learning and data science pipelines

Next Steps
~~~~~~~~~~

This minimal Python interface provides the foundation for more advanced features:

1. **Add more return data**: Include arrays, MultiFab statistics, etc.
2. **Parameter setting**: Allow modification of simulation parameters from Python
3. **Progress monitoring**: Add callback system for real-time updates
4. **Full pyamrex integration**: Access MultiFab data structures directly
5. **Workflow automation**: Build complex simulation pipelines
6. **Generic naming**: Replace heat equation-specific names (``amrex_heat``, ``max_temperature``) with generic equivalents (``amrex_sim``, ``max_value``) for reusability across different simulation types
7. **Numpy-compatible results**: Add options to return data as dictionaries, numpy arrays, or other formats that integrate well with the scientific Python ecosystem

Potential improvements for generic usage:

.. code-block:: cpp

   // Generic module and function names
   PYBIND11_MODULE(amrex_sim, m) {
       // Option 1: Return as dictionary for numpy compatibility
       m.def("run_dict", [](py::list args) {
           auto result = simulation_main(argc, argv);
           py::dict d;
           d["success"] = result.success;
           d["max_value"] = result.max_value;  // Generic field name
           d["final_time"] = result.final_time;
           return d;
       });

       // Option 2: Return numerical data as numpy array
       m.def("run_array", [](py::list args) {
           auto result = simulation_main(argc, argv);
           py::array_t<double> data = py::array_t<double>(3);
           // Fill array with [final_step, final_time, max_value]
           return py::make_tuple(result.success, data);
       });
   }

The key insight is that this simple pattern scales naturally to support more
complex use cases while maintaining the clean one-line interface.