# Heat Equation Python Driver

This example demonstrates the minimal setup needed to create a one-line Python interface for AMReX C++ simulation codes.

## Key Feature

The main goal is to enable this simple Python usage pattern:

```python
import amrex_heat

# One-line simulation execution with structured results
result = amrex_heat.run_simulation(["./HeatEquation_PythonDriver", "inputs"])

print(f"Max temperature: {result.max_temperature}")
print(f"Final time: {result.final_time}")
```

## Files

- `main.cpp` - Heat equation solver with `heat_equation_main()` function
- `bindings.cpp` - Minimal pybind11 interface (just the one-liner)
- `CMakeLists.txt` - Build configuration with pybind11 support
- `pybind11.cmake` - Pybind11 infrastructure from pyamrex
- `inputs` - Simulation parameters
- `test.py` - Example Python usage

## Architecture

The key architectural insight is the function refactoring pattern:

```cpp
// Reusable simulation function
SimulationResult heat_equation_main(int argc, char* argv[]) {
    // All simulation logic here
    return result;
}

// C++ entry point
int main(int argc, char* argv[]) {
    heat_equation_main(argc, argv);
    return 0;
}
```

This allows the same simulation code to be called from:
- Command line: `./HeatEquation_PythonDriver inputs`
- Python: `amrex_heat.run_simulation(["./HeatEquation_PythonDriver", "inputs"])`

## Building

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Running

### C++ executable
```bash
cd build
./HeatEquation_PythonDriver inputs
```

### Python interface
```bash
cd build
python ../test.py
```

## What's Minimal

This example includes only the essential components:

1. **SimulationResult struct** - Simple data structure for return values
2. **heat_equation_main function** - Refactored simulation code
3. **Minimal bindings** - Just `run_simulation()` function
4. **Basic CMake** - Pybind11 integration

No callback system, no complex data structures, no advanced features - just the core one-liner interface.

## Extending

This foundation can be extended with:
- Callback system for progress monitoring
- More return data (arrays, MultiFabs)
- Parameter setting from Python
- Integration with full pyamrex infrastructure