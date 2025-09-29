#!/usr/bin/env python3
"""
Simple test of the Heat Equation Python Driver

This demonstrates the minimal one-line interface to run AMReX simulations
from Python and get structured results back.
"""

import amrex_heat

def main():
    print("Heat Equation Python Driver Test")
    print("=" * 40)

    # The key feature: one-line simulation execution
    print("Running simulation...")
    result = amrex_heat.run_simulation(["./HeatEquation_PythonDriver", "inputs"])

    # Access results directly
    print(f"\nSimulation Results:")
    print(f"  Success: {result.success}")
    print(f"  Final step: {result.final_step}")
    print(f"  Final time: {result.final_time:.6f}")
    print(f"  Max temperature: {result.max_temperature:.6f}")

    # You can use the results for further processing
    if result.success:
        print(f"\n✓ Simulation completed successfully!")
        print(f"  Temperature decay: {2.0 - result.max_temperature:.6f}")
    else:
        print(f"\n✗ Simulation failed!")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())