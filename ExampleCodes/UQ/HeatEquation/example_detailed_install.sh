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
