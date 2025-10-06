#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 The AMReX Community
#
# This file is part of AMReX.
#
# License: BSD-3-Clause-LBNL
# Authors: Revathi Jambunathan, Edoardo Zoni, Olga Shapoval, David Grote, Axel Huebl

import amrex.space3d as amr
import numpy as np
from typing import Tuple, Dict, Union


def load_cupy():
    """Load appropriate array library (CuPy for GPU, NumPy for CPU)."""
    if amr.Config.have_gpu:
        try:
            import cupy as cp
            amr.Print("Note: found and will use cupy")
            return cp
        except ImportError:
            amr.Print("Warning: GPU found but cupy not available! Trying managed memory in numpy...")
            import numpy as np
            return np
        if amr.Config.gpu_backend == "SYCL":
            amr.Print("Warning: SYCL GPU backend not yet implemented for Python")
            import numpy as np
            return np
    else:
        import numpy as np
        amr.Print("Note: found and will use numpy")
        return np


def main(n_cell: int = 32, max_grid_size: int = 16, nsteps: int = 100,
         plot_int: int = 100, dt: float = 1e-5, plot_files_output: bool = False,
         verbose: int = 1, diffusion_coeff: float = 1.0, init_amplitude: float = 1.0,
         init_width: float = 0.01) -> Tuple[amr.MultiFab, amr.Geometry]:
    """
    Run the heat equation simulation.
    The main function, automatically called below if called as a script.

    Returns:
    --------
    tuple : (phi_new, geom)
        Final state and geometry
    """
    # CPU/GPU logic
    xp = load_cupy()

    # AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    dom_lo = amr.IntVect(*amr.d_decl(       0,        0,        0))
    dom_hi = amr.IntVect(*amr.d_decl(n_cell-1, n_cell-1, n_cell-1))

    # Make a single box that is the entire domain
    domain = amr.Box(dom_lo, dom_hi)

    # Make BoxArray and Geometry:
    # ba contains a list of boxes that cover the domain,
    # geom contains information such as the physical domain size,
    # number of points in the domain, and periodicity

    # Initialize the boxarray "ba" from the single box "domain"
    ba = amr.BoxArray(domain)
    # Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.max_size(max_grid_size)

    # This defines the physical box, [0,1] in each direction.
    real_box = amr.RealBox([*amr.d_decl( 0., 0., 0.)], [*amr.d_decl( 1., 1., 1.)])

    # This defines a Geometry object
    # periodic in all direction
    coord = 0 # Cartesian
    is_per = [*amr.d_decl(1,1,1)] # periodicity
    geom = amr.Geometry(domain, real_box, coord, is_per);

    # Extract dx from the geometry object
    dx = geom.data().CellSize()

    # Nghost = number of ghost cells for each array
    Nghost = 1

    # Ncomp = number of components for each array
    Ncomp = 1

    # How Boxes are distrubuted among MPI processes
    dm = amr.DistributionMapping(ba)

    # Allocate two phi multifabs: one will store the old state, the other the new.
    phi_old = amr.MultiFab(ba, dm, Ncomp, Nghost)
    phi_new = amr.MultiFab(ba, dm, Ncomp, Nghost)
    phi_old.set_val(0.)
    phi_new.set_val(0.)

    # time = starting time in the simulation
    time = 0.

    # Ghost cells
    ng = phi_old.n_grow_vect
    ngx = ng[0]
    ngy = ng[1]
    ngz = ng[2]

    # Loop over boxes
    for mfi in phi_old:
        bx = mfi.validbox()
        # phiOld is indexed in reversed order (z,y,x) and indices are local
        phiOld = xp.array(phi_old.array(mfi), copy=False)
        # set phi = 1 + amplitude * e^(-(r-0.5)^2 / width)
        x = (xp.arange(bx.small_end[0],bx.big_end[0]+1,1) + 0.5) * dx[0]
        y = (xp.arange(bx.small_end[1],bx.big_end[1]+1,1) + 0.5) * dx[1]
        z = (xp.arange(bx.small_end[2],bx.big_end[2]+1,1) + 0.5) * dx[2]
        rsquared = ((z[:         , xp.newaxis, xp.newaxis] - 0.5)**2
                  + (y[xp.newaxis, :         , xp.newaxis] - 0.5)**2
                  + (x[xp.newaxis, xp.newaxis, :         ] - 0.5)**2) / init_width
        phiOld[:, ngz:-ngz, ngy:-ngy, ngx:-ngx] = 1. + init_amplitude * xp.exp(-rsquared)

    # Write a plotfile of the initial data if plot_int > 0 and plot_files_output
    if plot_int > 0 and plot_files_output:
        step = 0
        pltfile = amr.concatenate("plt", step, 5)
        varnames = amr.Vector_string(['phi'])
        amr.write_single_level_plotfile(pltfile, phi_old, varnames, geom, time, 0)

    for step in range(1, nsteps+1):
        # Fill periodic ghost cells
        phi_old.fill_boundary(geom.periodicity())

        # new_phi = old_phi + dt * Laplacian(old_phi)
        # Loop over boxes
        for mfi in phi_old:
            phiOld = xp.array(phi_old.array(mfi), copy=False)
            phiNew = xp.array(phi_new.array(mfi), copy=False)
            hix = phiOld.shape[3]
            hiy = phiOld.shape[2]
            hiz = phiOld.shape[1]

            # Heat equation with parameterized diffusion
            # Advance the data by dt
            phiNew[:, ngz:-ngz,ngy:-ngy,ngx:-ngx] = (
                phiOld[:, ngz:-ngz,ngy:-ngy,ngx:-ngx] + dt * diffusion_coeff *
                     ((   phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx+1:hix-ngx+1]
                       -2*phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx  :-ngx     ]
                         +phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx-1:hix-ngx-1]) / dx[0]**2
                     +(   phiOld[:, ngz  :-ngz     , ngy+1:hiy-ngy+1, ngx  :-ngx     ]
                       -2*phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx  :-ngx     ]
                         +phiOld[:, ngz  :-ngz     , ngy-1:hiy-ngy-1, ngx  :-ngx     ]) / dx[1]**2
                     +(   phiOld[:, ngz+1:hiz-ngz+1, ngy  :-ngy     , ngx  :-ngx     ]
                       -2*phiOld[:, ngz  :-ngz     , ngy  :-ngy     , ngx  :-ngx     ]
                         +phiOld[:, ngz-1:hiz-ngz-1, ngy  :-ngy     , ngx  :-ngx     ]) / dx[2]**2))

        # Update time
        time = time + dt

        # Copy new solution into old solution
        amr.copy_mfab(dst=phi_old, src=phi_new, srccomp=0, dstcomp=0, numcomp=1, nghost=0)

        # Tell the I/O Processor to write out which step we're doing
        if(verbose > 0):
            amr.Print(f'Advanced step {step}\n')

        # Write a plotfile of the current data (plot_int was defined in the inputs file)
        if plot_int > 0 and step%plot_int == 0 and plot_files_output:
            pltfile = amr.concatenate("plt", step, 5)
            varnames = amr.Vector_string(['phi'])
            amr.write_single_level_plotfile(pltfile, phi_new, varnames, geom, time, step)

    return phi_new, geom


def parse_inputs() -> Dict[str, Union[int, float, bool]]:
    """Parse inputs using AMReX ParmParse to_dict method."""
    pp = amr.ParmParse("")

    # Add inputs file if it exists
    import os
    inputs_file = "inputs"
    if os.path.exists(inputs_file):
        pp.addfile(inputs_file)

    # Default values with their types
    defaults = {
        'n_cell': 32,
        'max_grid_size': 16,
        'nsteps': 1000,
        'plot_int': 100,
        'dt': 1.0e-5,
        'plot_files_output': False,
        'diffusion_coeff': 1.0,
        'init_amplitude': 1.0,
        'init_width': 0.01
    }

    try:
        # Convert entire ParmParse table to Python dictionary
        all_params = pp.to_dict()

        # Extract our specific parameters with proper type conversion
        params = {}
        for key, default_value in defaults.items():
            if key in all_params:
                try:
                    # Convert string to appropriate type based on default
                    if isinstance(default_value, int):
                        params[key] = int(all_params[key])
                    elif isinstance(default_value, float):
                        params[key] = float(all_params[key])
                    elif isinstance(default_value, bool):
                        # Handle boolean conversion from string
                        val_str = str(all_params[key]).lower()
                        if val_str in ('true', '1', 'yes', 'on'):
                            params[key] = True
                        elif val_str in ('false', '0', 'no', 'off'):
                            params[key] = False
                        else:
                            # If unrecognized, use default
                            params[key] = default_value
                    else:
                        params[key] = all_params[key]
                except (ValueError, TypeError) as e:
                    amr.Print(f"Warning: Could not convert parameter {key}='{all_params[key]}' to {type(default_value).__name__}, using default: {default_value}")
                    params[key] = default_value
            else:
                params[key] = default_value

        # Optional: print the parameters we're actually using
        amr.Print("Using parameters:")
        for key, value in params.items():
            amr.Print(f"  {key}: {value}")

        return params

    except Exception as e:
        amr.Print(f"Warning: Could not parse parameters with to_dict(): {e}")
        amr.Print("Using default values")
        return defaults

if __name__ == '__main__':
    # Initialize AMReX
    amr.initialize([])

    try:
        # Parse inputs
        params = parse_inputs()

        # Run simulation
        main(**params)

    finally:
        # Finalize AMReX
        amr.finalize()
