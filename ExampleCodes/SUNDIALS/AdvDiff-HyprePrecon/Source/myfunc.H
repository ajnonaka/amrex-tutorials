#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_MultiFab.H>

using namespace amrex;

void main_main ();

void ComputeDiffusion(MultiFab& S_rhs,
                      MultiFab& S_data,
                      const Real& Dx,
                      const Real& Dy,
                      const GpuArray<Real,AMREX_SPACEDIM> dx);

#endif
