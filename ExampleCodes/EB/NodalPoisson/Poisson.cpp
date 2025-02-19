#include "Poisson.H"

using namespace amrex;

void InitData (amrex::MultiFab& State, int icenter, int jcenter)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(State,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();
        const Array4<Real>& q = State.array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (i==icenter && j==jcenter) {
                q(i,j,k) = 1.0;
            } else {
                q(i,j,k) = 0.0;
            }
        });
    }
}
