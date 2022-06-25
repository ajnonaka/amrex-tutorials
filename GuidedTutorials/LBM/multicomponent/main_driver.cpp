#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include "StructFact.H"
#include "TimeCorrelation.H"

using namespace amrex;

#include "LBM_binary.H"

void main_driver(const char* argv) {

  // default grid parameters
  int nx = 16;
  int max_grid_size = 8;

  // default time stepping parameters
  int nsteps = 100;
  int plot_int = 10;

  // default amplitude of sinusoidal shear wave
  Real A = 0.001;
  
  // input parameters
  ParmParse pp;
  pp.query("nx", nx);
  pp.query("max_grid_size", max_grid_size);
  pp.query("nsteps", nsteps);
  pp.query("plot_int", plot_int);
  pp.query("density", density);
  pp.query("temperature", temperature);
  pp.query("tau", tau);
  pp.query("A", A);

  // default one ghost/halo layer
  int nghost = 1;
  
  // set up Box and Geomtry
  IntVect dom_lo(0, 0, 0);
  IntVect dom_hi(nx-1, nx-1, nx-1);
  Array<int,3> periodicity({1,1,1});

  Box domain(dom_lo, dom_hi);

  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);

  BoxArray ba(domain);

  // split BoxArray into chunks no larger than "max_grid_size" along a direction
  ba.maxSize(max_grid_size);

  DistributionMapping dm(ba);

  // set up MultiFabs
  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab moments(ba, dm, nvel, 0);
  MultiFab hydrovs(ba, dm, nvel, 0);

  // set variable names for output
  int numVars = moments.nComp();
  Vector< std::string > var_names(numVars);
  std::string name;
  int cnt = 0;
  // rho
  var_names[cnt++] = "rho";
  // velx, vely, velz
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    name = "u";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // pxx, pxy, pxz, pyy, pyz, pzz
  for (int i=0; i<AMREX_SPACEDIM; ++i) {
    for (int j=i; j<AMREX_SPACEDIM; ++j) {
      name = "p";
      name += (120+i);
      name += (120+j);
      var_names[cnt++] = name;
    }
  }
  // kinetic moments
  for (; cnt<numVars;) {
    name = "m";
    name += std::to_string(cnt);
    var_names[cnt++] = name;
  }

  // set up references to arrays
  auto const & f = fold.arrays();    // LB populations
  auto const & m = moments.arrays(); // LB moments
  auto const & h = hydrovs.arrays(); // hydrodynamic fields

  // INITIALIZE: set up sinusoidal shear wave u_y(x)=A*sin(k*x)
  Real time = 0.0;
  ParallelFor(fold, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real uy = A*std::sin(2.*M_PI*x/nx);
    const RealVect u = {0., uy, 0. };
    for (int i=0; i<nvel; ++i) {
      m[nbx](x,y,z,i) = mequilibrium(density, u)[i];
      f[nbx](x,y,z,i) = fequilibrium(density, u)[i];
    }
    for (int i=0; i<10; ++i) {
      h[nbx](x,y,z,i) = hydrovars(mequilibrium(density, u))[i];
    }
  });

  // Write a plotfile of the initial data if plot_int > 0
  if (plot_int > 0) {
    int step = 0;
    const std::string& pltfile = amrex::Concatenate("plt",step,5);
    WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, time, step);
  }
  Print() << "LB initialized\n";

  // TIMESTEP
  for (int step=1; step <= nsteps; ++step) {
    fold.FillBoundary(geom.periodicity());
    for (MFIter mfi(fold); mfi.isValid(); ++mfi) {
      const Box& valid_box = mfi.validbox();
      const Array4<Real>& fOld = fold.array(mfi);
      const Array4<Real>& fNew = fnew.array(mfi);
      const Array4<Real>& mom = moments.array(mfi);
      const Array4<Real>& hydrovars = hydrovs.array(mfi);
      ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
        stream_collide(x, y, z, mom, fOld, fNew, hydrovars, engine);
      });
    }
    std::swap(fold,fnew);
    Print() << "LB step " << step << "\n";

    // OUTPUT
    time = static_cast<Real>(step);
    if (plot_int > 0 && step%plot_int ==0) {
      const std::string& pltfile = amrex::Concatenate("plt",step,5);
      WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, time, step);
    }

  }

}
