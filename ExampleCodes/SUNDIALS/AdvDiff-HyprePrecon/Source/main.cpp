#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_TimeIntegrator.H>

#include "myfunc.H"

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{

    if (AMREX_SPACEDIM != 2) {
        amrex::Abort("Only 2D supported; recompile with DIM=2");
    }
    
    // **********************************
    // SIMULATION PARAMETERS

    // number of cells on each side of the domain
    int n_cell;

    // size of each box (or grid)
    int max_grid_size;

    // total steps in simulation
    int nsteps;

    // how often to write a plotfile
    int plot_int;

    // time step
    Real dt;

    // use adaptive time step (dt used to set output times)
    bool adapt_dt = false;

    // adaptive time step relative and absolute tolerances
    Real reltol = 1.0e-4;
    Real abstol = 1.0e-9;

    // Advection and Diffusion Coefficients
    Real advCoeffx = 1.0;
    Real advCoeffy = 1.0;
    Real diffCoeffx = 1.0;
    Real diffCoeffy = 1.0;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        // pp.get means we require the inputs file to have it
        // pp.query means we optionally need the inputs file to have it - but we must supply a default here
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
        pp.get("n_cell",n_cell);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

        // time step
        pp.get("dt",dt);

        // use adaptive step sizes
        pp.query("adapt_dt",adapt_dt);

        // adaptive step tolerances
        pp.query("reltol",reltol);
        pp.query("abstol",abstol);

        pp.query("advCoeffx",advCoeffx);
        pp.query("advCoeffx",advCoeffy);
        pp.query("diffCoeffx",diffCoeffx);
        pp.query("diffCoeffx",diffCoeffy);
    }

    // **********************************
    // SIMULATION SETUP

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity
    BoxArray ba;
    Geometry geom;

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box, [0,1] in each direction.
    RealBox real_box({AMREX_D_DECL(-1.,-1.,-1.)},
                     {AMREX_D_DECL( 1., 1., 1.)});

    // periodic in all direction
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

    // This defines a Geometry object
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // allocate phi MultiFab
    MultiFab phi(ba, dm, Ncomp, Nghost);

    // time = starting time in the simulation
    Real time = 0.0;

    // **********************************
    // INITIALIZE DATA

    // loop over boxes
    for (MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        const Array4<Real>& phi_array = phi.array(mfi);

        Real sigma = 0.1;
        Real a = 1.0/(sigma*sqrt(2*M_PI));
        Real b = -0.5/(sigma*sigma);
        
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real y = prob_lo[1] + (((Real) j) + 0.5) * dx[1];
            Real x = prob_lo[0] + (((Real) i) + 0.5) * dx[0];
            Real r = x * x + y * y;
            phi_array(i,j,k) = a * std::exp(b * r);
        });
    }

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,5);
        WriteSingleLevelPlotfile(pltfile, phi, {"phi"}, geom, time, 0);
    }

    auto rhs_function = [&](MultiFab& S_rhs, MultiFab& S_data, const Real /* time */) {

        // fill periodic ghost cells
        S_data.FillBoundary(geom.periodicity());

        S_rhs.setVal(0.);
        
        ComputeDiffusion(S_rhs, S_data, diffCoeffx, diffCoeffy, dx);
        ComputeAdvection(S_rhs, S_data, advCoeffx, advCoeffy, dx);
    };

    TimeIntegrator<MultiFab> integrator(phi, time);
    integrator.set_rhs(rhs_function);

    if (adapt_dt) {
        integrator.set_adaptive_step();
        integrator.set_tolerances(reltol, abstol);
    } else {
        integrator.set_time_step(dt);
    }

    Real evolution_start_time = ParallelDescriptor::second();

    for (int step = 1; step <= nsteps; ++step)
    {
        // Set time to evolve to
        time += dt;

        Real step_start_time = ParallelDescriptor::second();

        // Advance to output time
        integrator.evolve(phi, time);

        Real step_stop_time = ParallelDescriptor::second() - step_start_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds; dt = " << dt << " time = " << time << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,5);
            WriteSingleLevelPlotfile(pltfile, phi, {"phi"}, geom, time, step);
        }
    }

    Real evolution_stop_time = ParallelDescriptor::second() - evolution_start_time;
    ParallelDescriptor::ReduceRealMax(evolution_stop_time);
    amrex::Print() << "Total evolution time = " << evolution_stop_time << " seconds\n";
}

void ComputeDiffusion(MultiFab& S_rhs,
                      MultiFab& S_data,
                      const Real& Dx,
                      const Real& Dy,
                      const GpuArray<Real,AMREX_SPACEDIM> dx) {

    for ( MFIter mfi(S_data,TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.tilebox();

        const Array4<const Real>& phi_array = S_data.array(mfi);
        const Array4<      Real>& rhs_array = S_rhs.array(mfi);

        // fill the right-hand-side for phi
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            rhs_array(i,j,k) += Dx * ( (phi_array(i+1,j,k) - 2.*phi_array(i,j,k) + phi_array(i-1,j,k)) / (dx[0]*dx[0]) )
                              + Dy * ( (phi_array(i,j+1,k) - 2.*phi_array(i,j,k) + phi_array(i,j-1,k)) / (dx[1]*dx[1]) );
        });
    }
}


void ComputeAdvection(MultiFab& S_rhs,
                      MultiFab& S_data,
                      const Real& Ax,
                      const Real& Ay,
                      const GpuArray<Real,AMREX_SPACEDIM> dx) {

    Real dxInv = 1.0 / dx[0];
    Real dyInv = 1.0 / dx[1];
    Real sideCoeffx = Ax * dxInv;
    Real sideCoeffy = Ay * dyInv;

    for (MFIter mfi(S_data,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        const Array4<const Real>& phi_array = S_data.array(mfi);
        const Array4<      Real>& rhs_array = S_rhs.array(mfi);

        // x-direction
        if (Ax > 0)
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                rhs_array(i,j,k) -= sideCoeffx * (phi_array(i,j,k) - phi_array(i-1,j,k));
            });
        }
        else
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                rhs_array(i,j,k) -= sideCoeffx * (phi_array(i+1,j,k) - phi_array(i,j,k));
            });
        }

        // y-direction
        if (Ay > 0)
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                rhs_array(i,j,k) -= sideCoeffy * (phi_array(i,j,k) - phi_array(i,j-1,k));
            });
        }
        else
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                rhs_array(i,j,k) -= sideCoeffy * (phi_array(i,j+1,k) - phi_array(i,j,k));
            });
        }
    }
}
