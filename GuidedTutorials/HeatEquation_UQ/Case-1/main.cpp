/*
 * A simplified single file version of the HeatEquation_EX0_C exmaple.
 * This code is designed to be used with Demo_Tutorial.rst.
 *
 */


#include <AMReX.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {

    // **********************************
    // DECLARE SIMULATION PARAMETERS
    // **********************************

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

    // **********************************
    // DECLARE PHYSICS PARAMETERS
    // **********************************

    // diffusion coefficient for heat equation
    Real diffusion_coeff;

    // amplitude of initial temperature profile
    Real init_amplitude;

    // width parameter controlling spread of initial profile (variance, not std dev)
    Real init_width;

    // **********************************
    // DECLARE DATALOG PARAMETERS
    // **********************************
    const int datwidth = 24;
    const int datprecision = 16;
    const int timeprecision = 13;
    int datalog_int = -1;      // Interval for regular output (<=0 means no regular output)
    std::string datalog_filename = "datalog.txt";

    // **********************************
    // READ PARAMETER VALUES FROM INPUT DATA
    // **********************************
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

        // Default datalog_int to -1, allow us to set it to something else in the inputs file
        //  If datalog_int < 0 then no plot files will be written
        datalog_int = -1;
        pp.query("datalog_int",datalog_int);

        datalog_filename = "datalog.txt";
        pp.query("datalog",datalog_filename);

        // **********************************
        // READ PHYSICS PARAMETERS
        // **********************************

        // Diffusion coefficient - controls how fast heat spreads
        diffusion_coeff = 1.0;
        pp.query("diffusion_coeff", diffusion_coeff);  // Note: input name is "diffusion" but variable is "diffusion_coeff"

        // Initial temperature amplitude
        init_amplitude = 1.0;
        pp.query("init_amplitude", init_amplitude);

        // Width parameter - this is the variance (width²), not standard deviation
        // Smaller values = more concentrated, larger values = more spread out
        init_width = 0.01;  // Note: 0.01 to match your original rsquared/0.01
        pp.query("init_width", init_width);
    }

    // **********************************
    // DEFINE SIMULATION SETUP AND GEOMETRY
    // **********************************

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    // number of points in the domain, and periodicity
    BoxArray ba;
    Geometry geom;

    // define lower and upper indices
    IntVect dom_lo(0,0,0);
    IntVect dom_hi(n_cell-1, n_cell-1, n_cell-1);

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box, [0,1] in each direction.
    RealBox real_box({ 0., 0., 0.},
                     { 1., 1., 1.});

    // periodic in all direction
    Array<int,3> is_periodic{1,1,1};

    // This defines a Geometry object
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,3> dx = geom.CellSizeArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate two phi multifabs; one will store the old state, the other the new.
    MultiFab phi_old(ba, dm, Ncomp, Nghost);
    MultiFab phi_new(ba, dm, Ncomp, Nghost);
    MultiFab phi_tmp(ba, dm, Ncomp, Nghost);

    // time = starting time in the simulation
    Real time = 0.0;

    // **********************************
    // INITIALIZE DATA LOOP
    // **********************************

    // loop over boxes
    for (MFIter mfi(phi_old); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& phiOld = phi_old.array(mfi);

        // **********************************
        // SET INITIAL TEMPERATURE PROFILE
        // **********************************
        // Formula: T = 1 + amplitude * exp(-r^2 / width^2)
        // where r is distance from center (0.5, 0.5, 0.5)
        //
        // Parameters:
        // - amplitude: controls peak temperature above baseline (1.0)
        // - width: controls spread of initial hot spot
        //   - smaller width = more concentrated
        //   - larger width = more spread out
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // **********************************
            // SET VALUES FOR EACH CELL
            // **********************************

            Real x = (i+0.5) * dx[0];
            Real y = (j+0.5) * dx[1];
            Real z = (k+0.5) * dx[2];
            Real rsquared = ((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5))/init_width;
            phiOld(i,j,k) = 1. + init_amplitude * std::exp(-rsquared);
        });
    }

    // **********************************
    // WRITE DATALOG FILE
    // **********************************
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream datalog(datalog_filename);  // truncate mode to start fresh
        datalog << "#" << std::setw(datwidth-1) << "     max_temp";
        datalog << std::setw(datwidth) << "    mean_temp";
        datalog << std::setw(datwidth) << "     std_temp";
        datalog << std::setw(datwidth) << "    cell_temp";
        datalog << std::endl;
        datalog.close();
    }

    // **********************************
    // WRITE INITIAL PLOT FILE
    // **********************************

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = Concatenate("plt",step,5);
        WriteSingleLevelPlotfile(pltfile, phi_old, {"phi"}, geom, time, 0);
    }


    // **********************************
    // MAIN TIME EVOLUTION LOOP
    // **********************************

    for (int step = 1; step <= nsteps; ++step)
    {
        // fill periodic ghost cells
        phi_old.FillBoundary(geom.periodicity());

        // new_phi = old_phi + dt * Laplacian(old_phi)
        // loop over boxes
        for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& phiOld = phi_old.array(mfi);
            const Array4<Real>& phiNew = phi_new.array(mfi);

            // advance the data by dt
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {

                // **********************************
                // EVOLVE VALUES FOR EACH CELL
                // **********************************

                phiNew(i,j,k) = phiOld(i,j,k) + dt * diffusion_coeff *
                    ( (phiOld(i+1,j,k) - 2.*phiOld(i,j,k) + phiOld(i-1,j,k)) / (dx[0]*dx[0])
                     +(phiOld(i,j+1,k) - 2.*phiOld(i,j,k) + phiOld(i,j-1,k)) / (dx[1]*dx[1])
                     +(phiOld(i,j,k+1) - 2.*phiOld(i,j,k) + phiOld(i,j,k-1)) / (dx[2]*dx[2])
                        );
            });
        }

        // find the value in cell (9,9,9)
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& phiOld = phi_old.array(mfi);
            const Array4<Real>& phiNew = phi_new.array(mfi);

            // advance the data by dt
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                if (i==9 && j==9 && k==9) {
                    return{phiNew(i,j,k)};
                } else {
                    return {0.};
                }
            });
        }

        Real cell_temperature = get<0>(reduce_data.value());
        ParallelDescriptor::ReduceRealSum(cell_temperature);

        // **********************************
        // INCREMENT
        // **********************************

        // update time
        time = time + dt;

        // copy new solution into old solution
        MultiFab::Copy(phi_old, phi_new, 0, 0, 1, 0);

        // Tell the I/O Processor to write out which step we're doing
        Print() << "Advanced step " << step << "\n";

        // **********************************
        // WRITE DATALOG AT GIVEN INTERVAL
        // **********************************

        // Check if we should write datalog
        bool write_datalog = false;
        if (step == nsteps) {
            write_datalog = true;  // Write final step
        } else if (datalog_int > 0 && step % datalog_int == 0) {
            write_datalog = true;  // Write every datalog_int steps
        }

        MultiFab::Copy(phi_tmp, phi_new, 0, 0, 1, 0);
        Real max_temperature = phi_new.max(0);
        Real mean_temperature = phi_new.sum(0) / phi_new.boxArray().numPts();
        phi_tmp.plus(-mean_temperature,0,1,0);
        Real std_temperature = phi_tmp.norm2(0); // compute sqrt( sum(phi_tmp_i^2) );

        if (write_datalog && ParallelDescriptor::IOProcessor()) {
            std::ofstream datalog(datalog_filename, std::ios::app);

            // Write 4 statistics
            datalog << std::setw(datwidth) << std::setprecision(datprecision) << max_temperature;
            datalog << std::setw(datwidth) << std::setprecision(datprecision) << mean_temperature;
            datalog << std::setw(datwidth) << std::setprecision(datprecision) << std_temperature;
            datalog << std::setw(datwidth) << std::setprecision(datprecision) << cell_temperature;
            datalog << std::endl;

            datalog.close();
        }

        // **********************************
        // WRITE PLOTFILE AT GIVEN INTERVAL
        // **********************************

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = Concatenate("plt",step,5);
            WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, step);
        }
    }


    }
    Finalize();
    return 0;
}
