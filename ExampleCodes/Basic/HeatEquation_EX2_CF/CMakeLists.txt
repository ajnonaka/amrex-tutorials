if ( NOT AMReX_FINTERFACES OR ( AMReX_SPACEDIM EQUAL 1) )
   return()
endif ()

# List of source files
set(_sources main.cpp advance.cpp myfunc_F.H myfunc.H init_phi_${AMReX_SPACEDIM}d.f90 advance_${AMReX_SPACEDIM}d.f90)
list(TRANSFORM _sources PREPEND "Source/")

# List of input files
file( GLOB_RECURSE _input_files LIST_DIRECTORIES false Exec/input* )

setup_tutorial(_sources _input_files)

unset( _sources )
unset( _input_files   )
