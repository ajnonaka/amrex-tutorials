# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../../../amrex

# For Python support, clone pybind11 as sibling to amrex:
#   git clone -b v3.0.0 https://github.com/pybind/pybind11.git
# Or set PYBIND11_HOME to point to your pybind11 installation
PYBIND11_HOME ?= ../../../../pybind11

DEBUG     = FALSE
USE_MPI   = TRUE
USE_OMP   = FALSE
COMP      = gcc
DIM       = 3

USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

# Python main control flag
ifndef USE_PYTHON_MAIN
  USE_PYTHON_MAIN = FALSE
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

# Add pybind11 and Python includes when building for Python
ifeq ($(USE_PYTHON_MAIN),TRUE)
  INCLUDE_LOCATIONS += $(PYBIND11_HOME)/include

  # Add Python headers
  #PYTHON_CONFIG ?= python3-config

  # Add Python includes
  #PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes)
  #CXXFLAGS += $(PYTHON_INCLUDES)

  # Add Python linking flags - try both methods
  #PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed 2>/dev/null || $(PYTHON_CONFIG) --ldflags)
  #PYTHON_LIBS := $(shell $(PYTHON_CONFIG) --libs --embed 2>/dev/null || $(PYTHON_CONFIG) --libs)

  #LDFLAGS += $(PYTHON_LDFLAGS) $(PYTHON_LIBS)
endif

# Add PIC flag when building for Python
ifeq ($(USE_PYTHON_MAIN),TRUE)
  XTRA_CXXFLAGS += -fPIC
ifeq ($(USE_OMP),TRUE)
  LDFLAGS += -fopenmp
endif
endif

include Make.package
VPATH_LOCATIONS  += .
INCLUDE_LOCATIONS += .

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

# Python library building (using pybind11 naming convention)
# Python library building (using pybind11 naming convention)
ifeq ($(USE_PYTHON_MAIN),TRUE)

# Add pybind11 and Python configuration
INCLUDE_LOCATIONS += $(PYBIND11_HOME)/include
PYTHON_CONFIG ?= python3-config
CXXFLAGS += $(shell $(PYTHON_CONFIG) --includes)
CXXFLAGS += -fPIC

# Python linking flags for the shared library
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed 2>/dev/null || $(PYTHON_CONFIG) --ldflags)
PYTHON_LIBS := $(shell $(PYTHON_CONFIG) --libs --embed 2>/dev/null || $(PYTHON_CONFIG) --libs)

# In WarpX this is machine-dependent
SHARED_OPTION = -shared

# Pybind11-style naming
PYMODULE_NAME = amrex_heat
PYDIM = $(DIM)d

# Build both main executable and Python library
all: $(main_exe) lib$(PYMODULE_NAME).$(PYDIM).so

installamrex_heat: lib$(PYMODULE_NAME).$(PYDIM).so
	cp lib$(PYMODULE_NAME).$(PYDIM).so Python/py$(PYMODULE_NAME)
	cd Python; python3 setup.py install --force --with-lib$(PYMODULE_NAME) $(PYDIM) $(PYINSTALLOPTIONS)

lib$(PYMODULE_NAME).$(PYDIM).a: $(objForExecs)
	@echo Making static library $@ ...
	$(SILENT) $(AR) -crv $@ $^
	$(SILENT) $(RM) AMReX_buildInfo.cpp
	@echo SUCCESS

lib$(PYMODULE_NAME).$(PYDIM).so: $(objForExecs)
	@echo Making dynamic library $@ ...
ifeq ($(USE_CUDA),TRUE)
	$(SILENT) $(CXX) $(LINKFLAGS) $(SHARED_OPTION) -Xlinker=$(SHARED_OPTION) -Xlinker=-fPIC -o $@ $^ $(LDFLAGS) $(libraries) $(PYTHON_LDFLAGS) $(PYTHON_LIBS)
else
	$(SILENT) $(CXX) $(LINKFLAGS) $(SHARED_OPTION) -fPIC -o $@ $^ $(LDFLAGS) $(libraries) $(PYTHON_LDFLAGS) $(PYTHON_LIBS)
endif
	$(SILENT) $(RM) AMReX_buildInfo.cpp
	@echo SUCCESS

clean::
	$(SILENT) $(RM) -rf build
	$(SILENT) $(RM) -f lib$(PYMODULE_NAME).?d.a
	$(SILENT) $(RM) -f lib$(PYMODULE_NAME).?d.so
	$(SILENT) $(RM) -f py$(PYMODULE_NAME)/lib$(PYMODULE_NAME).?d.so
	$(SILENT) $(RM) -f $(PYMODULE_NAME).*.so  # Clean pybind11-style names too

else
# When USE_PYTHON_MAIN=FALSE, only build main executable
# (existing AMReX build system handles this)
endif

# Convenience targets
.PHONY: python_lib python_install inputs

python_lib:
	$(MAKE) USE_PYTHON_MAIN=TRUE lib$(PYMODULE_NAME).$(DIM)d.so

python_install:
	$(MAKE) USE_PYTHON_MAIN=TRUE installamrex_heat

inputs:
	cp -r inputs/* .
