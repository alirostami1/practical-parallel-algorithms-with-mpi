CC   = mpiicx
GCC  = gcc
LINKER = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -qopenmp
endif

VERSION  = --version
CFLAGS   =  -O3 -xHost -qopt-zmm-usage=high -std=c99 $(OPENMP) -Wno-unused-command-line-argument
LFLAGS   = $(OPENMP)
DEFINES  = -D_GNU_SOURCE -DVERBOSE
INCLUDES =
LIBS     =
