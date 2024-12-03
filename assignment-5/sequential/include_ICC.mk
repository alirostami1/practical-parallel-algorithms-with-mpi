CC   = icc
GCC  = gcc
LINKER = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -qopenmp
endif

VERSION  = --version
CFLAGS   =  -O3 -xHost -qopt-zmm-usage=high -std=c99 $(OPENMP)
LFLAGS   = $(OPENMP)
DEFINES  = -D_GNU_SOURCE
INCLUDES =
LIBS     =
