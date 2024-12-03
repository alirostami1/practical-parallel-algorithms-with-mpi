CC   = clang
GCC  = cc
LINKER = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -fopenmp
#OPENMP   = -Xpreprocessor -fopenmp #required on Macos with homebrew libomp
LIBS     = # -lomp
endif

VERSION  = --version
CFLAGS   = -Ofast -std=c99 $(OPENMP)
#CFLAGS   = -Ofast -fnt-store=aggressive  -std=c99 $(OPENMP) #AMD CLANG
LFLAGS   = $(OPENMP)
DEFINES  = -D_GNU_SOURCE# -DDEBUG
INCLUDES = -I/usr/local/include
