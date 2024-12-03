CC   = clang
GCC  = cc
LINKER = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -fopenmp
#OPENMP   = -Xpreprocessor -fopenmp #required on Macos with homebrew libomp
LIBS     = # -lomp
endif

VERSION  = --version
# CFLAGS   = -O3 -std=c17 $(OPENMP)
CFLAGS   = -Ofast -std=c17
#CFLAGS   = -Ofast -fnt-store=aggressive  -std=c99 $(OPENMP) #AMD CLANG
LFLAGS   = $(OPENMP) -lm
DEFINES  = -D_GNU_SOURCE# -DDEBUG
INCLUDES =
