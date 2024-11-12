#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "timing.h"

#define ARRAY_ALIGNMENT 64 // Define ARRAY_ALIGNMENT with an appropriate value

extern double dmvm(double *restrict y, const double *restrict a,
                   const double *restrict x, int N, int iter);

int sizeOfRank(int rank, int size, int N) {
  return (N / size) + ((N % size > rank) ? 1 : 0);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t bytesPerWord = sizeof(double);
  size_t N = 0;
  size_t iter = 1;
  double *a, *x, *y;
  double t0, t1;
  double walltime;

  if (argc > 2) {
    N = atoi(argv[1]);
    iter = atoi(argv[2]);
  } else {
    if (rank == 0) {
      printf("Usage: %s <N> <iter>\n", argv[0]);
    }
    MPI_Finalize();
    exit(EXIT_SUCCESS);
  }

  int Nlocal = sizeOfRank(rank, size, N);
  a = (double *)allocate(ARRAY_ALIGNMENT, Nlocal * N * bytesPerWord);
  x = (double *)allocate(ARRAY_ALIGNMENT, N * bytesPerWord);
  y = (double *)allocate(ARRAY_ALIGNMENT, Nlocal * bytesPerWord);

  // Initialize arrays
  if (rank == 0) {
    double *full_a = (double *)allocate(ARRAY_ALIGNMENT, N * N * bytesPerWord);
    double *full_x = (double *)allocate(ARRAY_ALIGNMENT, N * bytesPerWord);

    for (int i = 0; i < N; i++) {
      full_x[i] = (double)i;
      for (int j = 0; j < N; j++) {
        full_a[i * N + j] = (double)j + i;
      }
    }

    MPI_Scatter(full_a, Nlocal * N, MPI_DOUBLE, a, Nlocal * N, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Bcast(full_x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(full_a);
    free(full_x);
  } else {
    MPI_Scatter(NULL, Nlocal * N, MPI_DOUBLE, a, Nlocal * N, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  int upperNeighbor = (rank - 1 + size) % size;
  int lowerNeighbor = (rank + 1) % size;
  int num = N / size;
  int rest = N % size;
  int cs = rank * num + (rest > rank ? rank : rest);
  int Ncurrent = Nlocal;

  // Perform computation with ring shift
  t0 = MPI_Wtime();
  for (int k = 0; k < iter; k++) {
    for (int rot = 0; rot < size; rot++) {
      for (int r = 0; r < Nlocal; r++) {
        for (int c = cs; c < cs + Ncurrent; c++) {
          y[r] += a[r * N + c] * x[c - cs];
        }
      }
      cs += Ncurrent;
      if (cs >= N)
        cs = 0; // wrap around
      Ncurrent = sizeOfRank(upperNeighbor, size, N);
      if (rot != size - 1) {
        MPI_Status status;
        MPI_Sendrecv_replace(x, num + (rest > 0 ? 1 : 0), MPI_DOUBLE,
                             upperNeighbor, 0, lowerNeighbor, 0, MPI_COMM_WORLD,
                             &status);
      }
    }
  }
  t1 = MPI_Wtime();
  walltime = t1 - t0;

  double *full_y = NULL;
  if (rank == 0) {
    full_y = (double *)allocate(ARRAY_ALIGNMENT, N * bytesPerWord);
  }

  MPI_Gather(y, Nlocal, MPI_DOUBLE, full_y, Nlocal, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    double flops = (double)2.0 * N * N * iter;
    printf("%zu %zu %.2f %.2f\n", iter, N, 1.0E-06 * flops / walltime,
           walltime);
    free(full_y);
  }

  free(a);
  free(x);
  free(y);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
