/*
 * Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocate.h"
#include "parameter.h"
#include "solver.h"
#include "util.h"

#define P(i, j) p[(j) * (imaxLocal + 2) + (i)]
#define F(i, j) f[(j) * (imaxLocal + 2) + (i)]
#define G(i, j) g[(j) * (imaxLocal + 2) + (i)]
#define U(i, j) u[(j) * (imaxLocal + 2) + (i)]
#define V(i, j) v[(j) * (imaxLocal + 2) + (i)]
#define RHS(i, j) rhs[(j) * (imaxLocal + 2) + (i)]
#define Pall(i, j) Pall[(j) * (imax + 2) + (i)]
#define Uall(i, j) Uall[(j) * (imax + 2) + (i)]
#define Vall(i, j) Vall[(j) * (imax + 2) + (i)]
#define Root 0

static int sizeOfRank(int rank, int size, int N) {
  return N / size + ((N % size > rank) ? 1 : 0);
}

/* Use this function to check if your exchange function works or not. */
/* Assigns every ranks local pressure array with rank number.         */
/* After exchange, you should see your neighbours rank in ghost cells */
/* of the local pressure array.                                       */
static void printExchange(Solver *solver, double *grid) {
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;
  double *p = grid;
  for (int j = 0; j < jmaxLocal + 2; j++) {
    for (int i = 0; i < imaxLocal + 2; i++) {
      P(i, j) = solver->rank;
    }
  }

  exchange(solver, p);

  for (int i = 0; i < solver->size; i++) {
    if (i == solver->rank) {
      printf("### RANK %d "
             "#######################################################\n",
             solver->rank);
      for (int j = 0; j < jmaxLocal + 2; j++) {
        printf("%02d: ", j);
        for (int i = 0; i < imaxLocal + 2; i++) {
          printf("%12.2f  ", grid[j * (imaxLocal + 2) + i]);
        }
        printf("\n");
      }
      fflush(stdout);
    }
    MPI_Barrier(solver->comm);
  }
}

/* Use this function to check if your shift function works or not. */
/* Assigns every ranks local pressure array with rank number.      */
/* After shift, you should see your neighbours rank in ghost cells */
/* of the local F and G array.                                     */
static void printShift(Solver *solver) {
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;
  double *f = solver->f;
  double *g = solver->g;

  for (int j = 0; j < jmaxLocal + 2; j++) {
    for (int i = 0; i < imaxLocal + 2; i++) {
      F(i, j) = solver->rank;
      G(i, j) = solver->rank;
    }
  }

  shift(solver);

  printf("Vector F\n");
  for (int i = 0; i < solver->size; i++) {
    if (i == solver->rank) {
      printf("### RANK %d "
             "#######################################################\n",
             solver->rank);
      for (int j = 0; j < jmaxLocal + 2; j++) {
        printf("%02d: ", j);
        for (int i = 0; i < imaxLocal + 2; i++) {
          printf("%12.2f  ", F(i, j));
        }
        printf("\n");
      }
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  printf("Vector G\n");
  for (int i = 0; i < solver->size; i++) {
    if (i == solver->rank) {
      printf("### RANK %d "
             "#######################################################\n",
             solver->rank);
      for (int j = 0; j < jmaxLocal + 2; j++) {
        printf("%02d: ", j);
        for (int i = 0; i < imaxLocal + 2; i++) {
          printf("%12.2f  ", G(i, j));
        }
        printf("\n");
      }
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

/* subroutines local to this module */
static int sum(int *sizes, int init, int offset, int coord) {
  int sum = 0;

  for (int i = init - offset; coord > 0; i -= offset, --coord) {
    sum += sizes[i];
  }

  return sum;
}

void exchange(Solver *solver, double *grid) {
  /* TODO ghost cell exchange with respective neighbors   */
  /* Remember this is a 2D domain decomposition and you   */
  /* have your displacements in solver->sdispls and       */
  /* solver->rdispls. Use solver->bufferTypes to indicate */
  /* datatype to send. You have initialized them during   */
  /* initSolver() call and you can use them directly here.*/
  /* Refer to lecture slides for more information.        */
#ifdef DEBUG
  double *u = grid;
  int imaxLocal = solver->imaxLocal;
  // Print ghost cells before exchange
  for (int i = 0; i < solver->imaxLocal + 2; i++) {
    printf("Rank %d: Before exchange, ghost cell U[%d][0] = %f\n", solver->rank,
           i, U(i, 0));
  }
#endif
  int counts[NDIRS] = {1, 1, 1, 1};
  MPI_Neighbor_alltoallw(grid, counts, solver->sdispls, solver->bufferTypes,
                         grid, counts, solver->rdispls, solver->bufferTypes,
                         solver->comm);
#ifdef DEBUG
  // Print ghost cells after exchange
  for (int i = 0; i < solver->imaxLocal + 2; i++) {
    printf("Rank %d: After exchange, ghost cell U[%d][0] = %f\n", solver->rank,
           i, U(i, 0));
  }
#endif
}

void shift(Solver *solver) {
  /* TODO shift (send) f ghost layer from left to right */
  /* TODO shift (send) g ghost layer from bottom to top */
  double *f = solver->f;
  double *g = solver->g;

  MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                             MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  /* shift G                                              */
  /* receive ghost cells from bottom neighbor             */
  /* use solver->bufferTypes and solver->neighbours array */
  /* to receive from BOTTOM rank                          */
  double *buf = g + 1;
  if (solver->neighbours[BOTTOM] != MPI_PROC_NULL) {
    MPI_Irecv(g, 1, solver->bufferTypes[BOTTOM], solver->neighbours[BOTTOM], 0,
              solver->comm, &requests[0]);
  }

  /* send ghost cells to top neighbor                     */
  /* use solver->bufferTypes and solver->neighbours array */
  /* to send to TOP rank                                  */
  buf = g + (solver->jmaxLocal) * (solver->imaxLocal + 2) + 1;
  if (solver->neighbours[TOP] != MPI_PROC_NULL) {
    MPI_Isend(g + solver->jmaxLocal * (solver->imaxLocal + 2), 1,
              solver->bufferTypes[TOP], solver->neighbours[TOP], 0,
              solver->comm, &requests[1]);
  }

  /* shift F                                              */
  /* receive ghost cells from left neighbor               */
  /* use solver->bufferTypes and solver->neighbours array */
  /* to receive from LEFT rank                            */
  buf = f + (solver->imaxLocal + 2);
  if (solver->neighbours[LEFT] != MPI_PROC_NULL) {
    MPI_Irecv(f, 1, solver->bufferTypes[LEFT], solver->neighbours[LEFT], 0,
              solver->comm, &requests[2]);
  }

  /* send ghost cells to right neighbor                   */
  /* use solver->bufferTypes and solver->neighbours array */
  /* to send to RIGHT rank                                */
  buf = f + (solver->imaxLocal + 2) + (solver->imaxLocal);
  if (solver->neighbours[RIGHT] != MPI_PROC_NULL) {
    MPI_Isend(f + solver->imaxLocal, 1, solver->bufferTypes[RIGHT],
              solver->neighbours[RIGHT], 0, solver->comm, &requests[3]);
  }

  MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

static int isBoundary(Solver *solver, int direction) {
  /* Use solver->coords to check if the rank has boundaries */
  /* Fill in the rest of the boundaries. One example given  */
  switch (direction) {
  case LEFT:
    return solver->coords[IDIM] == 0; // Left boundary: X-coordinate is 0
  case RIGHT:
    return solver->coords[IDIM] == solver->dims[IDIM] - 1; // Right boundary
  case BOTTOM:
    return solver->coords[JDIM] == 0; // Bottom boundary: Y-coordinate is 0
  case TOP:
    return solver->coords[JDIM] == solver->dims[JDIM] - 1; // Top boundary
  }
  return 1;
}

static void assembleResult(Solver *solver, double *src, double *dst) {
  /* do not adapt these to local imaxLocal and jmaxLocal */
  int imax = solver->imax;
  int jmax = solver->jmax;

  MPI_Request *requests;
  int numRequests = 1;

  if (solver->rank == 0) {
    numRequests = solver->size + 1;
  } else {
    numRequests = 1;
  }

  requests = (MPI_Request *)malloc(numRequests * sizeof(MPI_Request));

  /* all ranks send their bulk array, including the external boundary layer */
  MPI_Datatype bulkType;
  int oldSizes[NDIMS] = {solver->jmaxLocal + 2, solver->imaxLocal + 2};
  int newSizes[NDIMS] = {solver->jmaxLocal, solver->imaxLocal};
  int starts[NDIMS] = {1, 1};

  if (isBoundary(solver, LEFT)) {
    newSizes[CIDIM] += 1;
    starts[CIDIM] = 0;
  }
  if (isBoundary(solver, RIGHT)) {
    newSizes[CIDIM] += 1;
  }
  if (isBoundary(solver, BOTTOM)) {
    newSizes[CJDIM] += 1;
    starts[CJDIM] = 0;
  }
  if (isBoundary(solver, TOP)) {
    newSizes[CJDIM] += 1;
  }

  /* Refer to lecture slides for more information                       */
  /* https://rookiehpc.org/mpi/docs/mpi_type_create_subarray/index.html */
  /* Create a subarray in bulkType object declared above                */
  /* Use oldSizes, newSizes and starts defined above                    */
  MPI_Type_create_subarray(NDIMS, oldSizes, newSizes, starts, MPI_ORDER_C,
                           MPI_DOUBLE, &bulkType);
  MPI_Type_commit(&bulkType);

  /* Send the data using bulkType subarray to rank 0    */
  MPI_Isend(src, 1, bulkType, Root, 0, MPI_COMM_WORLD, &requests[0]);

  int newSizesI[solver->size];
  int newSizesJ[solver->size];
  MPI_Gather(&newSizes[CIDIM], 1, MPI_INT, newSizesI, 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&newSizes[CJDIM], 1, MPI_INT, newSizesJ, 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  /* rank 0 assembles the subdomains */
  if (solver->rank == 0) {
    for (int i = 0; i < solver->size; i++) {
      MPI_Datatype domainType;
      int oldSizes[NDIMS] = {jmax + 2, imax + 2};
      int newSizes[NDIMS] = {newSizesJ[i], newSizesI[i]};
      int coords[NDIMS];
      MPI_Cart_coords(solver->comm, i, NDIMS, coords);
      int starts[NDIMS] = {sum(newSizesJ, i, 1, coords[JDIM]),
                           sum(newSizesI, i, solver->dims[JDIM], coords[IDIM])};
      printf("Rank: %d, Coords(i,j): %d %d, Size(i,j): %d %d, Target "
             "Size(i,j): %d %d "
             "Starts(i,j): %d %d\n",
             i, coords[IDIM], coords[JDIM], oldSizes[CIDIM], oldSizes[CJDIM],
             newSizes[CIDIM], newSizes[CJDIM], starts[CIDIM], starts[CJDIM]);

      /* Refer to lecture slides for more information                       */
      /* Create a subarray in domainType object declared above              */
      /* Use oldSizes, newSizes and starts defined above                    */
      MPI_Type_create_subarray(NDIMS, oldSizes, newSizes, starts, MPI_ORDER_C,
                               MPI_DOUBLE, &domainType);

      MPI_Type_commit(&domainType);

      /* Recv the data using domainType subarray from rest of the ranks */
      MPI_Irecv(dst, 1, domainType, i, 0, MPI_COMM_WORLD, &requests[i + 1]);
      MPI_Type_free(&domainType);
#ifdef DEBUG
      printf("Rank %d: after Irecv\n", solver->rank);
#endif
    }
  }

  MPI_Waitall(numRequests, requests, MPI_STATUSES_IGNORE);
}

void collectResult(Solver *solver) {
  size_t bytesize = (solver->imax + 2) * (solver->jmax + 2) * sizeof(double);

  double *Pall = allocate(64, bytesize);
  double *Uall = allocate(64, bytesize);
  double *Vall = allocate(64, bytesize);
#ifdef DEBUG
  printf("rank = %d, after allocation\n", solver->rank);
#endif
  /* collect P */
  assembleResult(solver, solver->p, Pall);
#ifdef DEBUG
  printf("rank = %d, after assembleResult p\n", solver->rank);
#endif

  /* collect U */
  assembleResult(solver, solver->u, Uall);
#ifdef DEBUG
  printf("rank = %d, after assembleResult u\n", solver->rank);
#endif

  /* collect V */
  assembleResult(solver, solver->v, Vall);
#ifdef DEBUG
  printf("rank = %d, after assembleResult v\n", solver->rank);
#endif

  if (solver->rank == 0) {
    writeResult(solver, Pall, Uall, Vall);
  }

  free(Pall);
  free(Uall);
  free(Vall);
}

static void printConfig(Solver *solver) {
  if (solver->rank == 0) {
    printf("Parameters for #%s#\n", solver->problem);
    printf("Boundary conditions Left:%d Right:%d Bottom:%d Top:%d\n",
           solver->bcLeft, solver->bcRight, solver->bcBottom, solver->bcTop);
    printf("\tReynolds number: %.2f\n", solver->re);
    printf("\tGx Gy: %.2f %.2f\n", solver->gx, solver->gy);
    printf("Geometry data:\n");
    printf("\tDomain box size (x, y): %.2f, %.2f\n", solver->xlength,
           solver->ylength);
    printf("\tCells (x, y): %d, %d\n", solver->imax, solver->jmax);
    printf("Timestep parameters:\n");
    printf("\tDefault stepsize: %.2f, Final time %.2f\n", solver->dt,
           solver->te);
    printf("\tdt bound: %.6f\n", solver->dtBound);
    printf("\tTau factor: %.2f\n", solver->tau);
    printf("Iterative solver parameters:\n");
    printf("\tMax iterations: %d\n", solver->itermax);
    printf("\tepsilon (stopping tolerance) : %f\n", solver->eps);
    printf("\tgamma factor: %f\n", solver->gamma);
    printf("\tomega (SOR relaxation): %f\n", solver->omega);
    printf("Communication parameters:\n");
  }
  for (int i = 0; i < solver->size; i++) {
    if (i == solver->rank) {
      printf("\tRank %d of %d\n", solver->rank, solver->size);
      printf("\tNeighbours (bottom, top, left, right): %d %d, %d, %d\n",
             solver->neighbours[BOTTOM], solver->neighbours[TOP],
             solver->neighbours[LEFT], solver->neighbours[RIGHT]);
      printf("\tIs boundary:\n");
      printf("\t\tLEFT: %d\n", isBoundary(solver, LEFT));
      printf("\t\tRIGHT: %d\n", isBoundary(solver, RIGHT));
      printf("\t\tBOTTOM: %d\n", isBoundary(solver, BOTTOM));
      printf("\t\tTOP: %d\n", isBoundary(solver, TOP));
      printf("\tCoordinates (i,j) %d %d\n", solver->coords[IDIM],
             solver->coords[JDIM]);
      printf("\tDims (i,j) %d %d\n", solver->dims[IDIM], solver->dims[JDIM]);
      printf("\tLocal domain size (i,j) %dx%d\n", solver->imaxLocal,
             solver->jmaxLocal);
      fflush(stdout);
    }
    MPI_Barrier(solver->comm);
  }
}

void initSolver(Solver *solver, Parameter *params) {
  MPI_Comm_rank(MPI_COMM_WORLD, &(solver->rank));
  MPI_Comm_size(MPI_COMM_WORLD, &(solver->size));
  solver->problem = params->name;
  solver->bcLeft = params->bcLeft;
  solver->bcRight = params->bcRight;
  solver->bcBottom = params->bcBottom;
  solver->bcTop = params->bcTop;
  solver->imax = params->imax;
  solver->jmax = params->jmax;
  solver->xlength = params->xlength;
  solver->ylength = params->ylength;
  solver->dx = params->xlength / params->imax;
  solver->dy = params->ylength / params->jmax;
  solver->eps = params->eps;
  solver->omega = params->omg;
  solver->itermax = params->itermax;
  solver->re = params->re;
  solver->gx = params->gx;
  solver->gy = params->gy;
  solver->dt = params->dt;
  solver->te = params->te;
  solver->tau = params->tau;
  solver->gamma = params->gamma;

  int imax = solver->imax;
  int jmax = solver->jmax;

  /* Use virtual topologies to get dimension and coordinates of the rank */
  int dims[NDIMS] = {0, 0};
  int periods[NDIMS] = {0, 0};
  int coords[NDIMS] = {0, 0};

  /* Retrieves best possible #ranks in X and Y dimensions from the given
   * communicator size. */
  /* Say for 6 ranks, best possible #ranks in in X and Y dimension would be 3
   * and 2         */
  /* Store in dims array */
  /* https://rookiehpc.org/mpi/docs/mpi_dims_create/index.html */
  MPI_Dims_create(solver->size, NDIMS, dims); // Auto-calculate grid dimensions
  memcpy(&solver->dims, &dims, NDIMS * sizeof(int));

  /* Create a new communicator based on the dimensions             */
  /* Store the new communicator in &solver->comm                   */
  /* Use NDIMS to specify dimension size                           */
  /* https://rookiehpc.org/mpi/docs/mpi_cart_create/index.html     */
  MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, 1, &solver->comm);

  /* Use newly created communicator solver->comm                   */
  /* For left and right neighbors                                  */
  /* Store in solver->neighbors[LEFT] and solver->neighbors[RIGHT] */
  /* https://rookiehpc.org/mpi/docs/mpi_cart_shift/index.html      */
  MPI_Cart_shift(solver->comm, 0, 1, &solver->neighbours[LEFT],
                 &solver->neighbours[RIGHT]);

  /* Use newly created communicator solver->comm                   */
  /* For bottom and top neighbors                                  */
  /* Store in solver->neighbors[BOTTOM] and solver->neighbors[TOP] */
  MPI_Cart_shift(solver->comm, 1, 1, &solver->neighbours[BOTTOM],
                 &solver->neighbours[TOP]);

  /* Retrieve the cartesion topology of the rank in solver->coords */
  /* https://rookiehpc.org/mpi/docs/mpi_cart_get/index.html        */
  MPI_Cart_get(solver->comm, NDIMS, dims, periods, coords);
  memcpy(&solver->coords, &coords, NDIMS * sizeof(int));

  solver->imaxLocal = sizeOfRank(solver->coords[IDIM], dims[IDIM], imax);
  solver->jmaxLocal = sizeOfRank(solver->coords[JDIM], dims[JDIM], jmax);

  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;
  size_t bytesize = (imaxLocal + 2) * (jmaxLocal + 2) * sizeof(double);

  MPI_Datatype row_type, col_type;

#ifdef DEBUG
  printf("Rank %d: imaxLocal %d, jmaxLocal %d, dims[IDIM] %d, dims[JDIM] %d\n",
         solver->rank, imaxLocal, jmaxLocal, dims[IDIM], dims[JDIM]);
#endif

  // Contiguous row type for horizontal ghost layers
  MPI_Type_contiguous(solver->imaxLocal, MPI_DOUBLE, &row_type);
  MPI_Type_commit(&row_type);

  // Vector type for vertical ghost layers
  MPI_Type_vector(solver->jmaxLocal, 1, solver->imaxLocal + 2, MPI_DOUBLE,
                  &col_type);
  MPI_Type_commit(&col_type);

  solver->bufferTypes[LEFT] = col_type;
  solver->bufferTypes[RIGHT] = col_type;
  solver->bufferTypes[BOTTOM] = row_type;
  solver->bufferTypes[TOP] = row_type;

  size_t dblsize = sizeof(double);
  solver->sdispls[LEFT] = ((imaxLocal + 2) + 1) * dblsize;
  solver->sdispls[RIGHT] = ((imaxLocal + 2) + imaxLocal) * dblsize;
  solver->sdispls[BOTTOM] = ((imaxLocal + 2) + 1) * dblsize;
  solver->sdispls[TOP] = ((jmaxLocal) * (imaxLocal + 2) + 1) * dblsize;
  solver->rdispls[LEFT] = (imaxLocal + 2) * dblsize;
  solver->rdispls[RIGHT] = ((imaxLocal + 2) + (imaxLocal + 1)) * dblsize;
  solver->rdispls[BOTTOM] = 1 * dblsize;
  solver->rdispls[TOP] = ((jmaxLocal + 1) * (imaxLocal + 2) + 1) * dblsize;

  solver->u = allocate(64, bytesize);
  solver->v = allocate(64, bytesize);
  solver->p = allocate(64, bytesize);
  solver->rhs = allocate(64, bytesize);
  solver->f = allocate(64, bytesize);
  solver->g = allocate(64, bytesize);

  /* TODO adapt to local size imaxLocal & jmaxLocal */
  for (int j = 0; j < jmaxLocal + 2; j++) {
    for (int i = 0; i < imaxLocal + 2; i++) {
      int idx = j * (imaxLocal + 2) + i;
      solver->u[idx] = params->u_init;
      solver->v[idx] = params->v_init;
      solver->p[idx] = params->p_init;
      solver->rhs[idx] = 0.0;
      solver->f[idx] = 0.0;
      solver->g[idx] = 0.0;
    }
  }

#ifdef DEBUG
  double *u = solver->u;
  double *v = solver->v;
  double *p = solver->p;
  for (int j = 0; j < solver->jmaxLocal + 2; j++) {
    for (int i = 0; i < solver->imaxLocal + 2; i++) {
      printf("Rank %d: u[%d][%d] = %f\n", solver->rank, i, j, U(i, j));
      printf("Rank %d: v[%d][%d] = %f\n", solver->rank, i, j, V(i, j));
      printf("Rank %d: p[%d][%d] = %f\n", solver->rank, i, j, P(i, j));
    }
  }
#endif

  double dx = solver->dx;
  double dy = solver->dy;
  double invSqrSum = 1.0 / (dx * dx) + 1.0 / (dy * dy);
  solver->dtBound = 0.5 * solver->re * 1.0 / invSqrSum;

#ifdef VERBOSE
  printConfig(solver);
#endif
}

void computeRHS(Solver *solver) {
  int imax = solver->imax;
  int jmax = solver->jmax;
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;
  double idx = 1.0 / solver->dx;
  double idy = 1.0 / solver->dy;
  double idt = 1.0 / solver->dt;
  double *rhs = solver->rhs;
  double *f = solver->f;
  double *g = solver->g;

  /* Shift function is similar to exchange. Just that you need to exchange
   * values for F and G array */
  /* A simple Send Recv function for F and G array can be implemented. */
  /* You can look at the formulation below to guess which values will be
   * required. */
  /* For F, you just have to send the values from left neighbor to right
   * neighbor's ghost cells */
  /* For G, you just have to send the values from bottom neighbor to top
   * neighbor's ghost cells */

  shift(solver);

  /* TODO adapt to local size imaxLocal & jmaxLocal */
  for (int j = 1; j <= jmaxLocal; j++) {
    for (int i = 1; i <= imaxLocal; i++) {
      RHS(i, j) =
          ((F(i, j) - F(i - 1, j)) * idx + (G(i, j) - G(i, j - 1)) * idy) * idt;
    }
  }
}

void solve(Solver *solver) {
  int imax = solver->imax;
  int jmax = solver->jmax;
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;
  double eps = solver->eps;
  int itermax = solver->itermax;
  double dx2 = solver->dx * solver->dx;
  double dy2 = solver->dy * solver->dy;
  double idx2 = 1.0 / dx2;
  double idy2 = 1.0 / dy2;
  double factor = solver->omega * 0.5 * (dx2 * dy2) / (dx2 + dy2);
  double *p = solver->p;
  double *rhs = solver->rhs;
  double epssq = eps * eps;
  int it = 0;
  double res = 1.0;

  while ((res >= epssq) && (it < itermax)) {
    res = 0.0;

    /* Exchange the ghost cells with respective neighbors */
    /* Remember this is a 2D domain decomposition.        */
    exchange(solver, p);

    /* TODO adapt to local size imaxLocal & jmaxLocal     */
    for (int j = 1; j <= jmaxLocal; j++) {
      for (int i = 1; i <= imaxLocal; i++) {

        double r =
            RHS(i, j) - ((P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) * idx2 +
                         (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) * idy2);

        P(i, j) -= (factor * r);
        res += (r * r);
      }
    }

    /* Extend boundary condition handling from Poisson assignment   */
    /* use isBoundary(solver, LEFT/RIGHT/TOP/BOTTOM) to check       */
    /* whether a rank has boundary and apply BC only for those rank */
    /* Use indices for more intuition.                              */
    if (isBoundary(solver, LEFT)) {
      for (int j = 1; j <= jmaxLocal; j++) {
        P(0, j) = P(1, j); // Copy interior boundary to left ghost cells
      }
    }
    if (isBoundary(solver, RIGHT)) {
      for (int j = 1; j <= jmaxLocal; j++) {
        P(imaxLocal + 1, j) = P(imaxLocal, j); // Right ghost cells
      }
    }
    if (isBoundary(solver, BOTTOM)) {
      for (int i = 1; i <= imaxLocal; i++) {
        P(i, 0) = P(i, 1); // Bottom ghost cells
      }
    }
    if (isBoundary(solver, TOP)) {
      for (int i = 1; i <= imaxLocal; i++) {
        P(i, jmaxLocal + 1) = P(i, jmaxLocal); // Top ghost cells
      }
    }

    /* TODO add global MPI_SUM reduction for res */
    double globalRes = 0.0;
    MPI_Allreduce(&res, &globalRes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    res = globalRes / (double)(imax * jmax);
    it++;
  }

#ifdef VERBOSE
  if (solver->rank == 0)
    printf("Solver took %d iterations to reach %f\n", it, sqrt(res));
#endif
}

static double maxElement(Solver *solver, double *m) {
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;

  // Compute local maximum
  double maxval = DBL_MIN;
  for (int j = 1; j <= jmaxLocal; j++) {   // Local rows
    for (int i = 1; i <= imaxLocal; i++) { // Local columns
      maxval = MAX(maxval, fabs(m[j * (imaxLocal + 2) + i]));
    }
  }

  // Perform global reduction to find the maximum value across all ranks
  double globalMax = 0.0;
  MPI_Allreduce(&maxval, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return globalMax;
}

void normalizePressure(Solver *solver) {
  /* TODO adapt to local size imaxLocal & jmaxLocal */
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;
  int size = (imaxLocal + 2) * (jmaxLocal + 2); // Adapt size to local subdomain
  double *p = solver->p;
  double avgP = 0.0;

  // Compute local sum of pressure
  for (int i = 0; i < size; i++) {
    avgP += p[i];
  }

  /* TODO add global MPI_SUM reduction for avgP */
  double globalSum = 0.0;
  MPI_Allreduce(&avgP, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /* do not adapt to local size imaxLocal & jmaxLocal */
  avgP = globalSum / (solver->imax * solver->jmax);

  for (int i = 0; i < size; i++) {
    p[i] = p[i] - avgP;
  }
}

void computeTimestep(Solver *solver) {
  double dt = solver->dtBound;
  double dx = solver->dx;
  double dy = solver->dy;
  double umax = maxElement(solver, solver->u);
  double vmax = maxElement(solver, solver->v);

  if (umax > 0) {
    dt = (dt > dx / umax) ? dx / umax : dt;
  }
  if (vmax > 0) {
    dt = (dt > dy / vmax) ? dy / vmax : dt;
  }

  solver->dt = dt * solver->tau;
}

/* TODO adapt to local size imaxLocal & jmaxLocal */
/* TODO Use isBoundary(solver, TOP/BOTTOM/LEFT/RIGHT) to check */
/* whether the rank has any boundaries and apply NOSLIP, SLIP or OUTFLOW
 * boundary condition */
void setBoundaryConditions(Solver *solver) {
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;
  double *u = solver->u;
  double *v = solver->v;

  // Left boundary
  if (isBoundary(solver, LEFT)) {
    switch (solver->bcLeft) {
    case NOSLIP:
      for (int j = 1; j <= jmaxLocal; j++) {
        U(0, j) = 0.0;
        V(0, j) = -V(1, j);
      }
      break;
    case SLIP:
      for (int j = 1; j <= jmaxLocal; j++) {
        U(0, j) = 0.0;
        V(0, j) = V(1, j);
      }
      break;
    case OUTFLOW:
      for (int j = 1; j <= jmaxLocal; j++) {
        U(0, j) = U(1, j);
        V(0, j) = V(1, j);
      }
      break;
    case PERIODIC:
      break;
    }
  }

  // Right boundary
  if (isBoundary(solver, RIGHT)) {
    switch (solver->bcRight) {
    case NOSLIP:
      for (int j = 1; j <= jmaxLocal; j++) {
        U(imaxLocal + 1, j) = 0.0;
        V(imaxLocal + 1, j) = -V(imaxLocal, j);
      }
      break;
    case SLIP:
      for (int j = 1; j <= jmaxLocal; j++) {
        U(imaxLocal + 1, j) = 0.0;
        V(imaxLocal + 1, j) = V(imaxLocal, j);
      }
      break;
    case OUTFLOW:
      for (int j = 1; j <= jmaxLocal; j++) {
        U(imaxLocal + 1, j) = U(imaxLocal, j);
        V(imaxLocal + 1, j) = V(imaxLocal, j);
      }
      break;
    case PERIODIC:
      break;
    }
  }

  // Bottom boundary
  if (isBoundary(solver, BOTTOM)) {
    switch (solver->bcBottom) {
    case NOSLIP:
      for (int i = 1; i <= imaxLocal; i++) {
        V(i, 0) = 0.0;
        U(i, 0) = -U(i, 1);
      }
      break;
    case SLIP:
      for (int i = 1; i <= imaxLocal; i++) {
        V(i, 0) = 0.0;
        U(i, 0) = U(i, 1);
      }
      break;
    case OUTFLOW:
      for (int i = 1; i <= imaxLocal; i++) {
        U(i, 0) = U(i, 1);
        V(i, 0) = V(i, 1);
      }
      break;
    case PERIODIC:
      break;
    }
  }

  // Top boundary
  if (isBoundary(solver, TOP)) {
    switch (solver->bcTop) {
    case NOSLIP:
      for (int i = 1; i <= imaxLocal; i++) {
        V(i, jmaxLocal + 1) = 0.0;
        U(i, jmaxLocal + 1) = -U(i, jmaxLocal);
      }
      break;
    case SLIP:
      for (int i = 1; i <= imaxLocal; i++) {
        V(i, jmaxLocal + 1) = 0.0;
        U(i, jmaxLocal + 1) = U(i, jmaxLocal);
      }
      break;
    case OUTFLOW:
      for (int i = 1; i <= imaxLocal; i++) {
        U(i, jmaxLocal + 1) = U(i, jmaxLocal);
        V(i, jmaxLocal + 1) = V(i, jmaxLocal);
      }
      break;
    case PERIODIC:
      break;
    }
  }
}

/* TODO adapt to local size imaxLocal & jmaxLocal */
void setSpecialBoundaryCondition(Solver *solver) {
  int imax = solver->imax;
  int jmax = solver->jmax;
  double mDy = solver->dy;
  double *u = solver->u;
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;

  /* In case of dcavity, apply special boundary condition to those */
  /* ranks who has top boundary. Check the indices for intuition.  */
  /* Use isBoundary(solver, TOP/BOTTOM/LEFT/RIGHT) to check if a   */
  /* rank has TOP boundary and the apply to it.                    */
  if (strcmp(solver->problem, "dcavity") == 0) {
    if (isBoundary(solver, TOP)) {
      for (int i = 1; i <= imaxLocal; i++) { // Use local indices
        U(i, jmaxLocal + 1) = 2.0 - U(i, jmaxLocal);
      }
    }
  }

  /* In case of canal, apply special boundary condition to those        */
  /* ranks who has left boundary. Check the indices for intuition.      */
  /* Use isBoundary(solver, TOP/BOTTOM/LEFT/RIGHT) to check if a        */
  /* rank has LEFT boundary and the apply to it.                        */
  /* Also you have to adapt the special boundary condition for each     */
  /* rank with LEFT boundary because in sequential case, it a           */
  /* parabola. Use solver->dims to adjust the formula according to rank */
  else if (strcmp(solver->problem, "canal") == 0) {
    if (isBoundary(solver, LEFT)) {
      double ylength = solver->ylength;
      double y;

      for (int j = 1; j <= jmaxLocal; j++) { // Use local indices
        y = mDy *
            (j - 0.5 +
             solver->coords[JDIM] * jmaxLocal); // Adjust for global position
        U(0, j) = y * (ylength - y) * 4.0 / (ylength * ylength);
      }
    }
  }
}

/* TODO adapt to local size imaxLocal & jmaxLocal */
void computeFG(Solver *solver) {
  double *u = solver->u;
  double *v = solver->v;
  double *f = solver->f;
  double *g = solver->g;
  int imax = solver->imax;
  int jmax = solver->jmax;
  double gx = solver->gx;
  double gy = solver->gy;
  double gamma = solver->gamma;
  double dt = solver->dt;
  double inverseRe = 1.0 / solver->re;
  double inverseDx = 1.0 / solver->dx;
  double inverseDy = 1.0 / solver->dy;
  double du2dx, dv2dy, duvdx, duvdy;
  double du2dx2, du2dy2, dv2dx2, dv2dy2;
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;

  exchange(solver, u);
  exchange(solver, v);

#ifdef DEBUG
  printf("Rank %d: imaxLocal %d, jmaxLocal %d, dims[IDIM] %d, dims[JDIM] %d\n",
         solver->rank, imaxLocal, jmaxLocal, solver->dims[IDIM],
         solver->dims[JDIM]);
#endif

  for (int j = 1; j <= jmaxLocal; j++) {
    for (int i = 1; i <= imaxLocal; i++) {
      du2dx = inverseDx * 0.25 *
                  ((U(i, j) + U(i + 1, j)) * (U(i, j) + U(i + 1, j)) -
                   (U(i, j) + U(i - 1, j)) * (U(i, j) + U(i - 1, j))) +
              gamma * inverseDx * 0.25 *
                  (fabs(U(i, j) + U(i + 1, j)) * (U(i, j) - U(i + 1, j)) +
                   fabs(U(i, j) + U(i - 1, j)) * (U(i, j) - U(i - 1, j)));

      duvdy =
          inverseDy * 0.25 *
              ((V(i, j) + V(i + 1, j)) * (U(i, j) + U(i, j + 1)) -
               (V(i, j - 1) + V(i + 1, j - 1)) * (U(i, j) + U(i, j - 1))) +
          gamma * inverseDy * 0.25 *
              (fabs(V(i, j) + V(i + 1, j)) * (U(i, j) - U(i, j + 1)) +
               fabs(V(i, j - 1) + V(i + 1, j - 1)) * (U(i, j) - U(i, j - 1)));

      du2dx2 =
          inverseDx * inverseDx * (U(i + 1, j) - 2.0 * U(i, j) + U(i - 1, j));
      du2dy2 =
          inverseDy * inverseDy * (U(i, j + 1) - 2.0 * U(i, j) + U(i, j - 1));
      F(i, j) =
          U(i, j) + dt * (inverseRe * (du2dx2 + du2dy2) - du2dx - duvdy + gx);

      duvdx =
          inverseDx * 0.25 *
              ((U(i, j) + U(i, j + 1)) * (V(i, j) + V(i + 1, j)) -
               (U(i - 1, j) + U(i - 1, j + 1)) * (V(i, j) + V(i - 1, j))) +
          gamma * inverseDx * 0.25 *
              (fabs(U(i, j) + U(i, j + 1)) * (V(i, j) - V(i + 1, j)) +
               fabs(U(i - 1, j) + U(i - 1, j + 1)) * (V(i, j) - V(i - 1, j)));

      dv2dy = inverseDy * 0.25 *
                  ((V(i, j) + V(i, j + 1)) * (V(i, j) + V(i, j + 1)) -
                   (V(i, j) + V(i, j - 1)) * (V(i, j) + V(i, j - 1))) +
              gamma * inverseDy * 0.25 *
                  (fabs(V(i, j) + V(i, j + 1)) * (V(i, j) - V(i, j + 1)) +
                   fabs(V(i, j) + V(i, j - 1)) * (V(i, j) - V(i, j - 1)));

      dv2dx2 =
          inverseDx * inverseDx * (V(i + 1, j) - 2.0 * V(i, j) + V(i - 1, j));
      dv2dy2 =
          inverseDy * inverseDy * (V(i, j + 1) - 2.0 * V(i, j) + V(i, j - 1));
      G(i, j) =
          V(i, j) + dt * (inverseRe * (dv2dx2 + dv2dy2) - duvdx - dv2dy + gy);
    }
  }

  /* Adapt boundary conditons for F only for ranks with LEFT and RIGHT boundary
   */
  /* Check the indices for more intuition. Use isBoundary(solver, LEFT/RIGHT) */
  /* to check if a rank has LEFT or RIGHT boundary. */
  /* ----------------------------- boundary of F --------------------------- */
  /* Adapt boundary conditions for F only for ranks with LEFT and RIGHT boundary
   */
  if (isBoundary(solver, LEFT)) {
    for (int j = 1; j <= jmaxLocal; j++) {
      F(0, j) = U(0, j);
    }
  }
  if (isBoundary(solver, RIGHT)) {
    for (int j = 1; j <= jmaxLocal; j++) {
      F(imaxLocal + 1, j) = U(imaxLocal + 1, j);
    }
  }

  /* Adapt boundary conditons for G only for ranks with TOP and BOTTOM boundary
   */
  /* Check the indices for more intuition. Use isBoundary(solver, TOP/BOTTOM) */
  /* to check if a rank has TOP or BOTTOM boundary. */
  /* ----------------------------- boundary of G --------------------------- */
  /* Adapt boundary conditions for G only for ranks with TOP and BOTTOM boundary
   */
  if (isBoundary(solver, BOTTOM)) {
    for (int i = 1; i <= imaxLocal; i++) {
      G(i, 0) = V(i, 0);
    }
  }
  if (isBoundary(solver, TOP)) {
    for (int i = 1; i <= imaxLocal; i++) {
      G(i, jmaxLocal + 1) = V(i, jmaxLocal + 1);
    }
  }
}

/* TODO adapt to local size imaxLocal & jmaxLocal */
void adaptUV(Solver *solver) {
  int imax = solver->imax;
  int jmax = solver->jmax;
  int imaxLocal = solver->imaxLocal;
  int jmaxLocal = solver->jmaxLocal;
  double *p = solver->p;
  double *u = solver->u;
  double *v = solver->v;
  double *f = solver->f;
  double *g = solver->g;
  double factorX = solver->dt / solver->dx;
  double factorY = solver->dt / solver->dy;

  for (int j = 1; j < jmaxLocal; j++) {
    for (int i = 1; i < imaxLocal; i++) {
      U(i, j) = F(i, j) - (P(i + 1, j) - P(i, j)) * factorX;
      V(i, j) = G(i, j) - (P(i, j + 1) - P(i, j)) * factorY;
    }
  }
}

void writeResult(Solver *solver, double *Pall, double *Uall, double *Vall) {

  int imax = solver->imax;
  int jmax = solver->jmax;
  double dx = solver->dx;
  double dy = solver->dy;
  double x = 0.0, y = 0.0;

  FILE *fp;
  fp = fopen("pressure.dat", "w");

  if (fp == NULL) {
    printf("Error!\n");
    exit(EXIT_FAILURE);
  }

#ifdef DEBUG
  printf("WRITING PRESSURE DATA\n");
#endif

  for (int j = 1; j < jmax + 1; j++) {
    y = (double)(j - 0.5) * dy;
    for (int i = 1; i < imax + 1; i++) {
      x = (double)(i - 0.5) * dx;
      fprintf(fp, "%.2f %.2f %f\n", x, y, Pall(i, j));
    }
    fprintf(fp, "\n");
  }

  fclose(fp);

  fp = fopen("velocity.dat", "w");

  if (fp == NULL) {
    printf("Error!\n");
    exit(EXIT_FAILURE);
  }

#ifdef DEBUG
  printf("WRITING VELOCITY DATA\n");
#endif

  for (int j = 1; j < jmax + 1; j++) {
    y = dy * (j - 0.5);
    for (int i = 1; i < imax + 1; i++) {
      x = dx * (i - 0.5);
      double velU = (Uall(i, j) + Uall(i - 1, j)) / 2.0;
      double velV = (Vall(i, j) + Vall(i, j - 1)) / 2.0;
      double len = sqrt((velU * velU) + (velV * velV));
      fprintf(fp, "%.2f %.2f %f %f %f\n", x, y, velU, velV, len);
    }
  }

  fclose(fp);
}
