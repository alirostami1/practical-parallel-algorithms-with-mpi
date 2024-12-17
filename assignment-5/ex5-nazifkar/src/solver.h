/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#ifndef __SOLVER_H_
#define __SOLVER_H_
#include "parameter.h"
#include <mpi.h>

enum BC { NOSLIP = 1, SLIP, OUTFLOW, PERIODIC };
enum direction { LEFT = 0, RIGHT, BOTTOM, TOP, NDIRS };
enum dimension { IDIM = 0, JDIM, NDIMS };
enum cdimension { CJDIM = 0, CIDIM };

typedef struct {
    /* geometry and grid information */
    double dx, dy;
    int imax, jmax;
    int imaxLocal, jmaxLocal;
    double xlength, ylength;
    /* arrays */
    double *p, *rhs;
    double *f, *g;
    double *u, *v;
    double *Pall, *Uall, *Vall;
    /* parameters */
    double eps, omega;
    double re, tau, gamma;
    double gx, gy;
    /* time stepping */
    int itermax;
    double dt, te;
    double dtBound;
    char* problem;
    int bcLeft, bcRight, bcBottom, bcTop;
    /* mpi */
    int rank;
    int size;

    MPI_Comm comm;

    MPI_Datatype bufferTypes[NDIRS];
    MPI_Aint sdispls[NDIRS];
    MPI_Aint rdispls[NDIRS];

    /* coordinates of the rank and the dimension of the rank */
    int neighbours[NDIRS];
    int coords[NDIMS];
    int dims[NDIMS];
} Solver;

void initSolver(Solver*, Parameter*);
void computeRHS(Solver*);
void solve(Solver*);
void normalizePressure(Solver*);
void exchange(Solver*, double*);
void computeTimestep(Solver*);
void setBoundaryConditions(Solver*);
void setSpecialBoundaryCondition(Solver*);
void computeFG(Solver*);
void shift(Solver* );
void adaptUV(Solver*);
void collectResult(Solver*);
void writeResult(Solver*, double*, double*, double*);
#endif
