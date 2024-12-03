/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#ifndef __SOLVER_H_
#define __SOLVER_H_
#include "parameter.h"

enum BC { NOSLIP = 1, SLIP, OUTFLOW, PERIODIC };

typedef struct {
    /* geometry and grid information */
    double dx, dy;
    int imax, jmax;
    double xlength, ylength;
    /* arrays */
    double *p, *rhs;
    double *f, *g;
    double *u, *v;
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
} Solver;

extern void initSolver(Solver*, Parameter*);
extern void computeRHS(Solver*);
extern void solve(Solver*);
extern void solveRB(Solver*);
extern void solveRBA(Solver*);
extern void normalizePressure(Solver*);
extern void computeTimestep(Solver*);
extern void setBoundaryConditions(Solver*);
extern void setSpecialBoundaryCondition(Solver*);
extern void computeFG(Solver*);
extern void adaptUV(Solver*);
extern void writeResult(Solver*);
#endif
