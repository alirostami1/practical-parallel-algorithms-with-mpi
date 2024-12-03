/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#ifndef __SOLVER_H_
#define __SOLVER_H_
#include "parameter.h"

typedef struct {
    double dx, dy;
    double ys;
    int imax, jmax;
    int jmaxLocal;
    int rank;
    int size;

    double *p, *rhs;
    double eps, omega;
    int itermax;
} Solver;

extern void initSolver(Solver*, Parameter*, int problem);
extern void getResult(Solver*);
extern void writeResult(Solver*, char*);
extern void solve(Solver*);
#endif
