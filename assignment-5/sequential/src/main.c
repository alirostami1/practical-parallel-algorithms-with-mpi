/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "parameter.h"
#include "progress.h"
#include "solver.h"
#include "timing.h"

int main(int argc, char** argv)
{
    double startTime, stopTime;
    Parameter params;
    Solver solver;
    initParameter(&params);

    if (argc != 2) {
        printf("Usage: %s <configFile>\n", argv[0]);
        exit(EXIT_SUCCESS);
    }

    readParameter(&params, argv[1]);
    printParameter(&params);
    initSolver(&solver, &params);
#ifndef VERBOSE
    initProgress(solver.te);
#endif

    double tau = solver.tau;
    double te  = solver.te;
    double t   = 0.0;
    int nt     = 0;

    startTime = getTimeStamp();
    while (t <= te) {
        if (tau > 0.0) computeTimestep(&solver);
        setBoundaryConditions(&solver);
        setSpecialBoundaryCondition(&solver);
        computeFG(&solver);
        computeRHS(&solver);
        if (nt % 100 == 0) normalizePressure(&solver);
        solve(&solver);
        adaptUV(&solver);
        t += solver.dt;
        nt++;

#ifdef VERBOSE
        printf("TIME %f , TIMESTEP %f\n", t, solver.dt);
#else
        printProgress(t);
#endif
    }
    stopTime = getTimeStamp();
    stopProgress();
    printf("Solution took %.2fs\n", stopTime - startTime);
    writeResult(&solver);
    return EXIT_SUCCESS;
}
