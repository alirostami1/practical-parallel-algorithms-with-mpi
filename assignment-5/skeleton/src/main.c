/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#include <float.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "parameter.h"
#include "progress.h"
#include "solver.h"
#include "timing.h"

int main(int argc, char** argv)
{
    int rank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start, end;
    Parameter params;
    Solver solver;

    initParameter(&params);

    if (argc != 2) {
        printf("Usage: %s <configFile>\n", argv[0]);
        exit(EXIT_SUCCESS);
    }

    readParameter(&params, argv[1]);
    if (rank == 0) {
        printParameter(&params);
    }
    initSolver(&solver, &params);

    double tau = solver.tau;
    double te  = solver.te;
    double t   = 0.0;

    start = getTimeStamp();
    while (t <= te) {
        if (tau > 0.0) {
            computeTimestep(&solver);
        }

        setBoundaryConditions(&solver);
        setSpecialBoundaryCondition(&solver);
        computeFG(&solver);
        computeRHS(&solver);
        solve(&solver);
        adaptUV(&solver);
        t += solver.dt;

#ifdef VERBOSE
        if (rank == 0) {
            printf("TIME %f , TIMESTEP %f\n", t, solver.dt);
        }
#endif
    }
    end = getTimeStamp();
    stopProgress();
    if (rank == 0) {
        printf("Solution took %.2fs\n", end - start);
    }
    collectResult(&solver);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
