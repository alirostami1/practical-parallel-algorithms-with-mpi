/*
 * Copyright (C) 2024 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "allocate.h"
#include "comm.h"
#include "parameter.h"
#include "progress.h"
#include "solver.h"
#include "timing.h"
#include "vtkWriter.h"

int main(int argc, char** argv)
{
    double timeStart, timeStop;
    Parameter p;
    Solver s;

    commInit(&s.comm, argc, argv);
    initParameter(&p);

    if (argc != 2) {
        printf("Usage: %s <configFile>\n", argv[0]);
        exit(EXIT_SUCCESS);
    }

    readParameter(&p, argv[1]);
    commPartition(&s.comm, p.kmax, p.jmax, p.imax);
    if (commIsMaster(&s.comm)) {
        printParameter(&p);
    }
    initSolver(&s, &p);
#ifndef VERBOSE
    initProgress(s.te);
#endif

    double tau = s.tau;
    double te  = s.te;
    double t   = 0.0;

    timeStart = getTimeStamp();
    while (t <= te) {
        if (tau > 0.0) computeTimestep(&s);
        setBoundaryConditions(&s);
        setSpecialBoundaryCondition(&s);
        computeFG(&s);
        computeRHS(&s);
        solve(&s);
        adaptUV(&s);
        t += s.dt;

#ifdef VERBOSE
        if (commIsMaster(&s.comm)) {
            printf("TIME %f , TIMESTEP %f\n", t, s.dt);
        }
#else
        printProgress(t);
#endif
    }
    timeStop = getTimeStamp();
#ifndef VERBOSE
    stopProgress();
#endif
    if (commIsMaster(&s.comm)) {
        printf("Solution took %.2fs\n", timeStop - timeStart);
    }

    double *pg, *ug, *vg, *wg;

    if (commIsMaster(&s.comm)) {
        size_t bytesize = s.grid.imax * s.grid.jmax * s.grid.kmax * sizeof(double);

        pg = allocate(64, bytesize);
        ug = allocate(64, bytesize);
        vg = allocate(64, bytesize);
        wg = allocate(64, bytesize);
    }

    commCollectResult(&s.comm,
        ug,
        vg,
        wg,
        pg,
        s.u,
        s.v,
        s.w,
        s.p,
        s.grid.kmax,
        s.grid.jmax,
        s.grid.imax);

    if (commIsMaster(&s.comm)) {
        VtkOptions opts = { .grid = s.grid };
        vtkOpen(&opts, s.problem);
        vtkScalar(&opts, "pressure", pg);
        vtkVector(&opts, "velocity", (VtkVector) { ug, vg, wg });
        vtkClose(&opts);
    }

    commFinalize(&s.comm);
    return EXIT_SUCCESS;
}
