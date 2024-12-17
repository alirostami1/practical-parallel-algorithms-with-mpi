/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#include <math.h>
// #include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "progress.h"

static double _end;
static int _current;
static int _rank = -1;

void initProgress(double end)
{
    // MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    _rank    = 0;
    _end     = end;
    _current = 0;

    if (_rank == 0) {
        printf("[          ]");
        fflush(stdout);
    }
}

void printProgress(double current)
{
    if (_rank == 0) {
        int new = (int)rint((current / _end) * 10.0);

        if (new > _current) {
            char progress[11];
            _current    = new;
            progress[0] = 0;

            for (int i = 0; i < 10; i++) {
                if (i < _current) {
                    sprintf(progress + strlen(progress), "#");
                } else {
                    sprintf(progress + strlen(progress), " ");
                }
            }
            printf("\r[%s]", progress);
        }
        fflush(stdout);
    }
}

void stopProgress()
{
    if (_rank == 0) {
        printf("\n");
        fflush(stdout);
    }
}
