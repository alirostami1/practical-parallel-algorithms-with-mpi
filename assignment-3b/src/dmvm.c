/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#include "timing.h"
#include <stdio.h>
#include <stdlib.h>

double dmvm(double* restrict y,
    const double* restrict a,
    const double* restrict x,
    int N,
    int iter)
{
    double ts, te;

    ts = getTimeStamp();
    for (int j = 0; j < iter; j++) {
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                y[r] = y[r] + a[r * N + c] * x[c];
            }
        }
#ifdef CHECK
        {
            double sum = 0.0;

            for (int i = 0; i < N; i++) {
                sum += y[i];
                y[i] = 0.0;
            }
            fprintf(stderr, "Sum: %f\n", sum);
        }
#endif
    }
    te = getTimeStamp();

    return te - ts;
}
