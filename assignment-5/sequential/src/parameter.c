/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parameter.h"
#include "util.h"
#define MAXLINE 4096

void initParameter(Parameter* param)
{
    param->xlength = 1.0;
    param->ylength = 1.0;
    param->imax    = 100;
    param->jmax    = 100;
    param->itermax = 1000;
    param->eps     = 0.0001;
    param->omg     = 1.7;
    param->re      = 100.0;
    param->gamma   = 0.9;
    param->tau     = 0.5;
}

void readParameter(Parameter* param, const char* filename)
{
    FILE* fp = fopen(filename, "r");
    char line[MAXLINE];
    int i;

    if (!fp) {
        fprintf(stderr, "Could not open parameter file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    while (!feof(fp)) {
        line[0] = '\0';
        fgets(line, MAXLINE, fp);
        for (i = 0; line[i] != '\0' && line[i] != '#'; i++)
            ;
        line[i] = '\0';

        char* tok = strtok(line, " ");
        char* val = strtok(NULL, " ");

#define PARSE_PARAM(p, f)                                                                \
    if (strncmp(tok, #p, sizeof(#p) / sizeof(#p[0]) - 1) == 0) {                         \
        param->p = f(val);                                                               \
    }
#define PARSE_STRING(p) PARSE_PARAM(p, strdup)
#define PARSE_INT(p)    PARSE_PARAM(p, atoi)
#define PARSE_REAL(p)   PARSE_PARAM(p, atof)

        if (tok != NULL && val != NULL) {
            PARSE_REAL(xlength);
            PARSE_REAL(ylength);
            PARSE_INT(imax);
            PARSE_INT(jmax);
            PARSE_INT(itermax);
            PARSE_REAL(eps);
            PARSE_REAL(omg);
            PARSE_REAL(re);
            PARSE_REAL(tau);
            PARSE_REAL(gamma);
            PARSE_REAL(dt);
            PARSE_REAL(te);
            PARSE_REAL(gx);
            PARSE_REAL(gy);
            PARSE_STRING(name);
            PARSE_INT(bcLeft);
            PARSE_INT(bcRight);
            PARSE_INT(bcBottom);
            PARSE_INT(bcTop);
            PARSE_REAL(u_init);
            PARSE_REAL(v_init);
            PARSE_REAL(p_init);
        }
    }

    fclose(fp);
}

void printParameter(Parameter* param)
{
    printf("Parameters for %s\n", param->name);
    printf("Boundary conditions Left:%d Right:%d Bottom:%d Top:%d\n",
        param->bcLeft,
        param->bcRight,
        param->bcBottom,
        param->bcTop);
    printf("\tReynolds number: %.2f\n", param->re);
    printf("\tInit arrays: U:%.2f V:%.2f P:%.2f\n",
        param->u_init,
        param->v_init,
        param->p_init);
    printf("Geometry data:\n");
    printf("\tDomain box size (x, y): %.2f, %.2f\n", param->xlength, param->ylength);
    printf("\tCells (x, y): %d, %d\n", param->imax, param->jmax);
    printf("Timestep parameters:\n");
    printf("\tDefault stepsize: %.2f, Final time %.2f\n", param->dt, param->te);
    printf("\tTau factor: %.2f\n", param->tau);
    printf("Iterative solver parameters:\n");
    printf("\tMax iterations: %d\n", param->itermax);
    printf("\tepsilon (stopping tolerance) : %f\n", param->eps);
    printf("\tgamma (stopping tolerance) : %f\n", param->gamma);
    printf("\tomega (SOR relaxation): %f\n", param->omg);
}
