/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocate.h"
#include "parameter.h"
#include "solver.h"
#include "util.h"

#define P(i, j)   p[(j) * (imax + 2) + (i)]
#define F(i, j)   f[(j) * (imax + 2) + (i)]
#define G(i, j)   g[(j) * (imax + 2) + (i)]
#define U(i, j)   u[(j) * (imax + 2) + (i)]
#define V(i, j)   v[(j) * (imax + 2) + (i)]
#define RHS(i, j) rhs[(j) * (imax + 2) + (i)]

static void print(Solver* solver, double* grid)
{
    int imax = solver->imax;

    for (int j = 0; j < solver->jmax + 2; j++) {
        printf("%02d: ", j);
        for (int i = 0; i < solver->imax + 2; i++) {
            printf("%12.8f  ", grid[j * (imax + 2) + i]);
        }
        printf("\n");
    }
    fflush(stdout);
}

static void printConfig(Solver* solver)
{
    printf("Parameters for #%s#\n", solver->problem);
    printf("Boundary conditions Left:%d Right:%d Bottom:%d Top:%d\n",
        solver->bcLeft,
        solver->bcRight,
        solver->bcBottom,
        solver->bcTop);
    printf("\tReynolds number: %.2f\n", solver->re);
    printf("\tGx Gy: %.2f %.2f\n", solver->gx, solver->gy);
    printf("Geometry data:\n");
    printf("\tDomain box size (x, y): %.2f, %.2f\n", solver->xlength, solver->ylength);
    printf("\tCells (x, y): %d, %d\n", solver->imax, solver->jmax);
    printf("Timestep parameters:\n");
    printf("\tDefault stepsize: %.2f, Final time %.2f\n", solver->dt, solver->te);
    printf("\tdt bound: %.6f\n", solver->dtBound);
    printf("\tTau factor: %.2f\n", solver->tau);
    printf("Iterative solver parameters:\n");
    printf("\tMax iterations: %d\n", solver->itermax);
    printf("\tepsilon (stopping tolerance) : %f\n", solver->eps);
    printf("\tgamma factor: %f\n", solver->gamma);
    printf("\tomega (SOR relaxation): %f\n", solver->omega);
}

void initSolver(Solver* solver, Parameter* params)
{
    solver->problem  = params->name;
    solver->bcLeft   = params->bcLeft;
    solver->bcRight  = params->bcRight;
    solver->bcBottom = params->bcBottom;
    solver->bcTop    = params->bcTop;
    solver->imax     = params->imax;
    solver->jmax     = params->jmax;
    solver->xlength  = params->xlength;
    solver->ylength  = params->ylength;
    solver->dx       = params->xlength / params->imax;
    solver->dy       = params->ylength / params->jmax;
    solver->eps      = params->eps;
    solver->omega    = params->omg;
    solver->itermax  = params->itermax;
    solver->re       = params->re;
    solver->gx       = params->gx;
    solver->gy       = params->gy;
    solver->dt       = params->dt;
    solver->te       = params->te;
    solver->tau      = params->tau;
    solver->gamma    = params->gamma;

    int imax    = solver->imax;
    int jmax    = solver->jmax;
    size_t size = (imax + 2) * (jmax + 2) * sizeof(double);
    solver->u   = allocate(64, size);
    solver->v   = allocate(64, size);
    solver->p   = allocate(64, size);
    solver->rhs = allocate(64, size);
    solver->f   = allocate(64, size);
    solver->g   = allocate(64, size);

    for (int i = 0; i < (imax + 2) * (jmax + 2); i++) {
        solver->u[i]   = params->u_init;
        solver->v[i]   = params->v_init;
        solver->p[i]   = params->p_init;
        solver->rhs[i] = 0.0;
        solver->f[i]   = 0.0;
        solver->g[i]   = 0.0;
    }

    double dx        = solver->dx;
    double dy        = solver->dy;
    double invSqrSum = 1.0 / (dx * dx) + 1.0 / (dy * dy);
    solver->dtBound  = 0.5 * solver->re * 1.0 / invSqrSum;
#ifdef VERBOSE
    printConfig(solver);
#endif
}

void computeRHS(Solver* solver)
{
    int imax    = solver->imax;
    int jmax    = solver->jmax;
    double idx  = 1.0 / solver->dx;
    double idy  = 1.0 / solver->dy;
    double idt  = 1.0 / solver->dt;
    double* rhs = solver->rhs;
    double* f   = solver->f;
    double* g   = solver->g;

    for (int j = 1; j < jmax + 1; j++) {
        for (int i = 1; i < imax + 1; i++) {
            RHS(i, j) = idt *
                        ((F(i, j) - F(i - 1, j)) * idx + (G(i, j) - G(i, j - 1)) * idy);
        }
    }
}

void solve(Solver* solver)
{
    int imax      = solver->imax;
    int jmax      = solver->jmax;
    double eps    = solver->eps;
    int itermax   = solver->itermax;
    double dx2    = solver->dx * solver->dx;
    double dy2    = solver->dy * solver->dy;
    double idx2   = 1.0 / dx2;
    double idy2   = 1.0 / dy2;
    double factor = solver->omega * 0.5 * (dx2 * dy2) / (dx2 + dy2);
    double* p     = solver->p;
    double* rhs   = solver->rhs;
    double epssq  = eps * eps;
    int it        = 0;
    double res    = 1.0;

    while ((res >= epssq) && (it < itermax)) {
        res = 0.0;

        for (int j = 1; j < jmax + 1; j++) {
            for (int i = 1; i < imax + 1; i++) {

                double r = RHS(i, j) -
                           ((P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) * idx2 +
                               (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) * idy2);

                P(i, j) -= (factor * r);
                res += (r * r);
            }
        }

        for (int i = 1; i < imax + 1; i++) {
            P(i, 0)        = P(i, 1);
            P(i, jmax + 1) = P(i, jmax);
        }

        for (int j = 1; j < jmax + 1; j++) {
            P(0, j)        = P(1, j);
            P(imax + 1, j) = P(imax, j);
        }

        res = res / (double)(imax * jmax);
#ifdef DEBUG
        printf("%d Residuum: %e\n", it, res);
#endif
        it++;
    }

#ifdef VERBOSE
    printf("Solver took %d iterations to reach %f\n", it, sqrt(res));
#endif
}

static double maxElement(Solver* solver, double* m)
{
    int size      = (solver->imax + 2) * (solver->jmax + 2);
    double maxval = DBL_MIN;

    for (int i = 0; i < size; i++) {
        maxval = MAX(maxval, fabs(m[i]));
    }

    return maxval;
}

void normalizePressure(Solver* solver)
{
    int size    = (solver->imax + 2) * (solver->jmax + 2);
    double* p   = solver->p;
    double avgP = 0.0;

    for (int i = 0; i < size; i++) {
        avgP += p[i];
    }
    avgP /= size;

    for (int i = 0; i < size; i++) {
        p[i] = p[i] - avgP;
    }
}

void computeTimestep(Solver* solver)
{
    double dt   = solver->dtBound;
    double dx   = solver->dx;
    double dy   = solver->dy;
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

void setBoundaryConditions(Solver* solver)
{
    int imax  = solver->imax;
    int jmax  = solver->jmax;
    double* u = solver->u;
    double* v = solver->v;

    // Left boundary
    switch (solver->bcLeft) {
    case NOSLIP:
        for (int j = 1; j < jmax + 1; j++) {
            U(0, j) = 0.0;
            V(0, j) = -V(1, j);
        }
        break;
    case SLIP:
        for (int j = 1; j < jmax + 1; j++) {
            U(0, j) = 0.0;
            V(0, j) = V(1, j);
        }
        break;
    case OUTFLOW:
        for (int j = 1; j < jmax + 1; j++) {
            U(0, j) = U(1, j);
            V(0, j) = V(1, j);
        }
        break;
    case PERIODIC:
        break;
    }

    // Right boundary
    switch (solver->bcRight) {
    case NOSLIP:
        for (int j = 1; j < jmax + 1; j++) {
            U(imax, j)     = 0.0;
            V(imax + 1, j) = -V(imax, j);
        }
        break;
    case SLIP:
        for (int j = 1; j < jmax + 1; j++) {
            U(imax, j)     = 0.0;
            V(imax + 1, j) = V(imax, j);
        }
        break;
    case OUTFLOW:
        for (int j = 1; j < jmax + 1; j++) {
            U(imax, j)     = U(imax - 1, j);
            V(imax + 1, j) = V(imax, j);
        }
        break;
    case PERIODIC:
        break;
    }

    // Bottom boundary
    switch (solver->bcBottom) {
    case NOSLIP:
        for (int i = 1; i < imax + 1; i++) {
            V(i, 0) = 0.0;
            U(i, 0) = -U(i, 1);
        }
        break;
    case SLIP:
        for (int i = 1; i < imax + 1; i++) {
            V(i, 0) = 0.0;
            U(i, 0) = U(i, 1);
        }
        break;
    case OUTFLOW:
        for (int i = 1; i < imax + 1; i++) {
            U(i, 0) = U(i, 1);
            V(i, 0) = V(i, 1);
        }
        break;
    case PERIODIC:
        break;
    }

    // Top boundary
    switch (solver->bcTop) {
    case NOSLIP:
        for (int i = 1; i < imax + 1; i++) {
            V(i, jmax)     = 0.0;
            U(i, jmax + 1) = -U(i, jmax);
        }
        break;
    case SLIP:
        for (int i = 1; i < imax + 1; i++) {
            V(i, jmax)     = 0.0;
            U(i, jmax + 1) = U(i, jmax);
        }
        break;
    case OUTFLOW:
        for (int i = 1; i < imax + 1; i++) {
            U(i, jmax + 1) = U(i, jmax);
            V(i, jmax)     = V(i, jmax - 1);
        }
        break;
    case PERIODIC:
        break;
    }
}

void setSpecialBoundaryCondition(Solver* solver)
{
    int imax   = solver->imax;
    int jmax   = solver->jmax;
    double mDy = solver->dy;
    double* u  = solver->u;

    if (strcmp(solver->problem, "dcavity") == 0) {
        for (int i = 1; i < imax; i++) {
            U(i, jmax + 1) = 2.0 - U(i, jmax);
        }
    } else if (strcmp(solver->problem, "canal") == 0) {
        double ylength = solver->ylength;
        double y;

        for (int j = 1; j < jmax + 1; j++) {
            y       = mDy * (j - 0.5);
            U(0, j) = y * (ylength - y) * 4.0 / (ylength * ylength);
        }
    }
}

void computeFG(Solver* solver)
{
    double* u        = solver->u;
    double* v        = solver->v;
    double* f        = solver->f;
    double* g        = solver->g;
    int imax         = solver->imax;
    int jmax         = solver->jmax;
    double gx        = solver->gx;
    double gy        = solver->gy;
    double gamma     = solver->gamma;
    double dt        = solver->dt;
    double inverseRe = 1.0 / solver->re;
    double inverseDx = 1.0 / solver->dx;
    double inverseDy = 1.0 / solver->dy;
    double du2dx, dv2dy, duvdx, duvdy;
    double du2dx2, du2dy2, dv2dx2, dv2dy2;

    for (int j = 1; j < jmax + 1; j++) {
        for (int i = 1; i < imax + 1; i++) {
            du2dx = inverseDx * 0.25 *
                        ((U(i, j) + U(i + 1, j)) * (U(i, j) + U(i + 1, j)) -
                            (U(i, j) + U(i - 1, j)) * (U(i, j) + U(i - 1, j))) +
                    gamma * inverseDx * 0.25 *
                        (fabs(U(i, j) + U(i + 1, j)) * (U(i, j) - U(i + 1, j)) +
                            fabs(U(i, j) + U(i - 1, j)) * (U(i, j) - U(i - 1, j)));

            duvdy = inverseDy * 0.25 *
                        ((V(i, j) + V(i + 1, j)) * (U(i, j) + U(i, j + 1)) -
                            (V(i, j - 1) + V(i + 1, j - 1)) * (U(i, j) + U(i, j - 1))) +
                    gamma * inverseDy * 0.25 *
                        (fabs(V(i, j) + V(i + 1, j)) * (U(i, j) - U(i, j + 1)) +
                            fabs(V(i, j - 1) + V(i + 1, j - 1)) *
                                (U(i, j) - U(i, j - 1)));

            du2dx2  = inverseDx * inverseDx * (U(i + 1, j) - 2.0 * U(i, j) + U(i - 1, j));
            du2dy2  = inverseDy * inverseDy * (U(i, j + 1) - 2.0 * U(i, j) + U(i, j - 1));
            F(i, j) = U(i, j) + dt * (inverseRe * (du2dx2 + du2dy2) - du2dx - duvdy + gx);

            duvdx = inverseDx * 0.25 *
                        ((U(i, j) + U(i, j + 1)) * (V(i, j) + V(i + 1, j)) -
                            (U(i - 1, j) + U(i - 1, j + 1)) * (V(i, j) + V(i - 1, j))) +
                    gamma * inverseDx * 0.25 *
                        (fabs(U(i, j) + U(i, j + 1)) * (V(i, j) - V(i + 1, j)) +
                            fabs(U(i - 1, j) + U(i - 1, j + 1)) *
                                (V(i, j) - V(i - 1, j)));

            dv2dy = inverseDy * 0.25 *
                        ((V(i, j) + V(i, j + 1)) * (V(i, j) + V(i, j + 1)) -
                            (V(i, j) + V(i, j - 1)) * (V(i, j) + V(i, j - 1))) +
                    gamma * inverseDy * 0.25 *
                        (fabs(V(i, j) + V(i, j + 1)) * (V(i, j) - V(i, j + 1)) +
                            fabs(V(i, j) + V(i, j - 1)) * (V(i, j) - V(i, j - 1)));

            dv2dx2  = inverseDx * inverseDx * (V(i + 1, j) - 2.0 * V(i, j) + V(i - 1, j));
            dv2dy2  = inverseDy * inverseDy * (V(i, j + 1) - 2.0 * V(i, j) + V(i, j - 1));
            G(i, j) = V(i, j) + dt * (inverseRe * (dv2dx2 + dv2dy2) - duvdx - dv2dy + gy);
        }
    }

    /* ---------------------- boundary of F --------------------------- */
    for (int j = 1; j < jmax + 1; j++) {
        F(0, j)    = U(0, j);
        F(imax, j) = U(imax, j);
    }

    /* ---------------------- boundary of G --------------------------- */
    for (int i = 1; i < imax + 1; i++) {
        G(i, 0)    = V(i, 0);
        G(i, jmax) = V(i, jmax);
    }
}

void adaptUV(Solver* solver)
{
    int imax       = solver->imax;
    int jmax       = solver->jmax;
    double* p      = solver->p;
    double* u      = solver->u;
    double* v      = solver->v;
    double* f      = solver->f;
    double* g      = solver->g;
    double factorX = solver->dt / solver->dx;
    double factorY = solver->dt / solver->dy;

    for (int j = 1; j < jmax + 1; j++) {
        for (int i = 1; i < imax + 1; i++) {
            U(i, j) = F(i, j) - (P(i + 1, j) - P(i, j)) * factorX;
            V(i, j) = G(i, j) - (P(i, j + 1) - P(i, j)) * factorY;
        }
    }
}

void writeResult(Solver* solver)
{
    int imax  = solver->imax;
    int jmax  = solver->jmax;
    double dx = solver->dx;
    double dy = solver->dy;
    double* p = solver->p;
    double* u = solver->u;
    double* v = solver->v;
    double x = 0.0, y = 0.0;

    FILE* fp;
    fp = fopen("pressure.dat", "w");

    if (fp == NULL) {
        printf("Error!\n");
        exit(EXIT_FAILURE);
    }

    for (int j = 1; j < jmax + 1; j++) {
        y = (double)(j - 0.5) * dy;
        for (int i = 1; i < imax + 1; i++) {
            x = (double)(i - 0.5) * dx;
            fprintf(fp, "%.2f %.2f %f\n", x, y, P(i, j));
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    fp = fopen("velocity.dat", "w");

    if (fp == NULL) {
        printf("Error!\n");
        exit(EXIT_FAILURE);
    }

    for (int j = 1; j < jmax + 1; j++) {
        y = dy * (j - 0.5);
        for (int i = 1; i < imax + 1; i++) {
            x            = dx * (i - 0.5);
            double vel_u = (U(i, j) + U(i - 1, j)) / 2.0;
            double vel_v = (V(i, j) + V(i, j - 1)) / 2.0;
            double len   = sqrt((vel_u * vel_u) + (vel_v * vel_v));
            fprintf(fp, "%.2f %.2f %f %f %f\n", x, y, vel_u, vel_v, len);
        }
    }

    fclose(fp);
}
