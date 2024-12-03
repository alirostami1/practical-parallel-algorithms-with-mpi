/*
 * Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#include "allocate.h"
#include "parameter.h"
#include "solver.h"

#define PI        3.14159265358979323846
#define P(i, j)   p[(j) * (imax + 2) + (i)]
#define RHS(i, j) rhs[(j) * (imax + 2) + (i)]

// static int sizeOfRank(int rank, int size, int N)
// {
//     return <EXTEND>;
// }

// static void exchange(Solver* solver)
// {
//     MPI_Request requests[4] = { MPI_REQUEST_NULL,
//         MPI_REQUEST_NULL,
//         MPI_REQUEST_NULL,
//         MPI_REQUEST_NULL };
//
//     /* exchange ghost cells with top neighbor */
//     if (solver->rank + 1 < solver->size) {
//         int top     = <EXTEND>;
//         double* src = <EXTEND>;
//         double* dst = <EXTEND>;
//
//         MPI_Isend(<EXTEND>);
//         MPI_Irecv(<EXTEND>);
//     }
//
//     /* exchange ghost cells with bottom neighbor */
//     if (solver->rank > 0) {
//         int bottom  = <EXTEND>;
//         double* src = <EXTEND>;
//         double* dst = <EXTEND>;
//
//         MPI_Isend(<EXTEND>);
//         MPI_Irecv(<EXTEND>);
//     }
//
//     MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
// }

// void getResult(Solver* solver)
// {
//     double* p = NULL;
//     int *rcvCounts, *displs;
//
//     if (solver->rank == 0) {
//         p = allocate(64, (solver->imax + 2) * (solver->jmax + 2) * sizeof(double));
//         rcvCounts    = <EXTEND>;
//         displs       = <EXTEND>;
//         rcvCounts[0] = <EXTEND>;
//         displs[0]    = <EXTEND>;
//         int cursor   = rcvCounts[0];
//
//         for (int i = 1; i < solver->size; i++) {
//             rcvCounts[i] = <EXTEND>
//             displs[i]    = cursor;
//             cursor += rcvCounts[i];
//         }
//     }
//
//     int cnt            = <EXTEND>;
//     double* sendbuffer = <EXTEND>;
//     MPI_Gatherv(<EXTEND>);
//     if (solver->rank == 0) {
//         writeResult(solver, p, "p.dat");
//         free(p);
//     }
// }

void initSolver(Solver* solver, Parameter* params, int problem)
{
    solver->imax    = params->imax;
    solver->jmax    = params->jmax;
    solver->dx      = params->xlength / params->imax;
    solver->dy      = params->ylength / params->jmax;
    solver->eps     = params->eps;
    solver->omega   = params->omg;
    solver->itermax = params->itermax;

    int imax        = solver->imax;
    int jmax        = solver->jmax;
    size_t bytesize = (imax + 2) * (jmax + 2) * sizeof(double);
    solver->p       = allocate(64, bytesize);
    solver->rhs     = allocate(64, bytesize);

    double dx   = solver->dx;
    double dy   = solver->dy;
    double* p   = solver->p;
    double* rhs = solver->rhs;

    // adapt for MPI case
    for (int j = 0; j < jmax + 2; j++) {
        for (int i = 0; i < imax + 2; i++) {
            P(i, j) = sin(2.0 * PI * i * dx * 2.0) + sin(2.0 * PI * j * dy * 2.0);
        }
    }

    if (problem == 2) {
        for (int j = 0; j < jmax + 2; j++) {
            for (int i = 0; i < imax + 2; i++) {
                RHS(i, j) = sin(2.0 * PI * i * dx);
            }
        }
    } else {
        for (int j = 0; j < jmax + 2; j++) {
            for (int i = 0; i < imax + 2; i++) {
                RHS(i, j) = 0.0;
            }
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
                           ((P(i - 1, j) - 2.0 * P(i, j) + P(i + 1, j)) * idx2 +
                               (P(i, j - 1) - 2.0 * P(i, j) + P(i, j + 1)) * idy2);

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

    // printf("Solver took %d iterations to reach %f\n", it, sqrt(res));
    printf("%d ", it);
}

void solveRB(Solver* solver)
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
    int pass, jsw, isw;

    while ((res >= epssq) && (it < itermax)) {
        res = 0.0;
        jsw = 1;

        for (pass = 0; pass < 2; pass++) {
            isw = jsw;

            for (int j = 1; j < jmax + 1; j++) {
                for (int i = isw; i < imax + 1; i += 2) {

                    double r = RHS(i, j) -
                               ((P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) * idx2 +
                                   (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) * idy2);

                    P(i, j) -= (factor * r);
                    res += (r * r);
                }
                isw = 3 - isw;
            }
            jsw = 3 - jsw;
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

    // printf("Solver took %d iterations to reach %f\n", it, sqrt(res));
    printf("%d ", it);
}

void solveRBA(Solver* solver)
{
    int imax      = solver->imax;
    int jmax      = solver->jmax;
    double eps    = solver->eps;
    int itermax   = solver->itermax;
    double dx2    = solver->dx * solver->dx;
    double dy2    = solver->dy * solver->dy;
    double idx2   = 1.0 / dx2;
    double idy2   = 1.0 / dy2;
    double factor = 0.5 * (dx2 * dy2) / (dx2 + dy2);
    double* p     = solver->p;
    double* rhs   = solver->rhs;
    double epssq  = eps * eps;
    int it        = 0;
    double res    = 1.0;
    int pass, jsw, isw;
    double omega = solver->omega;

    while ((res >= epssq) && (it < itermax)) {
        res = 0.0;
        jsw = 1;

        for (pass = 0; pass < 2; pass++) {
            isw = jsw;

            for (int j = 1; j < jmax + 1; j++) {
                for (int i = isw; i < imax + 1; i += 2) {

                    double r = RHS(i, j) -
                               ((P(i + 1, j) - 2.0 * P(i, j) + P(i - 1, j)) * idx2 +
                                   (P(i, j + 1) - 2.0 * P(i, j) + P(i, j - 1)) * idy2);

                    P(i, j) -= (omega * factor * r);
                    res += (r * r);
                }
                isw = 3 - isw;
            }
            jsw   = 3 - jsw;
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
        printf("%d Residuum: %e Omega: %e\n", it, res, omega);
#endif
        it++;
    }

    printf("%d ", it);
}

void writeResult(Solver* solver, char* filename)
{
    int imax  = solver->imax;
    int jmax  = solver->jmax;
    double* p = solver->p;

    FILE* fp;
    fp = fopen(filename, "w");

    if (fp == NULL) {
        printf("Error!\n");
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < jmax + 2; j++) {
        for (int i = 0; i < imax + 2; i++) {
            fprintf(fp, "%f ", P(i, j));
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}