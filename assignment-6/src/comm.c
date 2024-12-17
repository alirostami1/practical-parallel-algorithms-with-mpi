/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#if defined(_MPI)
#include <mpi.h>
#endif
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "allocate.h"
#include "comm.h"

#if defined(_MPI)
// subroutines local to this module
static int sizeOfRank(int rank, int size, int N)
{
    return N / size + ((N % size > rank) ? 1 : 0);
}

static void setupCommunication(Comm* c, Direction direction, int layer)
{
    int imaxLocal = c->imaxLocal;
    int jmaxLocal = c->jmaxLocal;
    int kmaxLocal = c->kmaxLocal;

    size_t dblsize = sizeof(double);
    int sizes[NDIMS];
    int subSizes[NDIMS];
    int starts[NDIMS];

    sizes[IDIM] = imaxLocal + 2; // Full size in x-dimension (including ghost cells)
    sizes[JDIM] = jmaxLocal + 2; // Full size in y-dimension (including ghost cells)
    sizes[KDIM] = kmaxLocal + 2; // Full size in z-dimension (including ghost cells)

    // Initialize subSizes and starts depending on the direction and layer
    if (layer == HALO) {
        // Halo layer communication (ghost cells)

        // Default subarray size is the full size minus the ghost layer in each dimension
        subSizes[IDIM] = imaxLocal;
        subSizes[JDIM] = jmaxLocal;
        subSizes[KDIM] = kmaxLocal;

        // Initialize the starting points for the halo layer
        starts[IDIM] = 1;
        starts[JDIM] = 1;
        starts[KDIM] = 1;

        switch (direction) {
            case LEFT:
                subSizes[IDIM] = 1;   // 1 slice in the x-direction
                starts[IDIM] = 0;     // Start from the leftmost ghost cell
                break;
            case RIGHT:
                subSizes[IDIM] = 1;
                starts[IDIM] = imaxLocal + 1; // Start from the rightmost ghost cell
                break;
            case BOTTOM:
                subSizes[JDIM] = 1;
                starts[JDIM] = 0;
                break;
            case TOP:
                subSizes[JDIM] = 1;
                starts[JDIM] = jmaxLocal + 1;
                break;
            case FRONT:
                subSizes[KDIM] = 1;
                starts[KDIM] = 0;
                break;
            case BACK:
                subSizes[KDIM] = 1;
                starts[KDIM] = kmaxLocal + 1;
                break;
        }

        // Create the receive buffer datatype for halo communication
        MPI_Type_create_subarray(NDIMS, sizes, subSizes, starts, MPI_ORDER_C, MPI_DOUBLE, &c->rbufferTypes[direction]);
        MPI_Type_commit(&c->rbufferTypes[direction]);

    } else if (layer == BULK) {
        // Bulk layer communication (interior data)

        // Default subarray size is the full interior size
        subSizes[IDIM] = imaxLocal;
        subSizes[JDIM] = jmaxLocal;
        subSizes[KDIM] = kmaxLocal;

        // Starting points for the bulk data (excluding ghost cells)
        starts[IDIM] = 1;
        starts[JDIM] = 1;
        starts[KDIM] = 1;

        // Create the send buffer datatype for bulk communication
        MPI_Type_create_subarray(NDIMS, sizes, subSizes, starts, MPI_ORDER_C, MPI_DOUBLE, &c->sbufferTypes[direction]);
        MPI_Type_commit(&c->sbufferTypes[direction]);
    }
}


static void assembleResult(Comm* c,
    double* src,
    double* dst,
    int imaxLocal[],
    int jmaxLocal[],
    int kmaxLocal[],
    int offset[],
    int kmax,
    int jmax,
    int imax)
{
    int numRequests = 1;

    if (c->rank == 0) {
        numRequests = c->size + 1;
    }

    MPI_Request requests[numRequests];

    /* all ranks send their interpolated bulk array */
    MPI_Isend(src,
        // fill
        &requests[0]);

    /* rank 0 assembles the subdomains */
    if (c->rank == 0) {
        for (int i = 0; i < c->size; i++) {
            // prepare variables
            MPI_Type_create_subarray(NDIMS,
                // fill
                &domainType);
            MPI_Type_commit(&domainType);

            MPI_Irecv(dst, 1, domainType, i, 0, c->comm, &requests[i + 1]);
            MPI_Type_free(&domainType);
        }
    }

    MPI_Waitall(numRequests, requests, MPI_STATUSES_IGNORE);
}

static int sum(int* sizes, int position)
{
    int sum = 0;

    for (int i = 0; i < position; i++) {
        sum += sizes[i];
    }

    return sum;
}
#endif

// exported subroutines
void commReduction(double* v, int op)
{
#if defined(_MPI)
    if (op == MAX) {
        // fill
    } else if (op == SUM) {
        // fill
    }
#endif
}

int commIsBoundary(Comm* c, Direction direction)
{
#if defined(_MPI)
    switch (direction) {
    case LEFT:
    // fill all directions
    case NDIRS:
        printf("ERROR!\n");
        break;
    }
#endif

    return 1;
}

void commExchange(Comm* c, double* grid)
{
#if defined(_MPI)
    int counts[6]      = { 1, 1, 1, 1, 1, 1 };
    MPI_Aint displs[6] = { 0, 0, 0, 0, 0, 0 };

    MPI_Neighbor_alltoallw(grid,
        // fill
        c->comm);
#endif
}

void commShift(Comm* c, double* f, double* g, double* h)
{
#if defined(_MPI)
    MPI_Request requests[6] = { MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL,
        MPI_REQUEST_NULL };

    /* shift G */
    /* receive ghost cells from bottom neighbor */
    MPI_Irecv(g,
        // fill
        &requests[0]);

    /* send ghost cells to top neighbor */
    MPI_Isend(g, // fill
        &requests[1]);

    /* shift F */
    /* receive ghost cells from left neighbor */
    MPI_Irecv(f,
        // fill
        ,
        &requests[2]);

    /* send ghost cells to right neighbor */
    MPI_Isend(f,
        // fill
        &requests[3]);

    /* shift H */
    /* receive ghost cells from front neighbor */
    MPI_Irecv(h,
        // fill
        &requests[4]);

    /* send ghost cells to back neighbor */
    MPI_Isend(h,
        // fill
        &requests[5]);

    MPI_Waitall(6, requests, MPI_STATUSES_IGNORE);
#endif
}

#define G(v, i, j, k)                                                                    \
    v[(k) * (imaxLocal + 2) * (jmaxLocal + 2) + (j) * (imaxLocal + 2) + (i)]

void commCollectResult(Comm* c,
    double* ug,
    double* vg,
    double* wg,
    double* pg,
    double* u,
    double* v,
    double* w,
    double* p,
    int kmax,
    int jmax,
    int imax)
{
    int imaxLocal = c->imaxLocal;
    int jmaxLocal = c->jmaxLocal;
    int kmaxLocal = c->kmaxLocal;

#if defined(_MPI)
    int offset[c->size * NDIMS];
    int imaxLocalAll[c->size];
    int jmaxLocalAll[c->size];
    int kmaxLocalAll[c->size];

    MPI_Gather(&imaxLocal, 1, MPI_INT, imaxLocalAll, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&jmaxLocal, 1, MPI_INT, jmaxLocalAll, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&kmaxLocal, 1, MPI_INT, kmaxLocalAll, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (c->rank == 0) {
        for (int i = 0; i < c->size; i++) {
            int coords[NCORDS];
            MPI_Cart_coords(c->comm, i, NDIMS, coords);
            offset[i * NDIMS + IDIM] = sum(imaxLocalAll, coords[ICORD]);
            offset[i * NDIMS + JDIM] = sum(jmaxLocalAll, coords[JCORD]);
            offset[i * NDIMS + KDIM] = sum(kmaxLocalAll, coords[KCORD]);
            printf("Rank: %d, Coords(k,j,i): %d %d %d, Size(k,j,i): %d %d %d, "
                   "Offset(k,j,i): %d %d %d\n",
                i,
                coords[KCORD],
                coords[JCORD],
                coords[ICORD],
                kmaxLocalAll[i],
                jmaxLocalAll[i],
                imaxLocalAll[i],
                offset[i * NDIMS + KDIM],
                offset[i * NDIMS + JDIM],
                offset[i * NDIMS + IDIM]);
        }
    }

    size_t bytesize = imaxLocal * jmaxLocal * kmaxLocal * sizeof(double);
    double* tmp     = allocate(64, bytesize);
    int idx         = 0;

    /* collect P */
    for (int k = 1; k < kmaxLocal + 1; k++) {
        for (int j = 1; j < jmaxLocal + 1; j++) {
            for (int i = 1; i < imaxLocal + 1; i++) {
                tmp[idx++] = G(p, i, j, k);
            }
        }
    }

    assembleResult(c,
        tmp,
        pg,
        imaxLocalAll,
        jmaxLocalAll,
        kmaxLocalAll,
        offset,
        kmax,
        jmax,
        imax);

    /* collect U */
    idx = 0;

    for (int k = 1; k < kmaxLocal + 1; k++) {
        for (int j = 1; j < jmaxLocal + 1; j++) {
            for (int i = 1; i < imaxLocal + 1; i++) {
                tmp[idx++] = (G(u, i, j, k) + G(u, i - 1, j, k)) / 2.0;
            }
        }
    }

    assembleResult(c,
        tmp,
        ug,
        imaxLocalAll,
        jmaxLocalAll,
        kmaxLocalAll,
        offset,
        kmax,
        jmax,
        imax);

    /* collect V */
    idx = 0;

    for (int k = 1; k < kmaxLocal + 1; k++) {
        for (int j = 1; j < jmaxLocal + 1; j++) {
            for (int i = 1; i < imaxLocal + 1; i++) {
                tmp[idx++] = (G(v, i, j, k) + G(v, i, j - 1, k)) / 2.0;
            }
        }
    }

    assembleResult(c,
        tmp,
        vg,
        imaxLocalAll,
        jmaxLocalAll,
        kmaxLocalAll,
        offset,
        kmax,
        jmax,
        imax);

    /* collect W */
    idx = 0;

    for (int k = 1; k < kmaxLocal + 1; k++) {
        for (int j = 1; j < jmaxLocal + 1; j++) {
            for (int i = 1; i < imaxLocal + 1; i++) {
                tmp[idx++] = (G(w, i, j, k) + G(w, i, j, k - 1)) / 2.0;
            }
        }
    }

    assembleResult(c,
        tmp,
        wg,
        imaxLocalAll,
        jmaxLocalAll,
        kmaxLocalAll,
        offset,
        kmax,
        jmax,
        imax);

    free(tmp);
#else
    int idx = 0;

    for (int k = 1; k < kmaxLocal + 1; k++) {
        for (int j = 1; j < jmaxLocal + 1; j++) {
            for (int i = 1; i < imaxLocal + 1; i++) {
                pg[idx++] = G(p, i, j, k);
            }
        }
    }

    idx = 0;

    for (int k = 1; k < kmaxLocal + 1; k++) {
        for (int j = 1; j < jmaxLocal + 1; j++) {
            for (int i = 1; i < imaxLocal + 1; i++) {
                ug[idx++] = (G(u, i, j, k) + G(u, i - 1, j, k)) / 2.0;
            }
        }
    }

    idx = 0;

    for (int k = 1; k < kmaxLocal + 1; k++) {
        for (int j = 1; j < jmaxLocal + 1; j++) {
            for (int i = 1; i < imaxLocal + 1; i++) {
                vg[idx++] = (G(v, i, j, k) + G(v, i, j - 1, k)) / 2.0;
            }
        }
    }

    idx = 0;

    for (int k = 1; k < kmaxLocal + 1; k++) {
        for (int j = 1; j < jmaxLocal + 1; j++) {
            for (int i = 1; i < imaxLocal + 1; i++) {
                wg[idx++] = (G(w, i, j, k) + G(w, i, j, k - 1)) / 2.0;
            }
        }
    }
#endif
}

void commPrintConfig(Comm* c)
{
#if defined(_MPI)
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if (commIsMaster(c)) {
        printf("Communication setup:\n");
    }

    for (int i = 0; i < c->size; i++) {
        if (i == c->rank) {
            printf("\tRank %d of %d\n", c->rank, c->size);
            printf("\tNeighbours (front, back, bottom, top, left, right): %d, %d, %d, "
                   "%d, %d, %d\n",
                c->neighbours[FRONT],
                c->neighbours[BACK],
                c->neighbours[BOTTOM],
                c->neighbours[TOP],
                c->neighbours[LEFT],
                c->neighbours[RIGHT]);
            printf("\tCoordinates (k,j,i) %d %d %d\n",
                c->coords[KCORD],
                c->coords[JCORD],
                c->coords[ICORD]);
            printf("\tLocal domain size (k,j,i) %dx%dx%d\n",
                c->kmaxLocal,
                c->jmaxLocal,
                c->imaxLocal);
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void commInit(Comm* c, int argc, char** argv)
{
#if defined(_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &(c->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(c->size));
#else
    c->rank      = 0;
    c->size      = 1;
#endif
}

void commPartition(Comm* c, int kmax, int jmax, int imax)
{
#if defined(_MPI)
    // setup MPI cartesian topology

    // setup buffer types for communication
    setupCommunication(c, LEFT, BULK);
    // call for all other cases
#else
    c->imaxLocal = imax;
    c->jmaxLocal = jmax;
    c->kmaxLocal = kmax;
#endif
}

void commGetOffsets(Comm* c, int offsets[], int kmax, int jmax, int imax)
{
#if defined(_MPI)
    int sum = 0;

    for (int i = 0; i < c->coords[ICORD]; i++) {
        sum += sizeOfRank(i, c->dims[ICORD], imax);
    }
    offsets[IDIM] = sum;
    sum           = 0;

    for (int i = 0; i < c->coords[JCORD]; i++) {
        sum += sizeOfRank(i, c->dims[JCORD], jmax);
    }
    offsets[JDIM] = sum;
    sum           = 0;

    for (int i = 0; i < c->coords[KCORD]; i++) {
        sum += sizeOfRank(i, c->dims[KCORD], kmax);
    }
    offsets[KDIM] = sum;
#endif
}

void commFinalize(Comm* c)
{
#if defined(_MPI)
    for (int i = 0; i < NDIRS; i++) {
        MPI_Type_free(&c->sbufferTypes[i]);
        MPI_Type_free(&c->rbufferTypes[i]);
    }

    MPI_Finalize();
#endif
}
