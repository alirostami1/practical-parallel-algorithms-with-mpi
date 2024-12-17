/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of nusif-solver.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 */
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "vtkWriter.h"
#define G(v, i, j, k) v[(k)*imax * jmax + (j)*imax + (i)]

// reset fileview for output of string headers
static void resetFileview(VtkOptions* o)
{
    MPI_Offset disp;
    MPI_File_sync(o->fh);
    MPI_Barrier(o->comm.comm);
    MPI_File_get_size(o->fh, &disp);
    MPI_File_set_view(o->fh, disp, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
}

static double floatSwap(double f)
{
    union {
        double f;
        char b[8];
    } dat1, dat2;

    dat1.f    = f;
    dat2.b[0] = dat1.b[7];
    dat2.b[1] = dat1.b[6];
    dat2.b[2] = dat1.b[5];
    dat2.b[3] = dat1.b[4];
    dat2.b[4] = dat1.b[3];
    dat2.b[5] = dat1.b[2];
    dat2.b[6] = dat1.b[1];
    dat2.b[7] = dat1.b[0];
    return dat2.f;
}

static void writeHeader(VtkOptions* o)
{
	
	// Only rank 0 should write this header to the file.
	// Use MPI IO along with rank check to write this header to the vtk file.
	
    fprintf(o->fh, "# vtk DataFile Version 3.0\n");
    fprintf(o->fh, "PAMPI cfd solver output\n");
    if (o->fmt == ASCII) {
        fprintf(o->fh, "ASCII\n");
    } else if (o->fmt == BINARY) {
        fprintf(o->fh, "BINARY\n");
    }

    fprintf(o->fh, "DATASET STRUCTURED_POINTS\n");
    fprintf(o->fh, "DIMENSIONS %d %d %d\n", o->grid.imax, o->grid.jmax, o->grid.kmax);
    fprintf(o->fh,
        "ORIGIN %f %f %f\n",
        o->grid.dx * 0.5,
        o->grid.dy * 0.5,
        o->grid.dz * 0.5);
    fprintf(o->fh, "SPACING %f %f %f\n", o->grid.dx, o->grid.dy, o->grid.dz);
    fprintf(o->fh, "POINT_DATA %d\n", o->grid.imax * o->grid.jmax * o->grid.kmax);
}

void vtkOpen(VtkOptions* o, char* problem)
{
	
	// Open the vtk file using MPI IO
	// Use MPI_File_open() in CREATE or WRONLY mode
	
    char filename[50];

    snprintf(filename, 50, "%s.vtk", problem);
    o->fh = fopen(filename, "w");
    writeHeader(o);

    printf("Writing VTK output for %s\n", problem);
}

static void writeScalar(VtkOptions* o, double* s)
{
	
    int imax = o->grid.imax;
    int jmax = o->grid.jmax;
    int kmax = o->grid.kmax;

    for (int k = 0; k < kmax; k++) {
        for (int j = 0; j < jmax; j++) {
            for (int i = 0; i < imax; i++) {
                if (o->fmt == ASCII) {
                    fprintf(o->fh, "%f\n", G(s, i, j, k));
                } else if (o->fmt == BINARY) {
                    fwrite((double[1]) { floatSwap(G(s, i, j, k)) },
                        sizeof(double),
                        1,
                        o->fh);
                }
            }
        }
    }
    if (o->fmt == BINARY) fprintf(o->fh, "\n");
}

static bool isInitialized(FILE* ptr)
{
    if (ptr == NULL) {
        printf("vtkWriter not initialize! Call vtkOpen first!\n");
        return false;
    }
    return true;
}

void vtkScalar(VtkOptions* o, char* name, double* s)
{
	// Write MPI IO code to parallelise writing to a single file.
	// Steps to perform MPI IO
	// 1. Use resetFileview(o); here before starting MPI IO
	
    printf("Register scalar %s\n", name);
    if (!isInitialized(o->fh)) return;
    
	// Make sure this is only written by rank 0.
	fprintf(o->fh, "SCALARS %s double 1\n", name);
    fprintf(o->fh, "LOOKUP_TABLE default\n");
	
	// 2. Get offsets from all ranks using commGetOffsets from comm.c. This will be 
	// 3. Use the MPI_File_sync and MPI_File_get_size (along with MPI_Offset) to get file size.
	// 4. Use offsets gathered in 3 to create subarray. Create subarray from global problem (Imax, Jmax, Kmax)
	// 5. Finally use the MPI_File_set_view to set the view in file for each rank.
	// 6. Instead of using writeScalar(), create another subarray excluding the ghost cells (imaxlocal+2 etc.)
	// 7. Now use the MPI_File_write using the subarray created in 6 so that each rank can write to the same file at the given displacements.
	
    writeScalar(o, s);
	
	
	// Binary segment must be terminated with newline character
    // resetFileview(o);
    // if (commIsMaster(&o->comm)) {
        // MPI_File_write(o->fh, "\n", 1, MPI_CHAR, MPI_STATUS_IGNORE);
    // }
}

static void writeVector(VtkOptions* o, VtkVector vec)
{
    int imax = o->grid.imax;
    int jmax = o->grid.jmax;
    int kmax = o->grid.kmax;

    for (int k = 0; k < kmax; k++) {
        for (int j = 0; j < jmax; j++) {
            for (int i = 0; i < imax; i++) {
                if (o->fmt == ASCII) {
                    fprintf(o->fh,
                        "%f %f %f\n",
                        G(vec.u, i, j, k),
                        G(vec.v, i, j, k),
                        G(vec.w, i, j, k));
                } else if (o->fmt == BINARY) {
                    fwrite((double[3]) { floatSwap(G(vec.u, i, j, k)),
                               floatSwap(G(vec.v, i, j, k)),
                               floatSwap(G(vec.w, i, j, k)) },
                        sizeof(double),
                        3,
                        o->fh);
                }
            }
        }
    }
    if (o->fmt == BINARY) fprintf(o->fh, "\n");
}

void vtkVector(VtkOptions* o, char* name, VtkVector vec)
{
	// Write MPI IO code to parallelise writing to a single file.
	// Steps to perform MPI IO
	// 1. Use resetFileview(o); here before starting MPI IO
	
    printf("Register vector %s\n", name);
    if (!isInitialized(o->fh)) return;
	
	// Make sure this is only written by rank 0.
    fprintf(o->fh, "VECTORS %s double\n", name);
	
	
	// 2. Get offsets from all ranks using commGetOffsets from comm.c. This will be 
	// 3. Use the MPI_File_sync and MPI_File_get_size (along with MPI_Offset) to get file size.
	// 4. Use offsets gathered in 3 to create subarray. Create subarray from global problem (Imax, Jmax, Kmax)
	// Note : Feel free to use a temporary array to gather the x, y and z velocities in contiguous fashion to write vectors.
	// 5. Finally use the MPI_File_set_view to set the view in file for each rank.
	// 6. Instead of using writeVector(), create another subarray excluding the ghost cells (imaxlocal+2 etc.)
	// 7. Now use the MPI_File_write using the subarray created in 6 so that each rank can write to the same file at the given displacements.
	
    writeVector(o, vec);
	
	// Binary segment must be terminated with newline character
    // resetFileview(o);
    // if (commIsMaster(&o->comm)) {
	// MPI_File_write(o->fh, "\n", 1, MPI_CHAR, MPI_STATUS_IGNORE);
    // }
}

void vtkClose(VtkOptions* o)
{
    fclose(o->fh);
    o->fh = NULL;
}
