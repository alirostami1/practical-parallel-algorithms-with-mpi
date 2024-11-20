/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#ifndef AFFINITY_H
#define AFFINITY_H

extern int affinity_getProcessorId();
extern void affinity_pinProcess(int);
extern void affinity_pinThread(int);

#endif /*AFFINITY_H*/
