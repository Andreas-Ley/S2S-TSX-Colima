/*
 * Speckle2Speckle based despeckling filter for TerraSAR-X Spotlight mode
 * Copyright (C) 2020  Andreas Ley <mail@andreas-ley.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __S2S_net_TsX_NETWORK_H_INCLUDED__
#define __S2S_net_TsX_NETWORK_H_INCLUDED__

            
#ifdef __cplusplus
extern "C" {
#endif
            
typedef struct {
    unsigned D,H,W,C,N;
    float *data;
} Tensor;

typedef struct {
    float *data;
} ScratchBuffer;

Tensor allocateTensor(unsigned w, unsigned h, unsigned d, unsigned c, unsigned n);
void freeTensor(Tensor *tensor);
void freeBuffer(ScratchBuffer *buffer);
void S2S_net_TsX_network_forward(const Tensor *input, Tensor *output, ScratchBuffer *buffer);

#ifdef __cplusplus
}
#endif

#endif

