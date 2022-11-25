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

