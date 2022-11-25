#include "S2S_net_TsX_network.h"

#include <stdlib.h>

#include <immintrin.h>

extern const float S2S_net_TsX_in_conv_weights[1][5][5][8][1];
extern const float S2S_net_TsX_in_conv_bias[8];
extern const float S2S_net_TsX_S2S_B_0_DS_p_weights[1][2][2][16][8];
extern const float S2S_net_TsX_S2S_B_0_DS_p_bias[16];
extern const float S2S_net_TsX_S2S__D_B_0_0_0_p_weights[1][3][3][16][32];
extern const float S2S_net_TsX_S2S__D_B_0_0_0_p_bias[16];
extern const float S2S_net_TsX_S2S__D_B_0_0_1_p_weights[1][3][3][16][32];
extern const float S2S_net_TsX_S2S__D_B_0_0_1_p_bias[16];
extern const float S2S_net_TsX_S2S__D_B_0_1_0_p_weights[1][3][3][16][32];
extern const float S2S_net_TsX_S2S__D_B_0_1_0_p_bias[16];
extern const float S2S_net_TsX_S2S__D_B_0_1_1_p_weights[1][3][3][16][32];
extern const float S2S_net_TsX_S2S__D_B_0_1_1_p_bias[16];
extern const float S2S_net_TsX_S2S_B_1_DS_p_weights[1][2][2][32][16];
extern const float S2S_net_TsX_S2S_B_1_DS_p_bias[32];
extern const float S2S_net_TsX_S2S__D_B_1_0_0_p_weights[1][3][3][32][64];
extern const float S2S_net_TsX_S2S__D_B_1_0_0_p_bias[32];
extern const float S2S_net_TsX_S2S__D_B_1_0_1_p_weights[1][3][3][32][64];
extern const float S2S_net_TsX_S2S__D_B_1_0_1_p_bias[32];
extern const float S2S_net_TsX_S2S__D_B_1_1_0_p_weights[1][3][3][32][64];
extern const float S2S_net_TsX_S2S__D_B_1_1_0_p_bias[32];
extern const float S2S_net_TsX_S2S__D_B_1_1_1_p_weights[1][3][3][32][64];
extern const float S2S_net_TsX_S2S__D_B_1_1_1_p_bias[32];
extern const float S2S_net_TsX_S2S_B_2_DS_p_weights[1][2][2][64][32];
extern const float S2S_net_TsX_S2S_B_2_DS_p_bias[64];
extern const float S2S_net_TsX_S2S__D_B_2_0_0_p_weights[1][3][3][64][128];
extern const float S2S_net_TsX_S2S__D_B_2_0_0_p_bias[64];
extern const float S2S_net_TsX_S2S__D_B_2_0_1_p_weights[1][3][3][64][128];
extern const float S2S_net_TsX_S2S__D_B_2_0_1_p_bias[64];
extern const float S2S_net_TsX_S2S__D_B_2_1_0_p_weights[1][3][3][64][128];
extern const float S2S_net_TsX_S2S__D_B_2_1_0_p_bias[64];
extern const float S2S_net_TsX_S2S__D_B_2_1_1_p_weights[1][3][3][64][128];
extern const float S2S_net_TsX_S2S__D_B_2_1_1_p_bias[64];
extern const float S2S_net_TsX_S2S_B_2_US_p_weights[1][2][2][32][64];
extern const float S2S_net_TsX_S2S_B_2_US_p_bias[32];
extern const float S2S_net_TsX_S2S__U_B_2_0_0_p_weights[1][3][3][32][64];
extern const float S2S_net_TsX_S2S__U_B_2_0_0_p_bias[32];
extern const float S2S_net_TsX_S2S__U_B_2_0_1_p_weights[1][3][3][32][64];
extern const float S2S_net_TsX_S2S__U_B_2_0_1_p_bias[32];
extern const float S2S_net_TsX_S2S__U_B_2_1_0_p_weights[1][3][3][32][64];
extern const float S2S_net_TsX_S2S__U_B_2_1_0_p_bias[32];
extern const float S2S_net_TsX_S2S__U_B_2_1_1_p_weights[1][3][3][32][64];
extern const float S2S_net_TsX_S2S__U_B_2_1_1_p_bias[32];
extern const float S2S_net_TsX_S2S_B_1_US_p_weights[1][2][2][16][32];
extern const float S2S_net_TsX_S2S_B_1_US_p_bias[16];
extern const float S2S_net_TsX_S2S__U_B_1_0_0_p_weights[1][3][3][16][32];
extern const float S2S_net_TsX_S2S__U_B_1_0_0_p_bias[16];
extern const float S2S_net_TsX_S2S__U_B_1_0_1_p_weights[1][3][3][16][32];
extern const float S2S_net_TsX_S2S__U_B_1_0_1_p_bias[16];
extern const float S2S_net_TsX_S2S__U_B_1_1_0_p_weights[1][3][3][16][32];
extern const float S2S_net_TsX_S2S__U_B_1_1_0_p_bias[16];
extern const float S2S_net_TsX_S2S__U_B_1_1_1_p_weights[1][3][3][16][32];
extern const float S2S_net_TsX_S2S__U_B_1_1_1_p_bias[16];
extern const float S2S_net_TsX_S2S_B_0_US_p_weights[1][2][2][8][16];
extern const float S2S_net_TsX_S2S_B_0_US_p_bias[8];
extern const float S2S_net_TsX_S2S__U_B_0_0_0_p_weights[1][3][3][8][16];
extern const float S2S_net_TsX_S2S__U_B_0_0_0_p_bias[8];
extern const float S2S_net_TsX_S2S__U_B_0_0_1_p_weights[1][3][3][8][16];
extern const float S2S_net_TsX_S2S__U_B_0_0_1_p_bias[8];
extern const float S2S_net_TsX_S2S__U_B_0_1_0_p_weights[1][3][3][8][16];
extern const float S2S_net_TsX_S2S__U_B_0_1_0_p_bias[8];
extern const float S2S_net_TsX_S2S__U_B_0_1_1_p_weights[1][3][3][8][16];
extern const float S2S_net_TsX_S2S__U_B_0_1_1_p_bias[8];
extern const float S2S_net_TsX_out_conv_weights[1][5][5][1][8];
extern const float S2S_net_TsX_out_conv_bias[1];
void conv_N16_5x5x1_CI8_CO1_padd2x2x0(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 5,
        K_H = 5,
        K_D = 1,
        C_O = 1,
        C_I = 8,
        N = 16,
        PADD_X = 2,
        PADD_Y = 2,
        PADD_Z = 0,
        STRIDE_X = 1,
        STRIDE_Y = 1,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = 0; z_k < K_D; z_k++)
                        for (unsigned y_k = 0; y_k < K_H; y_k++)
                            for (unsigned x_k = 0; x_k < K_W; x_k++) {
                                unsigned x_ = x * STRIDE_X + x_k;
                                unsigned y_ = y * STRIDE_Y + y_k;
                                unsigned z_ = z * STRIDE_Z + z_k;

                                if (x_ < PADD_X) continue;
                                x_ -= PADD_X;
                                if (x_ >= src->W) continue;

                                if (y_ < PADD_Y) continue;
                                y_ -= PADD_Y;
                                if (y_ >= src->H) continue;

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void conv_N16_3x3x1_CI16_CO8_padd1x1x0(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 3,
        K_H = 3,
        K_D = 1,
        C_O = 8,
        C_I = 16,
        N = 16,
        PADD_X = 1,
        PADD_Y = 1,
        PADD_Z = 0,
        STRIDE_X = 1,
        STRIDE_Y = 1,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = 0; z_k < K_D; z_k++)
                        for (unsigned y_k = 0; y_k < K_H; y_k++)
                            for (unsigned x_k = 0; x_k < K_W; x_k++) {
                                unsigned x_ = x * STRIDE_X + x_k;
                                unsigned y_ = y * STRIDE_Y + y_k;
                                unsigned z_ = z * STRIDE_Z + z_k;

                                if (x_ < PADD_X) continue;
                                x_ -= PADD_X;
                                if (x_ >= src->W) continue;

                                if (y_ < PADD_Y) continue;
                                y_ -= PADD_Y;
                                if (y_ >= src->H) continue;

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void conv_N16_5x5x1_CI1_CO8_padd2x2x0(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 5,
        K_H = 5,
        K_D = 1,
        C_O = 8,
        C_I = 1,
        N = 16,
        PADD_X = 2,
        PADD_Y = 2,
        PADD_Z = 0,
        STRIDE_X = 1,
        STRIDE_Y = 1,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = 0; z_k < K_D; z_k++)
                        for (unsigned y_k = 0; y_k < K_H; y_k++)
                            for (unsigned x_k = 0; x_k < K_W; x_k++) {
                                unsigned x_ = x * STRIDE_X + x_k;
                                unsigned y_ = y * STRIDE_Y + y_k;
                                unsigned z_ = z * STRIDE_Z + z_k;

                                if (x_ < PADD_X) continue;
                                x_ -= PADD_X;
                                if (x_ >= src->W) continue;

                                if (y_ < PADD_Y) continue;
                                y_ -= PADD_Y;
                                if (y_ >= src->H) continue;

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void conv_N16_2x2x1_CI8_CO16_stride2x2x1(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 2,
        K_H = 2,
        K_D = 1,
        C_O = 16,
        C_I = 8,
        N = 16,
        PADD_X = 0,
        PADD_Y = 0,
        PADD_Z = 0,
        STRIDE_X = 2,
        STRIDE_Y = 2,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = 0; z_k < K_D; z_k++)
                        for (unsigned y_k = 0; y_k < K_H; y_k++)
                            for (unsigned x_k = 0; x_k < K_W; x_k++) {
                                unsigned x_ = x * STRIDE_X + x_k;
                                unsigned y_ = y * STRIDE_Y + y_k;
                                unsigned z_ = z * STRIDE_Z + z_k;

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void conv_N16_3x3x1_CI32_CO16_padd1x1x0(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 3,
        K_H = 3,
        K_D = 1,
        C_O = 16,
        C_I = 32,
        N = 16,
        PADD_X = 1,
        PADD_Y = 1,
        PADD_Z = 0,
        STRIDE_X = 1,
        STRIDE_Y = 1,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = 0; z_k < K_D; z_k++)
                        for (unsigned y_k = 0; y_k < K_H; y_k++)
                            for (unsigned x_k = 0; x_k < K_W; x_k++) {
                                unsigned x_ = x * STRIDE_X + x_k;
                                unsigned y_ = y * STRIDE_Y + y_k;
                                unsigned z_ = z * STRIDE_Z + z_k;

                                if (x_ < PADD_X) continue;
                                x_ -= PADD_X;
                                if (x_ >= src->W) continue;

                                if (y_ < PADD_Y) continue;
                                y_ -= PADD_Y;
                                if (y_ >= src->H) continue;

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void conv_N16_2x2x1_CI16_CO32_stride2x2x1(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 2,
        K_H = 2,
        K_D = 1,
        C_O = 32,
        C_I = 16,
        N = 16,
        PADD_X = 0,
        PADD_Y = 0,
        PADD_Z = 0,
        STRIDE_X = 2,
        STRIDE_Y = 2,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = 0; z_k < K_D; z_k++)
                        for (unsigned y_k = 0; y_k < K_H; y_k++)
                            for (unsigned x_k = 0; x_k < K_W; x_k++) {
                                unsigned x_ = x * STRIDE_X + x_k;
                                unsigned y_ = y * STRIDE_Y + y_k;
                                unsigned z_ = z * STRIDE_Z + z_k;

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void conv_N16_3x3x1_CI64_CO32_padd1x1x0(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 3,
        K_H = 3,
        K_D = 1,
        C_O = 32,
        C_I = 64,
        N = 16,
        PADD_X = 1,
        PADD_Y = 1,
        PADD_Z = 0,
        STRIDE_X = 1,
        STRIDE_Y = 1,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = 0; z_k < K_D; z_k++)
                        for (unsigned y_k = 0; y_k < K_H; y_k++)
                            for (unsigned x_k = 0; x_k < K_W; x_k++) {
                                unsigned x_ = x * STRIDE_X + x_k;
                                unsigned y_ = y * STRIDE_Y + y_k;
                                unsigned z_ = z * STRIDE_Z + z_k;

                                if (x_ < PADD_X) continue;
                                x_ -= PADD_X;
                                if (x_ >= src->W) continue;

                                if (y_ < PADD_Y) continue;
                                y_ -= PADD_Y;
                                if (y_ >= src->H) continue;

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void conv_N16_2x2x1_CI32_CO64_stride2x2x1(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 2,
        K_H = 2,
        K_D = 1,
        C_O = 64,
        C_I = 32,
        N = 16,
        PADD_X = 0,
        PADD_Y = 0,
        PADD_Z = 0,
        STRIDE_X = 2,
        STRIDE_Y = 2,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = 0; z_k < K_D; z_k++)
                        for (unsigned y_k = 0; y_k < K_H; y_k++)
                            for (unsigned x_k = 0; x_k < K_W; x_k++) {
                                unsigned x_ = x * STRIDE_X + x_k;
                                unsigned y_ = y * STRIDE_Y + y_k;
                                unsigned z_ = z * STRIDE_Z + z_k;

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void conv_N16_3x3x1_CI128_CO64_padd1x1x0(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 3,
        K_H = 3,
        K_D = 1,
        C_O = 64,
        C_I = 128,
        N = 16,
        PADD_X = 1,
        PADD_Y = 1,
        PADD_Z = 0,
        STRIDE_X = 1,
        STRIDE_Y = 1,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = 0; z_k < K_D; z_k++)
                        for (unsigned y_k = 0; y_k < K_H; y_k++)
                            for (unsigned x_k = 0; x_k < K_W; x_k++) {
                                unsigned x_ = x * STRIDE_X + x_k;
                                unsigned y_ = y * STRIDE_Y + y_k;
                                unsigned z_ = z * STRIDE_Z + z_k;

                                if (x_ < PADD_X) continue;
                                x_ -= PADD_X;
                                if (x_ >= src->W) continue;

                                if (y_ < PADD_Y) continue;
                                y_ -= PADD_Y;
                                if (y_ >= src->H) continue;

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void transp_conv_N16_2x2x1_CI16_CO8_stride2x2x1(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 2,
        K_H = 2,
        K_D = 1,
        C_O = 8,
        C_I = 16,
        N = 16,
        PADD_X = 0,
        PADD_Y = 0,
        PADD_Z = 0,
        STRIDE_X = 2,
        STRIDE_Y = 2,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    
    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = STRIDE_Z-1-z % STRIDE_Z; z_k < K_D; z_k+=STRIDE_Z)
                        for (unsigned y_k = STRIDE_Y-1-y % STRIDE_Y; y_k < K_H; y_k+=STRIDE_Y)
                            for (unsigned x_k = STRIDE_X-1-x % STRIDE_X; x_k < K_W; x_k+=STRIDE_X) {
                                unsigned x_ = (x + x_k) / STRIDE_X;
                                unsigned y_ = (y + y_k) / STRIDE_Y;
                                unsigned z_ = (z + z_k) / STRIDE_Z;
                                

                                

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void transp_conv_N16_2x2x1_CI32_CO16_stride2x2x1(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 2,
        K_H = 2,
        K_D = 1,
        C_O = 16,
        C_I = 32,
        N = 16,
        PADD_X = 0,
        PADD_Y = 0,
        PADD_Z = 0,
        STRIDE_X = 2,
        STRIDE_Y = 2,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    
    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = STRIDE_Z-1-z % STRIDE_Z; z_k < K_D; z_k+=STRIDE_Z)
                        for (unsigned y_k = STRIDE_Y-1-y % STRIDE_Y; y_k < K_H; y_k+=STRIDE_Y)
                            for (unsigned x_k = STRIDE_X-1-x % STRIDE_X; x_k < K_W; x_k+=STRIDE_X) {
                                unsigned x_ = (x + x_k) / STRIDE_X;
                                unsigned y_ = (y + y_k) / STRIDE_Y;
                                unsigned z_ = (z + z_k) / STRIDE_Z;
                                

                                

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}
void transp_conv_N16_2x2x1_CI64_CO32_stride2x2x1(Tensor *dst, const Tensor *src, const float *weights, const float *bias)
{
    enum { 
        K_W = 2,
        K_H = 2,
        K_D = 1,
        C_O = 32,
        C_I = 64,
        N = 16,
        PADD_X = 0,
        PADD_Y = 0,
        PADD_Z = 0,
        STRIDE_X = 2,
        STRIDE_Y = 2,
        STRIDE_Z = 1,
        UPSAMPLE_X = 1,
        UPSAMPLE_Y = 1,
        UPSAMPLE_Z = 1,
    };

    
    const unsigned inputStride_C = N;
    const unsigned inputStride_X = N * C_I;
    const unsigned inputStride_Y = N * C_I * src->W;
    const unsigned inputStride_Z = N * C_I * src->W * src->H;
    
    const unsigned outputStride_C = N;
    const unsigned outputStride_X = N * C_O;
    const unsigned outputStride_Y = N * C_O * dst->W;
    const unsigned outputStride_Z = N * C_O * dst->W * dst->H;

    
    #pragma omp parallel for collapse(2)
    for (unsigned z = 0; z < dst->D; z++)
        for (unsigned y = 0; y < dst->H; y++)
            for (unsigned x = 0; x < dst->W; x++)
                for (unsigned c_o = 0; c_o < C_O; c_o++) {
                    __m256 bias_value = _mm256_broadcast_ss(&bias[c_o]);
                    
                    __m256 sum[N/8];
                    for (unsigned n = 0; n < N; n+=8)
                        sum[n/8] = bias_value;
                
                    for (unsigned z_k = STRIDE_Z-1-z % STRIDE_Z; z_k < K_D; z_k+=STRIDE_Z)
                        for (unsigned y_k = STRIDE_Y-1-y % STRIDE_Y; y_k < K_H; y_k+=STRIDE_Y)
                            for (unsigned x_k = STRIDE_X-1-x % STRIDE_X; x_k < K_W; x_k+=STRIDE_X) {
                                unsigned x_ = (x + x_k) / STRIDE_X;
                                unsigned y_ = (y + y_k) / STRIDE_Y;
                                unsigned z_ = (z + z_k) / STRIDE_Z;
                                

                                

                                for (unsigned c_i = 0; c_i < C_I; c_i++) {
                                    __m256 kernel_value = _mm256_broadcast_ss(&weights[
                                                                z_k * (K_H * K_W * C_O * C_I) +
                                                                y_k * (K_W * C_O * C_I) +
                                                                x_k * (C_O * C_I) +
                                                                c_o * (C_I) +
                                                                c_i
                                                            ]);
                                    
                                    for (unsigned n = 0; n < N; n+=8)
                                        sum[n/8] = _mm256_add_ps(sum[n/8], 
                                                        _mm256_mul_ps(kernel_value, 
                                                                _mm256_loadu_ps(&src->data[
                                                                z_ * inputStride_Z +
                                                                y_ * inputStride_Y +
                                                                x_ * inputStride_X +
                                                                c_i * inputStride_C +
                                                                n
                                                            ])));
                                }
                            }
                    for (unsigned n = 0; n < N; n+=8)
                        _mm256_storeu_ps(&dst->data[
                            z * outputStride_Z +
                            y * outputStride_Y +
                            x * outputStride_X +
                            c_o * outputStride_C +
                            n], sum[n/8]);
                }
                
}

void addTensors(Tensor *output, const Tensor *input0, const Tensor *input1)
{
    const unsigned count = input0->W * input0->H * input0->D * input0->C * input0->N;
    for (unsigned i = 0; i < count; i++) {
        output->data[i] = input0->data[i] + input1->data[i];
    }
}

float max(float a, float b) {
    return a>b?a:b;
}


void ReLU(Tensor *output, const Tensor *input)
{
    const unsigned count = input->W * input->H * input->D * input->C * input->N;
    for (unsigned i = 0; i < count; i++) {
        float v = input->data[i];
        v = max(v, 0.0f);
        output->data[i] = v;
    }
}

void ConcatReLU(Tensor *output, const Tensor *input)
{
    const unsigned cells = input->W * input->H * input->D;
    const unsigned CN = input->C * input->N;
    for (unsigned i = 0; i < cells; i++) {
        for (unsigned j = 0; j < CN; j++) {
            float v = input->data[i*CN+j];
            float a = max(v, 0.0f);
            float b = max(-v, 0.0f);
            output->data[i*CN*2 + 0*CN + j] = a;
            output->data[i*CN*2 + 1*CN + j] = b;
        }
    }
}
Tensor allocateTensor(unsigned w, unsigned h, unsigned d, unsigned c, unsigned n)
{
    Tensor tensor;
    memset(&tensor, 0, sizeof(Tensor));
    tensor.data = (float*) malloc(w*h*d*c*n*sizeof(float));
    if (tensor.data != 0) {
        tensor.W = w;
        tensor.H = h;
        tensor.D = d;
        tensor.C = c;
        tensor.N = n;
    }
    return tensor;
}
void freeTensor(Tensor *tensor)
{
    if (tensor->data != 0)
        free(tensor->data);
    memset(tensor, 0, sizeof(Tensor));
}
void freeBuffer(ScratchBuffer *buffer)
{
    if (buffer->data != 0)
        free(buffer->data);
    memset(buffer, 0, sizeof(ScratchBuffer));
}
void S2S_net_TsX_network_forward(const Tensor *input, Tensor *output, ScratchBuffer *buffer)
{
    Tensor t_11 = allocateTensor(64, 64, 1, 64, 16); // 16384 KB
    Tensor t_10 = allocateTensor(64, 64, 1, 64, 16); // 16384 KB
    Tensor t_12 = allocateTensor(64, 64, 1, 128, 16); // 32768 KB
    Tensor t_7 = allocateTensor(128, 128, 1, 32, 16); // 32768 KB
    Tensor t_8 = allocateTensor(128, 128, 1, 32, 16); // 32768 KB
    Tensor t_9 = allocateTensor(128, 128, 1, 64, 16); // 65536 KB
    Tensor t_4 = allocateTensor(256, 256, 1, 16, 16); // 65536 KB
    Tensor t_5 = allocateTensor(256, 256, 1, 16, 16); // 65536 KB
    Tensor t_6 = allocateTensor(256, 256, 1, 32, 16); // 131072 KB
    Tensor t_0 = allocateTensor(512, 512, 1, 1, 16); // 16384 KB
    Tensor t_1 = allocateTensor(512, 512, 1, 8, 16); // 131072 KB
    Tensor t_2 = allocateTensor(512, 512, 1, 8, 16); // 131072 KB
    Tensor t_3 = allocateTensor(512, 512, 1, 16, 16); // 262144 KB
    // 976 MB total
    conv_N16_5x5x1_CI1_CO8_padd2x2x0(&t_2, input, S2S_net_TsX_in_conv_weights, S2S_net_TsX_in_conv_bias);
    conv_N16_2x2x1_CI8_CO16_stride2x2x1(&t_5, &t_2, S2S_net_TsX_S2S_B_0_DS_p_weights, S2S_net_TsX_S2S_B_0_DS_p_bias);
    ConcatReLU(&t_6, &t_5);
    conv_N16_3x3x1_CI32_CO16_padd1x1x0(&t_4, &t_6, S2S_net_TsX_S2S__D_B_0_0_0_p_weights, S2S_net_TsX_S2S__D_B_0_0_0_p_bias);
    ConcatReLU(&t_6, &t_4);
    conv_N16_3x3x1_CI32_CO16_padd1x1x0(&t_4, &t_6, S2S_net_TsX_S2S__D_B_0_0_1_p_weights, S2S_net_TsX_S2S__D_B_0_0_1_p_bias);
    addTensors(&t_5, &t_5, &t_4);
    ConcatReLU(&t_6, &t_5);
    conv_N16_3x3x1_CI32_CO16_padd1x1x0(&t_4, &t_6, S2S_net_TsX_S2S__D_B_0_1_0_p_weights, S2S_net_TsX_S2S__D_B_0_1_0_p_bias);
    ConcatReLU(&t_6, &t_4);
    conv_N16_3x3x1_CI32_CO16_padd1x1x0(&t_4, &t_6, S2S_net_TsX_S2S__D_B_0_1_1_p_weights, S2S_net_TsX_S2S__D_B_0_1_1_p_bias);
    addTensors(&t_5, &t_5, &t_4);
    conv_N16_2x2x1_CI16_CO32_stride2x2x1(&t_8, &t_5, S2S_net_TsX_S2S_B_1_DS_p_weights, S2S_net_TsX_S2S_B_1_DS_p_bias);
    ConcatReLU(&t_9, &t_8);
    conv_N16_3x3x1_CI64_CO32_padd1x1x0(&t_7, &t_9, S2S_net_TsX_S2S__D_B_1_0_0_p_weights, S2S_net_TsX_S2S__D_B_1_0_0_p_bias);
    ConcatReLU(&t_9, &t_7);
    conv_N16_3x3x1_CI64_CO32_padd1x1x0(&t_7, &t_9, S2S_net_TsX_S2S__D_B_1_0_1_p_weights, S2S_net_TsX_S2S__D_B_1_0_1_p_bias);
    addTensors(&t_8, &t_8, &t_7);
    ConcatReLU(&t_9, &t_8);
    conv_N16_3x3x1_CI64_CO32_padd1x1x0(&t_7, &t_9, S2S_net_TsX_S2S__D_B_1_1_0_p_weights, S2S_net_TsX_S2S__D_B_1_1_0_p_bias);
    ConcatReLU(&t_9, &t_7);
    conv_N16_3x3x1_CI64_CO32_padd1x1x0(&t_7, &t_9, S2S_net_TsX_S2S__D_B_1_1_1_p_weights, S2S_net_TsX_S2S__D_B_1_1_1_p_bias);
    addTensors(&t_8, &t_8, &t_7);
    conv_N16_2x2x1_CI32_CO64_stride2x2x1(&t_10, &t_8, S2S_net_TsX_S2S_B_2_DS_p_weights, S2S_net_TsX_S2S_B_2_DS_p_bias);
    ConcatReLU(&t_12, &t_10);
    conv_N16_3x3x1_CI128_CO64_padd1x1x0(&t_11, &t_12, S2S_net_TsX_S2S__D_B_2_0_0_p_weights, S2S_net_TsX_S2S__D_B_2_0_0_p_bias);
    ConcatReLU(&t_12, &t_11);
    conv_N16_3x3x1_CI128_CO64_padd1x1x0(&t_11, &t_12, S2S_net_TsX_S2S__D_B_2_0_1_p_weights, S2S_net_TsX_S2S__D_B_2_0_1_p_bias);
    addTensors(&t_10, &t_10, &t_11);
    ConcatReLU(&t_12, &t_10);
    conv_N16_3x3x1_CI128_CO64_padd1x1x0(&t_11, &t_12, S2S_net_TsX_S2S__D_B_2_1_0_p_weights, S2S_net_TsX_S2S__D_B_2_1_0_p_bias);
    ConcatReLU(&t_12, &t_11);
    conv_N16_3x3x1_CI128_CO64_padd1x1x0(&t_11, &t_12, S2S_net_TsX_S2S__D_B_2_1_1_p_weights, S2S_net_TsX_S2S__D_B_2_1_1_p_bias);
    addTensors(&t_10, &t_10, &t_11);
    transp_conv_N16_2x2x1_CI64_CO32_stride2x2x1(&t_7, &t_10, S2S_net_TsX_S2S_B_2_US_p_weights, S2S_net_TsX_S2S_B_2_US_p_bias);
    addTensors(&t_7, &t_7, &t_8);
    ConcatReLU(&t_9, &t_7);
    conv_N16_3x3x1_CI64_CO32_padd1x1x0(&t_8, &t_9, S2S_net_TsX_S2S__U_B_2_0_0_p_weights, S2S_net_TsX_S2S__U_B_2_0_0_p_bias);
    ConcatReLU(&t_9, &t_8);
    conv_N16_3x3x1_CI64_CO32_padd1x1x0(&t_8, &t_9, S2S_net_TsX_S2S__U_B_2_0_1_p_weights, S2S_net_TsX_S2S__U_B_2_0_1_p_bias);
    addTensors(&t_7, &t_7, &t_8);
    ConcatReLU(&t_9, &t_7);
    conv_N16_3x3x1_CI64_CO32_padd1x1x0(&t_8, &t_9, S2S_net_TsX_S2S__U_B_2_1_0_p_weights, S2S_net_TsX_S2S__U_B_2_1_0_p_bias);
    ConcatReLU(&t_9, &t_8);
    conv_N16_3x3x1_CI64_CO32_padd1x1x0(&t_8, &t_9, S2S_net_TsX_S2S__U_B_2_1_1_p_weights, S2S_net_TsX_S2S__U_B_2_1_1_p_bias);
    addTensors(&t_7, &t_7, &t_8);
    transp_conv_N16_2x2x1_CI32_CO16_stride2x2x1(&t_4, &t_7, S2S_net_TsX_S2S_B_1_US_p_weights, S2S_net_TsX_S2S_B_1_US_p_bias);
    addTensors(&t_4, &t_4, &t_5);
    ConcatReLU(&t_6, &t_4);
    conv_N16_3x3x1_CI32_CO16_padd1x1x0(&t_5, &t_6, S2S_net_TsX_S2S__U_B_1_0_0_p_weights, S2S_net_TsX_S2S__U_B_1_0_0_p_bias);
    ConcatReLU(&t_6, &t_5);
    conv_N16_3x3x1_CI32_CO16_padd1x1x0(&t_5, &t_6, S2S_net_TsX_S2S__U_B_1_0_1_p_weights, S2S_net_TsX_S2S__U_B_1_0_1_p_bias);
    addTensors(&t_4, &t_4, &t_5);
    ConcatReLU(&t_6, &t_4);
    conv_N16_3x3x1_CI32_CO16_padd1x1x0(&t_5, &t_6, S2S_net_TsX_S2S__U_B_1_1_0_p_weights, S2S_net_TsX_S2S__U_B_1_1_0_p_bias);
    ConcatReLU(&t_6, &t_5);
    conv_N16_3x3x1_CI32_CO16_padd1x1x0(&t_5, &t_6, S2S_net_TsX_S2S__U_B_1_1_1_p_weights, S2S_net_TsX_S2S__U_B_1_1_1_p_bias);
    addTensors(&t_4, &t_4, &t_5);
    transp_conv_N16_2x2x1_CI16_CO8_stride2x2x1(&t_1, &t_4, S2S_net_TsX_S2S_B_0_US_p_weights, S2S_net_TsX_S2S_B_0_US_p_bias);
    addTensors(&t_1, &t_1, &t_2);
    ConcatReLU(&t_3, &t_1);
    conv_N16_3x3x1_CI16_CO8_padd1x1x0(&t_2, &t_3, S2S_net_TsX_S2S__U_B_0_0_0_p_weights, S2S_net_TsX_S2S__U_B_0_0_0_p_bias);
    ConcatReLU(&t_3, &t_2);
    conv_N16_3x3x1_CI16_CO8_padd1x1x0(&t_2, &t_3, S2S_net_TsX_S2S__U_B_0_0_1_p_weights, S2S_net_TsX_S2S__U_B_0_0_1_p_bias);
    addTensors(&t_1, &t_1, &t_2);
    ConcatReLU(&t_3, &t_1);
    conv_N16_3x3x1_CI16_CO8_padd1x1x0(&t_2, &t_3, S2S_net_TsX_S2S__U_B_0_1_0_p_weights, S2S_net_TsX_S2S__U_B_0_1_0_p_bias);
    ConcatReLU(&t_3, &t_2);
    conv_N16_3x3x1_CI16_CO8_padd1x1x0(&t_2, &t_3, S2S_net_TsX_S2S__U_B_0_1_1_p_weights, S2S_net_TsX_S2S__U_B_0_1_1_p_bias);
    addTensors(&t_1, &t_1, &t_2);
    conv_N16_5x5x1_CI8_CO1_padd2x2x0(output, &t_1, S2S_net_TsX_out_conv_weights, S2S_net_TsX_out_conv_bias);
    freeTensor(&t_11);
    freeTensor(&t_10);
    freeTensor(&t_12);
    freeTensor(&t_7);
    freeTensor(&t_8);
    freeTensor(&t_9);
    freeTensor(&t_4);
    freeTensor(&t_5);
    freeTensor(&t_6);
    freeTensor(&t_0);
    freeTensor(&t_1);
    freeTensor(&t_2);
    freeTensor(&t_3);
}
