#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#define TILE_DIM 16

/**
 * Tiled CUDA kernel for 1.58-bit GEMM execution.
 * Avoids floating-point multiplications by performing sign-based accumulations.
 */
__global__ void tiled_ternary_gemm_kernel(
    const int8_t* __restrict__ X,         // activations [M, K]
    const uint8_t* __restrict__ W_packed,    // packed ternary weights [K, N] (4 weights per byte)
    float* __restrict__ Y,                // outputs [M, N]
    float scale_x, float scale_w,
    int M, int N, int K
) {
    // Shared memory tiles for activations and weights
    __shared__ int8_t shared_X[TILE_DIM][TILE_DIM];
    __shared__ uint8_t shared_W[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float accum = 0.0f;

    // Loop over shared tiles along K dimension
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load activations tile to shared memory
        if (row < M && (t * TILE_DIM + tx) < K) {
            shared_X[ty][tx] = X[row * K + (t * TILE_DIM + tx)];
        } else {
            shared_X[ty][tx] = 0;
        }

        // Load packed weights tile to shared memory
        // Packed rows represent depth dimensions packed in chunks of 4.
        if (col < N && (t * TILE_DIM + ty) < (K + 3) / 4) {
            shared_W[ty][tx] = W_packed[(t * TILE_DIM + ty) * N + col];
        } else {
            shared_W[ty][tx] = 0;
        }

        __syncthreads();

        // Perform accumulations inside the tile
        // Unpack weights on the fly to accumulate
        for (int k_offset = 0; k_offset < TILE_DIM; ++k_offset) {
            int k_global = t * TILE_DIM + k_offset;
            if (k_global >= K) break;

            int pack_idx = k_offset / 4;
            int bit_offset = (k_offset % 4) * 2;

            // Read the packed byte from shared memory
            uint8_t packed_byte = shared_W[pack_idx][tx];
            uint8_t bits = (packed_byte >> bit_offset) & 0x03;

            // Perform sign-based additions/subtractions (no float multiplications required)
            int8_t act = shared_X[ty][k_offset];
            if (bits == 0) {
                accum -= (float)act; // weight = -1.0
            } else if (bits == 2) {
                accum += (float)act; // weight = 1.0
            }
            // bits == 1 represents weight = 0, no-op
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Y[row * N + col] = accum * scale_x * scale_w;
    }
}

extern "C" void launch_tiled_ternary_gemm(
    const int8_t* X, const uint8_t* W_packed, float* Y,
    float scale_x, float scale_w, int M, int N, int K
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    tiled_ternary_gemm_kernel<<<grid, block>>>(X, W_packed, Y, scale_x, scale_w, M, N, K);
    cudaDeviceSynchronize();
}
