#include <iostream>
#include <vector>
#include <chrono>
#include <memory>

#include "../../include/memory_arena.hpp"
#include "../../include/kv_cache.hpp"
#include "../../include/weight_packing.hpp"

// Declaration of the CUDA kernel launcher
#ifdef USE_CUDA
extern "C" void launch_tiled_ternary_gemm(
    const int8_t* X, const uint8_t* W_packed, float* Y,
    float scale_x, float scale_w, int M, int N, int K
);
#else
// High-performance CPU fallback for systems without CUDA support
void launch_tiled_ternary_gemm(
    const int8_t* X, const uint8_t* W_packed, float* Y,
    float scale_x, float scale_w, int M, int N, int K
) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float accum = 0.0f;
            for (int k_pack = 0; k_pack < (K + 3) / 4; ++k_pack) {
                uint8_t packed_val = W_packed[k_pack * N + c];
                for (int i = 0; i < 4; ++i) {
                    int k = k_pack * 4 + i;
                    if (k >= K) break;
                    
                    uint8_t bit_pair = (packed_val >> (i * 2)) & 0x03;
                    float weight = 0.0f;
                    if (bit_pair == 0) weight = -1.0f;
                    else if (bit_pair == 2) weight = 1.0f;
                    
                    accum += (float)X[r * K + k] * weight;
                }
            }
            Y[r * N + c] = accum * scale_x * scale_w;
        }
    }
}
#endif

int main() {
    std::cout << "[Extended Modular BitNet Engine] Starting up...\n";

    // Step 1: Initialize Arena Allocator (Preallocate 10 MB for testing)
    bitnet::MemoryArena arena(10 * 1024 * 1024);
    std::cout << "Allocated Memory Arena of capacity: " << arena.capacity() << " bytes.\n";

    // Step 2: Initialize Paged KV Cache
    // 64 blocks, 8 tokens block size, 4 attention layers, 64 dimension per head
    bitnet::PagedKVCache kv_cache(64, 8, 4, 64);
    uint32_t active_seq = 0;
    uint32_t active_block = kv_cache.allocate_block(active_seq);
    std::cout << "Paged KV Cache: Allocated block " << active_block << " for sequence " << active_seq << ".\n";

    // Dimensions for testing matrix multiplication
    const int M = 1;
    const int K = 1024;
    const int N = 2048;

    // Step 3: Allocate workspace buffers from Arena (Zero Copy pointers)
    int8_t* X = static_cast<int8_t*>(arena.allocate(M * K * sizeof(int8_t)));
    float* Y = static_cast<float*>(arena.allocate(M * N * sizeof(float)));

    // Initialize mock activations in arena
    for (int i = 0; i < M * K; ++i) {
        X[i] = (i % 10) - 5;
    }

    // Step 4: Initialize raw ternary weights and pack
    std::vector<int8_t> raw_weights(K * N);
    for (int i = 0; i < K * N; ++i) {
        int r = i % 3;
        raw_weights[i] = (r == 0) ? -1 : ((r == 1) ? 1 : 0);
    }

    std::vector<uint8_t> packed_weights;
    bitnet::WeightPacker::pack(raw_weights, packed_weights);
    std::cout << "Packed " << raw_weights.size() << " ternary elements to " << packed_weights.size() << " bytes.\n";

    // Allocate packed weights block in Arena and copy
    uint8_t* gpu_packed_weights = static_cast<uint8_t*>(arena.allocate(packed_weights.size() * sizeof(uint8_t)));
    std::copy(packed_weights.begin(), packed_weights.end(), gpu_packed_weights);

    // Step 5: Execute matrix multiplication execution kernel
    float scale_x = 0.125f;
    float scale_w = 1.0f;

    std::cout << "Running GEMM kernel...\n";
    auto start = std::chrono::high_resolution_clock::now();

    launch_tiled_ternary_gemm(X, gpu_packed_weights, Y, scale_x, scale_w, M, N, K);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Tiled GEMM execution complete.\n";
    std::cout << "Duration: " << duration.count() << " ms\n";
    std::cout << "Output values (first 5 elements): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << Y[i] << " ";
    }
    std::cout << "\n";

    // Free KV Cache resources
    kv_cache.free_sequence(active_seq);
    std::cout << "Released Sequence contexts in KV cache.\n";

    return 0;
}
