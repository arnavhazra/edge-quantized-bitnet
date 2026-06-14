#pragma once
#include <vector>
#include <cstdint>
#include <cmath>

namespace bitnet {

/**
 * Handles packing/unpacking of ternary values {-1, 0, 1} to maximize memory bandwidth.
 * Each byte contains 4 values (2 bits per value: 00 -> -1, 01 -> 0, 10 -> 1).
 */
class WeightPacker {
public:
    static void pack(const std::vector<int8_t>& src, std::vector<uint8_t>& dest) {
        size_t num_elements = src.size();
        size_t packed_size = (num_elements + 3) / 4;
        dest.assign(packed_size, 0);

        for (size_t i = 0; i < num_elements; ++i) {
            size_t byte_idx = i / 4;
            size_t bit_offset = (i % 4) * 2;
            
            int8_t val = src[i];
            uint8_t bits = 1; // Default representation for 0
            if (val == -1) bits = 0; // -1
            else if (val == 1) bits = 2; // 1

            // Shift and merge bits into packed byte
            dest[byte_idx] |= (bits << bit_offset);
        }
    }

    static void unpack(const std::vector<uint8_t>& src, std::vector<int8_t>& dest, size_t original_size) {
        dest.resize(original_size);
        for (size_t i = 0; i < original_size; ++i) {
            size_t byte_idx = i / 4;
            size_t bit_offset = (i % 4) * 2;
            
            uint8_t bits = (src[byte_idx] >> bit_offset) & 0x03;
            int8_t val = 0;
            if (bits == 0) val = -1;
            else if (bits == 2) val = 1;
            
            dest[i] = val;
        }
    }
};

} // namespace bitnet
