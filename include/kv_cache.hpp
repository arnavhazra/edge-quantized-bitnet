#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace bitnet {

/**
 * Paged KV Cache memory manager.
 * Divides continuous generation states into small physical blocks.
 */
class PagedKVCache {
private:
    uint32_t m_num_blocks;
    uint32_t m_block_size; // Tokens per block
    uint32_t m_num_layers;
    uint32_t m_head_dim;
    
    // Free blocks tracking pool
    std::vector<uint32_t> m_free_blocks;
    
    // Maps: session/sequence_id -> list of allocated physical block indices
    std::vector<std::vector<uint32_t>> m_seq_to_blocks;

public:
    PagedKVCache(uint32_t num_blocks, uint32_t block_size, uint32_t num_layers, uint32_t head_dim)
        : m_num_blocks(num_blocks), m_block_size(block_size),
          m_num_layers(num_layers), m_head_dim(head_dim) {
        
        m_free_blocks.reserve(num_blocks);
        for (uint32_t i = 0; i < num_blocks; ++i) {
            m_free_blocks.push_back(i);
        }
        
        // Setup initial slots for active sequences
        m_seq_to_blocks.resize(16); // Support up to 16 concurrent context streams
    }

    /**
     * Allocates block from free block pool for a sequence.
     */
    uint32_t allocate_block(uint32_t seq_id) {
        if (m_free_blocks.empty()) {
            throw std::runtime_error("OOM: KV Cache allocation pool exhausted.");
        }
        
        uint32_t block = m_free_blocks.back();
        m_free_blocks.pop_back();
        
        if (seq_id >= m_seq_to_blocks.size()) {
            m_seq_to_blocks.resize(seq_id + 8);
        }
        m_seq_to_blocks[seq_id].push_back(block);
        return block;
    }

    /**
     * Releases block allocations associated with a finished sequence context.
     */
    void free_sequence(uint32_t seq_id) {
        if (seq_id < m_seq_to_blocks.size()) {
            for (uint32_t block : m_seq_to_blocks[seq_id]) {
                m_free_blocks.push_back(block);
            }
            m_seq_to_blocks[seq_id].clear();
        }
    }

    [[nodiscard]] const std::vector<uint32_t>& get_blocks(uint32_t seq_id) const {
        if (seq_id >= m_seq_to_blocks.size()) {
            throw std::out_of_range("Sequence ID out of bounds.");
        }
        return m_seq_to_blocks[seq_id];
    }
};

} // namespace bitnet
