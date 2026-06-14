#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <new>

namespace bitnet {

/**
 * High-performance, zero-fragmentation Memory Arena Allocator.
 * Pre-allocates single continuous buffer to eliminate malloc overhead on edge loops.
 */
class MemoryArena {
private:
    uint8_t* m_buffer;
    size_t m_capacity;
    size_t m_offset;

public:
    explicit MemoryArena(size_t capacity)
        : m_capacity(capacity), m_offset(0) {
        m_buffer = static_cast<uint8_t*>(::operator new(capacity, std::align_val_t{64}));
    }

    ~MemoryArena() {
        if (m_buffer) {
            ::operator delete(m_buffer, std::align_val_t{64});
        }
    }

    // Disable copy constructors to prevent double frees
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    /**
     * Allocates memory chunk aligned to requested boundary.
     */
    void* allocate(size_t bytes, size_t alignment = 64) {
        size_t current_ptr = reinterpret_cast<size_t>(m_buffer + m_offset);
        size_t align_offset = (alignment - (current_ptr % alignment)) % alignment;
        
        if (m_offset + align_offset + bytes > m_capacity) {
            throw std::bad_alloc();
        }

        m_offset += align_offset;
        void* ptr = m_buffer + m_offset;
        m_offset += bytes;
        return ptr;
    }

    /**
     * Resets the allocation offset to reuse the same memory block.
     */
    void reset() noexcept {
        m_offset = 0;
    }

    [[nodiscard]] size_t capacity() const noexcept { return m_capacity; }
    [[nodiscard]] size_t allocated_bytes() const noexcept { return m_offset; }
};

} // namespace bitnet
