#pragma once

#include <cstdint>

namespace ClefGateUp {
constexpr uint32_t Hidden = 5120;
constexpr uint32_t Intermediate = 8704;
constexpr uint32_t Groups = 24;
constexpr uint32_t RingSlots = 2;

enum Mode : uint32_t {
    Raw = 0,
    Fused = 1,
    ShortRaw = 2,
    ShortFused = 3,
};

constexpr bool IsFused(uint32_t mode)
{
    return mode == Fused || mode == ShortFused;
}

constexpr bool IsShort(uint32_t mode)
{
    return mode == ShortRaw || mode == ShortFused;
}

struct TilingData {
    uint32_t tokens;
    uint32_t mode;
};
static_assert(sizeof(TilingData) == 8);
}  // namespace ClefGateUp
