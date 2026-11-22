#pragma once

#include <cstdint>

namespace ClefMlp {
constexpr uint32_t H = 5120;
constexpr uint32_t I = 8704;
constexpr uint32_t TileM = 128;
constexpr uint32_t TileN = 256;
constexpr uint32_t Groups = 24;
constexpr uint32_t RingSlots = 2;
constexpr uint32_t SlotRow = 2 * TileN;
constexpr uint32_t SlotElements = TileM * SlotRow;
constexpr uint32_t RingElements = Groups * RingSlots * SlotElements;

struct TilingData {
    uint32_t tokens;
    uint32_t diagnostic;
};
static_assert(sizeof(TilingData) == 8);
}  // namespace ClefMlp
