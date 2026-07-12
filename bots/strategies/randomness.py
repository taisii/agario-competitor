"""Shared deterministic random primitives for replay-derived policies."""

MASK_64 = (1 << 64) - 1
GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15
UINT64_SCALE = float(1 << 64)


def mix64(value: int) -> int:
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & MASK_64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & MASK_64
    return (value ^ (value >> 31)) & MASK_64


def unit_interval(value: int) -> float:
    return mix64(value) / UINT64_SCALE
