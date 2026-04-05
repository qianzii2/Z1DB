from __future__ import annotations
"""HyperLogLog — APPROX_COUNT_DISTINCT. Paper: Flajolet et al., 2007.
Error < 2.3% with 2KB memory. Mergeable for parallel/distributed."""
import math
from metal.hash import murmur3_64


class HyperLogLog:
    """Cardinality estimator using 2^p registers (p=11 → 2048 registers ≈ 2KB)."""

    __slots__ = ('_p', '_m', '_registers', '_alpha')

    def __init__(self, p: int = 11) -> None:
        self._p = p
        self._m = 1 << p  # number of registers
        self._registers = bytearray(self._m)
        # Alpha constant for bias correction
        if self._m == 16:
            self._alpha = 0.673
        elif self._m == 32:
            self._alpha = 0.697
        elif self._m == 64:
            self._alpha = 0.709
        else:
            self._alpha = 0.7213 / (1 + 1.079 / self._m)

    def add(self, value: object) -> None:
        """Add a value to the sketch."""
        if isinstance(value, str):
            h = murmur3_64(value.encode('utf-8'))
        elif isinstance(value, int):
            h = murmur3_64(value.to_bytes(8, 'little', signed=True))
        elif isinstance(value, float):
            import struct
            h = murmur3_64(struct.pack('d', value))
        elif isinstance(value, bytes):
            h = murmur3_64(value)
        elif value is None:
            return
        else:
            h = murmur3_64(str(value).encode('utf-8'))

        h &= 0xFFFFFFFFFFFFFFFF
        idx = h >> (64 - self._p)  # top p bits → register index
        w = h & ((1 << (64 - self._p)) - 1)  # remaining bits
        # Count leading zeros + 1 in remaining bits
        rank = self._rho(w, 64 - self._p)
        if rank > self._registers[idx]:
            self._registers[idx] = rank

    def estimate(self) -> int:
        """Return estimated cardinality."""
        # Harmonic mean of 2^(-register[i])
        indicator = sum(2.0 ** (-r) for r in self._registers)
        raw = self._alpha * self._m * self._m / indicator

        # Small range correction
        if raw <= 2.5 * self._m:
            zeros = self._registers.count(0)
            if zeros > 0:
                raw = self._m * math.log(self._m / zeros)

        # Large range correction (for 64-bit hash)
        if raw > (1 << 32) / 30.0:
            raw = -(1 << 64) * math.log(1 - raw / (1 << 64))

        return int(raw + 0.5)

    def merge(self, other: HyperLogLog) -> None:
        """O(m) merge — take max of each register."""
        if self._m != other._m:
            raise ValueError("cannot merge HLLs of different precision")
        for i in range(self._m):
            if other._registers[i] > self._registers[i]:
                self._registers[i] = other._registers[i]

    def relative_error(self) -> float:
        """Theoretical relative standard error."""
        return 1.04 / math.sqrt(self._m)

    @staticmethod
    def _rho(w: int, max_bits: int) -> int:
        """Position of first 1-bit from the left (1-indexed). 0 → max_bits+1."""
        if w == 0:
            return max_bits + 1
        r = 1
        while (w & (1 << (max_bits - r))) == 0 and r <= max_bits:
            r += 1
        return r
