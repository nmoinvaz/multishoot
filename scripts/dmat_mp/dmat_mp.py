#!/usr/bin/env python3
"""The reference entries of tests/dmat_c.rs: the folded d/dtheta of the three grids at M = 8 and 16 from mpmath at 120
digits, each entry split greedily into hi = RN(D), lo = RN(D - hi), c = RN(D - hi - lo), written as the f64s' bits. A
part below 2^-300 of the entry's scale, the sum of max(|t|, 1) over its terms t, is the evaluation's rounding of a value
the parts before hold exactly (a cotangent of an odd multiple of pi / 2, or images that sum to a dyadic rational) and is set to zero, as in
src/periodic.rs.

    python3 scripts/dmat_mp/dmat_mp.py > tests/data/dmat_mp.txt

D_ij = sum over the images jj of j (j alone on the full grid, j and j + M with the sign -1 on the odd half grid and +1
on the even one), jj != i, of (-1)^(i - jj) cot((i - jj) pi / N) / 2, N = M on the full grid and 2M on the half grids:
the formula of src/periodic.rs, evaluated independently of its octo-double arithmetic."""
import struct

import mpmath as mp

mp.mp.dps = 120

# the flush threshold, relative to the entry's scale
TINY = mp.mpf(2) ** -300


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def entry(parity, m, i, j):
    n = m if parity == "full" else 2 * m
    images = {"full": [(j, 1)], "odd": [(j, 1), (j + m, -1)], "even": [(j, 1), (j + m, 1)]}[parity]
    v, scale = mp.mpf(0), mp.mpf(0)
    for jj, sgn in images:
        if jj != i:
            d = i - jj
            t = sgn * mp.mpf(0.5) * (-1) ** d * mp.cot(d * mp.pi / n)
            v += t
            scale += max(abs(t), 1)
    return v, scale


def split(v, scale):
    tiny = TINY * scale

    def part(r):
        return 0.0 if abs(r) <= tiny else float(r)

    hi = part(v)
    r = v - mp.mpf(hi)
    lo = part(r)
    c = part(r - mp.mpf(lo))
    return hi, lo, c


def main():
    print("# parity m i j hi lo c: the f64 bits in hex of RN(D), RN(D - hi), RN(D - hi - lo); scripts/dmat_mp")
    for m in (8, 16):
        for parity in ("odd", "even", "full"):
            for i in range(m):
                for j in range(m):
                    hi, lo, c = split(*entry(parity, m, i, j))
                    print(f"{parity} {m} {i} {j} {bits(hi):016x} {bits(lo):016x} {bits(c):016x}")


if __name__ == "__main__":
    main()
