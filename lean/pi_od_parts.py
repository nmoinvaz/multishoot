#!/usr/bin/env python3
"""PI_OD's parts in src/periodic.rs against the integers that Multishoot/PiOD.lean proves things of.

Part 0 is std::f64::consts::PI, and parts 1 to 7 are decimal literals: each must be m/2^k as piM and piK give it, and
each literal's digits must be piDecN's and piDecK's. The controls then change the last part's integer, and apart the
last literal's digits, in the Lean tables, and each must make the comparison fail.
"""

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rust_parts():
    src = (ROOT / "src/periodic.rs").read_text()
    body = re.search(r"pub const PI_OD: OD = OD \{\s*x: \[(.*?)\]", src, re.S).group(1)
    return [s.strip().replace("_f64", "").replace("_", "") for s in body.split(",") if s.strip()]


def lean_tables():
    src = (ROOT / "lean/Multishoot/PiOD.lean").read_text()

    def table(name):
        body = re.search(rf"def {name} : ℕ → [ℤℕ]\n(.*?)\n  \| _ =>", src, re.S).group(1)
        return {int(i): int(v) for i, v in re.findall(r"\| (\d+) => (-?\d+)", body)}

    return {name: table(name) for name in ("piM", "piK", "piDecN", "piDecK")}


def binary(x):
    """x = m/2^k with 2^52 <= |m| < 2^53"""
    f, e = math.frexp(x)
    return int(f * 2**53), 53 - e


def decimal(s):
    """the literal s = n/10^k"""
    mant, _, exp = s.lower().partition("e")
    whole, _, frac = mant.partition(".")
    return int(whole + frac), len(frac) - int(exp or 0)


def mismatches(parts, t):
    bad = []
    if len(parts) != 8 or parts[0] != "std::f64::consts::PI":
        bad.append(f"PI_OD is not std::f64::consts::PI and seven literals: {parts}")
        return bad
    for i, s in enumerate(parts):
        x = math.pi if i == 0 else float(s)
        if binary(x) != (t["piM"][i], t["piK"][i]):
            bad.append(f"part {i}: {s} is {binary(x)}, the Lean tables say {(t['piM'][i], t['piK'][i])}")
        if i > 0 and decimal(s) != (t["piDecN"][i], t["piDecK"][i]):
            bad.append(f"part {i}: {s} has digits {decimal(s)}, the Lean tables say {(t['piDecN'][i], t['piDecK'][i])}")
    return bad


def main():
    parts, t = rust_parts(), lean_tables()
    bad = mismatches(parts, t)
    for b in bad:
        print(b)
    if bad:
        return 1
    # one per comparison: the binary one sees only piM and piK, the decimal one only piDecN and piDecK
    for table, wrong in (("piM", "the last part one ulp up"), ("piDecN", "the last literal one unit up")):
        t = lean_tables()
        t[table][7] += 1
        if not mismatches(parts, t):
            print(f"the control failed: {wrong} matches")
            return 1
    print("PI_OD's eight parts are the integers of Multishoot/PiOD.lean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
