import numba
import numpy as np

from constants import *
from forward.original import forward_full_a


def forward_full_b(c, steps=ROUNDS):
    for _ in range(steps):
        d = forward1(c)
        c = forward2(d)
    return forward3(c)


def forward1(c):   # sbox 1
    return [CONFUSION[c[j]] for j in range(32)]


def forward2(d):  # dbox
    c = [0]*32
    for j in range(32):
        for k in range(32):
            c[j] ^= d[k] * ((DIFFUSION[j] >> k) & 1)
    return c


@numba.njit()
def forward2_(d):
    c = np.zeros(32, dtype=np.uint8)
    for j in range(32):
        for k in range(32):
            diff = ((DIFFUSION[j] >> k) & 1)
            c[j] ^= d[k] * diff
    return c


def forward3(c):  # sbox 2
    return [CONFUSION[c[i * 2]] ^ CONFUSION[c[i * 2 + 1] + 256] for i in range(16)]


if __name__ == '__main__':
    # proof that forward_full_a and forward_full_b are the same
    assert forward_full_a(EX_INP) == forward_full_b(EX_INP)

