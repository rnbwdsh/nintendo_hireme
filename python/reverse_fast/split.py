import itertools

import numpy as np
from numba import njit
from numba.typed import List

from constants import *
from forward.original import forward_full_a
from reverse import split
from reverse.split import rfw1, rfw3

rfw1np = List([np.array(rfw1[i], dtype=np.uint8) for i in range(len(rfw1))])
@njit()
def product_jit(arrays):  # https://gist.github.com/hernamesbarbara/68d073f551565de02ac5
    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)), dtype=np.uint8)
    if n == 0:
        return out

    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size

    n = arrays[-1].size
    for k in range(len(arrays)-2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j*m:(j+1)*m,k+1:] = out[0:m,k+1:]
    return out


@njit()
def reverse1(output, rfw1np):
    return product_jit([rfw1np[a] for a in output])


@njit()
def reverse2(d):
    c = np.zeros(32, dtype=np.uint8)
    c[0] = d[0]^d[7]^d[10]^d[12]^d[13]^d[15]^d[18]^d[19]^d[21]^d[22]^d[25]^d[28]^d[29]^d[30]^d[31]
    c[1] = d[1]^d[4]^d[7]^d[10]^d[11]^d[12]^d[14]^d[15]^d[16]^d[18]^d[21]^d[23]^d[25]^d[26]^d[28]
    c[2] = d[2]^d[5]^d[8]^d[11]^d[13]^d[15]^d[16]^d[17]^d[19]^d[20]^d[22]^d[26]^d[27]^d[28]^d[29]
    c[3] = d[3]^d[6]^d[9]^d[12]^d[14]^d[17]^d[18]^d[20]^d[21]^d[23]^d[24]^d[27]^d[28]^d[29]^d[30]
    c[4] = d[3]^d[4]^d[8]^d[9]^d[11]^d[14]^d[17]^d[18]^d[22]^d[23]^d[24]^d[25]^d[26]^d[27]^d[29]
    c[5] = d[0]^d[3]^d[5]^d[8]^d[10]^d[11]^d[14]^d[15]^d[17]^d[19]^d[20]^d[22]^d[24]^d[29]^d[30]
    c[6] = d[1]^d[6]^d[9]^d[11]^d[12]^d[15]^d[16]^d[18]^d[20]^d[21]^d[23]^d[24]^d[25]^d[30]^d[31]
    c[7] = d[2]^d[7]^d[8]^d[10]^d[13]^d[16]^d[17]^d[19]^d[21]^d[22]^d[24]^d[25]^d[26]^d[28]^d[31]
    c[8] = d[2]^d[4]^d[5]^d[7]^d[8]^d[15]^d[17]^d[20]^d[21]^d[22]^d[23]^d[26]^d[27]^d[29]^d[30]
    c[9] = d[2]^d[3]^d[4]^d[6]^d[7]^d[9]^d[12]^d[15]^d[17]^d[18]^d[20]^d[24]^d[26]^d[29]^d[31]
    c[10] = d[0]^d[3]^d[5]^d[7]^d[10]^d[13]^d[18]^d[19]^d[20]^d[21]^d[24]^d[25]^d[27]^d[28]^d[30]
    c[11] = d[1]^d[4]^d[6]^d[11]^d[14]^d[16]^d[19]^d[20]^d[21]^d[22]^d[25]^d[26]^d[28]^d[29]^d[31]
    c[12] = d[0]^d[1]^d[3]^d[6]^d[11]^d[12]^d[16]^d[17]^d[18]^d[19]^d[21]^d[25]^d[26]^d[30]^d[31]
    c[13] = d[0]^d[2]^d[3]^d[6]^d[7]^d[8]^d[11]^d[13]^d[16]^d[21]^d[22]^d[25]^d[27]^d[28]^d[30]
    c[14] = d[1]^d[3]^d[4]^d[7]^d[9]^d[14]^d[16]^d[17]^d[22]^d[23]^d[24]^d[26]^d[28]^d[29]^d[31]
    c[15] = d[0]^d[2]^d[5]^d[10]^d[15]^d[16]^d[17]^d[18]^d[20]^d[23]^d[24]^d[25]^d[27]^d[29]^d[30]
    c[16] = d[2]^d[3]^d[5]^d[6]^d[9]^d[12]^d[13]^d[14]^d[15]^d[16]^d[23]^d[26]^d[28]^d[29]^d[31]
    c[17] = d[0]^d[2]^d[5]^d[7]^d[9]^d[10]^d[12]^d[17]^d[20]^d[23]^d[26]^d[27]^d[28]^d[30]^d[31]
    c[18] = d[0]^d[1]^d[3]^d[4]^d[6]^d[10]^d[11]^d[12]^d[13]^d[18]^d[21]^d[24]^d[27]^d[29]^d[31]
    c[19] = d[1]^d[2]^d[4]^d[5]^d[7]^d[8]^d[11]^d[12]^d[13]^d[14]^d[19]^d[22]^d[25]^d[28]^d[30]
    c[20] = d[1]^d[2]^d[6]^d[7]^d[8]^d[9]^d[10]^d[11]^d[13]^d[19]^d[20]^d[24]^d[25]^d[27]^d[30]
    c[21] = d[1]^d[3]^d[4]^d[6]^d[8]^d[13]^d[14]^d[16]^d[19]^d[21]^d[24]^d[26]^d[27]^d[30]^d[31]
    c[22] = d[0]^d[2]^d[4]^d[5]^d[7]^d[8]^d[9]^d[14]^d[15]^d[17]^d[22]^d[25]^d[27]^d[28]^d[31]
    c[23] = d[0]^d[1]^d[3]^d[5]^d[6]^d[8]^d[9]^d[10]^d[12]^d[15]^d[18]^d[23]^d[24]^d[26]^d[29]
    c[24] = d[1]^d[4]^d[5]^d[6]^d[7]^d[10]^d[11]^d[13]^d[14]^d[18]^d[20]^d[21]^d[23]^d[24]^d[31]
    c[25] = d[1]^d[2]^d[4]^d[8]^d[10]^d[13]^d[15]^d[18]^d[19]^d[20]^d[22]^d[23]^d[25]^d[28]^d[31]
    c[26] = d[2]^d[3]^d[4]^d[5]^d[8]^d[9]^d[11]^d[12]^d[14]^d[16]^d[19]^d[21]^d[23]^d[26]^d[29]
    c[27] = d[0]^d[3]^d[4]^d[5]^d[6]^d[9]^d[10]^d[12]^d[13]^d[15]^d[17]^d[20]^d[22]^d[27]^d[30]
    c[28] = d[0]^d[1]^d[2]^d[3]^d[5]^d[9]^d[10]^d[14]^d[15]^d[16]^d[17]^d[19]^d[22]^d[27]^d[28]
    c[29] = d[0]^d[5]^d[6]^d[9]^d[11]^d[12]^d[14]^d[16]^d[18]^d[19]^d[22]^d[23]^d[24]^d[27]^d[29]
    c[30] = d[0]^d[1]^d[6]^d[7]^d[8]^d[10]^d[12]^d[13]^d[15]^d[17]^d[19]^d[20]^d[23]^d[25]^d[30]
    c[31] = d[0]^d[1]^d[2]^d[4]^d[7]^d[8]^d[9]^d[11]^d[13]^d[14]^d[16]^d[18]^d[21]^d[26]^d[31]
    return c


def reverse3(output, raw=False):
    """" all elements of rfw3.values are exactly len==2 """
    options = [rfw3[o] for o in output]
    if raw:
        return options
    for r3 in itertools.product(*options):
        yield list(itertools.chain(*r3))


@njit()
def forward_full_c(c, steps=ROUNDS):
    """ full forward repeat(sbox, dbox) steps times; sbox2. For debugging, steps = 1 """
    c = list(c)  # if it isn't a list, make it one
    d = [0] * 32
    for i in range(steps):
        for j in range(32):
            d[j] = CONFUSION[c[j]]
            c[j] = 0
        for j in range(32):
            for k in range(32):
                c[j] ^= d[k] * ((DIFFUSION[j] >> k) & 1)
    return [CONFUSION[c[i * 2]] ^ CONFUSION[c[i * 2 + 1] + 256] for i in range(16)]


if __name__ == '__main__':
    inp_ = List(EX_INP)
    
    assert list(reverse1(inp_, rfw1np)) == list(split.reverse1(EX_INP))
    assert list(reverse2(EX_INP)) == list(split.reverse2(EX_INP))
    assert list(forward_full_c(EX_INP)) == list(forward_full_a(EX_INP))
    # assert list(reverse3_(inp)) == list(reverse3(inp))
