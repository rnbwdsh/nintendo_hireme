import itertools
from collections import defaultdict

import numpy as np

from forward.split import forward1, forward2, forward3, CONFUSION, EX_INP, DIFFUSION
from .binary_matrix_inversion import inverse_binary, b2b

# reverse step 1
rfw1 = defaultdict(list)
for i in range(256):
    rfw1[CONFUSION[i]].append(i)
# print("rfw1 double values", {k: v for k, v in rfw1.items() if len(v) > 1})
# print("rfw1 missing keys", set(range(256)).difference(rfw1))
rfw1 = [rfw1[i] for i in range(256)]


def reverse1(output):
    """ elements can be len0, len1 or len2 """
    options = [rfw1[o] for o in output]
    return list(itertools.product(*options))


# reverse step 3
rfw3 = defaultdict(list)
for i in range(256):
    for j in range(256):
        rev = CONFUSION[i] ^ CONFUSION[j + 256]
        rfw3[rev].append([i, j])
rfw3 = tuple([tuple(rfw3[i]) for i in range(256)])


def reverse3(output, raw=False):
    """" all elements of rfw3.values are exactly len==2 """
    options = [rfw3[o] for o in output]
    if raw:
        return options
    for r3 in itertools.product(*options):
        yield list(itertools.chain(*r3))



# build diffusion matrix for step 2 and inverse to have a reversible fw2 with sagemath
fw2np = np.zeros((256, 256), dtype=np.bool)
for i in range(256):
    one_hot = (int(1)<<(255-i)).to_bytes(32, "big")  # 1 bit at i on, 31 off
    fw2np[i] = b2b(bytes(forward2(one_hot)))

bw2np = inverse_binary(fw2np)


# def reverse2(output):  # @ dot product
#     return b2b((b2b(output) @ bw2np) % 2)
def reverse2(d):
    c = np.zeros(32, dtype=np.uint8)
    for j in range(32):
        for k in range(32):
            diff = ((DIFFUSION[j] >> k) & 1)
            c[j] ^= d[k] * diff
    return c


if __name__ == '__main__':
    assert EX_INP in reverse1(forward1(EX_INP))
    assert (EX_INP == reverse2(forward2(EX_INP))).all()
    assert all([i in fwrw for i, fwrw in zip(EX_INP, reverse3(forward3(EX_INP), raw=True))])
    assert (fw2np == bw2np).all()
    #np.savetxt("forward.csv", fw2np, fmt="%d")
    #np.savetxt("reverse.csv", bw2np, fmt="%d")
