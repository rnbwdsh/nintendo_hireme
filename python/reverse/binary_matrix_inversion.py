import numpy as np
from BitVector import BitVector

# if you have sagemath, you can just use
# bw2np = matrix(GF(2), fw2np).inverse()


def rref_binary(m):
    """helper for gf2 matrix inversion: Converts a matrix to reduced row echelon form (RREF)"""
    n_rows, n_cols = m.shape
    current_row = 0
    for j in range(n_cols):
        if current_row >= n_rows:
            break
        pivot_row = current_row
        while pivot_row < n_rows and m[pivot_row, j] == 0:
            pivot_row += 1
        if pivot_row == n_rows:
            continue
        m[[current_row, pivot_row]] = m[[pivot_row, current_row]]
        pivot_row = current_row
        current_row += 1
        for i in range(current_row, n_rows):
            if m[i, j] == 1:
                m[i] = (m[i] + m[pivot_row]) % 2
    for i in reversed(range(current_row)):
        pivot_col = 0
        while pivot_col < n_cols and m[i, pivot_col]==0:
            pivot_col += 1
        if pivot_col == n_cols:
            continue
        for j in range(i):
            if m[j, pivot_col] == 1:
                m[j] = (m[j] + m[i]) % 2
    return m


def inverse_binary(m):
    n_rows, n_cols = m.shape
    if n_rows != n_cols:
        raise Exception("Matrix has to be square")
    augmented_matrix = np.hstack([m, np.eye(n_rows)]) # Augmented matrix
    rref_form = rref_binary(augmented_matrix)
    return rref_form[:, n_rows:]


def b2b(b):
    """ Helper: bits-to-bytes and bytes-to-bits, b2b(b2b(x)) == x"""
    if type(b) == list or type(b) == tuple:
        b = bytes(b)
    if type(b) == bytes:
        return np.array(BitVector(rawbytes=b), dtype=np.uint8)
    elif type(b) == np.ndarray:
        return tuple(bytes.fromhex(BitVector(bitlist=b).get_bitvector_in_hex()))
    else: # unreachable
        raise TypeError(b)
