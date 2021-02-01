from constants import *

# original challenge, pythonified
def forward_full_a(c, steps=ROUNDS):
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
    # b = 0x739b780d213d3cc03268674c5e9b25ffc623adcd13ba240d99af6961e1a0958b.to_bytes(32, "big")
    for a in open("/home/m/Desktop/a.txt"):
        b = int(a, 16).to_bytes(32, "big")
        res = forward_full_a(b)
        print("".join([chr(r) for r in res]))
    #print(*forward_full_a(inp))
    #import os
    #os.system("gcc original.cpp -o original; ./original")