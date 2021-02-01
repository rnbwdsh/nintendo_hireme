from multiprocessing import Pool
from time import time

from numba import jit
from tqdm import tqdm

from reverse_fast.split import *
from constants import *


@jit(nopython=True)
def reverse21_(options, rfw1np):
    return [reverse1(reverse2(r3), rfw1np) for r3 in options]


def reverse21(options):
    rev21 = reverse21_(options, rfw1np)
    rev21 = list(filter(len, rev21))
    return np.vstack(rev21) if rev21 else np.zeros((0, 32))


def reverse(output, threads=32):
    generator = reverse3(output)
    pool = Pool(threads)
    while True:
        options = np.array([next(generator) for _ in range(BATCH_SIZE)], dtype=np.uint8)
        tq = tqdm(range(ROUNDS), leave=False)
        for _ in tq:
            options = np.vstack(pool.map(reverse21, np.array_split(options, threads)))
            if not len(options):
                break
            tq.set_postfix_str(f"size {len(options)}")
        yield options


if __name__ == '__main__':
    # for solutions in \
    start = time()
    for _ in range(BATCH_RUNS):
        print(len(next(reverse(TARGET))))
    print("Execution time = ", time() - start)
    """:
        print("found solutions:", len(solutions))
        with open("solutions.txt", "a") as solf:
            solf.write("\n".join([",".join([hex(s) for s in solution]) + "\n" for solution in solutions]))
            # verification code
        fwo = forward_full_a(solutions[0])
        fwo_str = "".join([chr(i) for i in fwo])
        print(solution, "leads to", fwo_str)
    """