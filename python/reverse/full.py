from time import time

from constants import *
from reverse.split import *


def reverse21(options):
    options_next = []
    for r3o in options:
        r2 = reverse2(r3o)  # unique result
        options_next += reverse1(r2)  # yields list of options
    return options_next


def reverse(output):
    for start_option, r3 in enumerate(reverse3(output)):
        options = {bytes(r3)}
        for round in range(ROUNDS):
            options = reverse21(options)
            if not options:
                break
            # else:
            #     print(start_option, round, len(options))
        if options:
            yield options
        if start_option > (1 << 14):
            return


if __name__ == '__main__':
    start = time()
    for rev in reverse(TARGET):
        print(rev)
    """for o in reverse(target):
        fwo = forward_full_a(o)
        fwo_str = "".join([chr(i) for i in fwo])
        print(o, "leads to", fwo_str)
    """
    print("Execution time = ", time() - start)
