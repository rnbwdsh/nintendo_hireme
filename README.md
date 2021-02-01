# Report - Hireme Nintendo Logic Reversing Challenge

* [Introduction](#introduction)
* [Program disection - python](#program-disection---python)
* [Basic Blocks - Cryptography and Logic](#basic-blocks---cryptography-and-logic)
  + [Forward 1: S-Box 1](#forward-1--s-box-1)
  + [Forward 2 - D-Box](#forward-2---d-box)
  + [Forward 2 - Galois field 2](#forward-2---galois-field-2)
  + [Forward 3 - S-Box 2](#forward-3---s-box-2)
  + [Putting it all together](#putting-it-all-together)
* [Optimization](#optimization)
  + [Numba](#numba)
  + [C++/CUDA](#c---cuda)
  + [CuPy](#cupy)
* [Methodology](#methodology)
* [Results](#results)
* [Discussion](#discussion)
* [Conclusion](#conclusion)

## Introduction
In 2019 [NERD (Nintendo European Research and Development)](https://www.nerd.nintendo.com/jobs) released the [HireMe challenge](https://www.nerd.nintendo.com/files/HireMe) which, according to tests various skills in the fields of   
* Logic Reversing
* Discrete Mathematics and Cryptography
* C/C++ Programming

The challenge is split into 3 separate levels: Finding an iterative solution in seconds, milliseconds and finding all 2^128 solutions.

As the first levels are somewhat explored on the internet [[1]](https://github.com/IamLupo/Nintendo-HireMe) [[2]](https://github.com/alberthier/nerd.nintendo.com) [[3]](https://github.com/donno2048/Nintendo), but mostly solved with python-implementations that take multiple seconds for one solution, I tried to write an efficient solver that can find gigabytes of solutions in a feasible time.

This is the original challenge:
```cpp
#include <string.h>

typedef unsigned char u8;
typedef unsigned int u32;

u8 confusion[512]={
0xac,0xd1,0x25,0x94,0x1f,0xb3,0x33,0x28,0x7c,0x2b,0x17,0xbc,0xf6,0xb0,0x55,0x5d,
0x8f,0xd2,0x48,0xd4,0xd3,0x78,0x62,0x1a,0x02,0xf2,0x01,0xc9,0xaa,0xf0,0x83,0x71,
0x72,0x4b,0x6a,0xe8,0xe9,0x42,0xc0,0x53,0x63,0x66,0x13,0x4a,0xc1,0x85,0xcf,0x0c,
0x24,0x76,0xa5,0x6e,0xd7,0xa1,0xec,0xc6,0x04,0xc2,0xa2,0x5c,0x81,0x92,0x6c,0xda,
0xc6,0x86,0xba,0x4d,0x39,0xa0,0x0e,0x8c,0x8a,0xd0,0xfe,0x59,0x96,0x49,0xe6,0xea,
0x69,0x30,0x52,0x1c,0xe0,0xb2,0x05,0x9b,0x10,0x03,0xa8,0x64,0x51,0x97,0x02,0x09,
0x8e,0xad,0xf7,0x36,0x47,0xab,0xce,0x7f,0x56,0xca,0x00,0xe3,0xed,0xf1,0x38,0xd8,
0x26,0x1c,0xdc,0x35,0x91,0x43,0x2c,0x74,0xb4,0x61,0x9d,0x5e,0xe9,0x4c,0xbf,0x77,
0x16,0x1e,0x21,0x1d,0x2d,0xa9,0x95,0xb8,0xc3,0x8d,0xf8,0xdb,0x34,0xe1,0x84,0xd6,
0x0b,0x23,0x4e,0xff,0x3c,0x54,0xa7,0x78,0xa4,0x89,0x33,0x6d,0xfb,0x79,0x27,0xc4,
0xf9,0x40,0x41,0xdf,0xc5,0x82,0x93,0xdd,0xa6,0xef,0xcd,0x8d,0xa3,0xae,0x7a,0xb6,
0x2f,0xfd,0xbd,0xe5,0x98,0x66,0xf3,0x4f,0x57,0x88,0x90,0x9c,0x0a,0x50,0xe7,0x15,
0x7b,0x58,0xbc,0x07,0x68,0x3a,0x5f,0xee,0x32,0x9f,0xeb,0xcc,0x18,0x8b,0xe2,0x57,
0xb7,0x49,0x37,0xde,0xf5,0x99,0x67,0x5b,0x3b,0xbb,0x3d,0xb5,0x2d,0x19,0x2e,0x0d,
0x93,0xfc,0x7e,0x06,0x08,0xbe,0x3f,0xd9,0x2a,0x70,0x9a,0xc8,0x7d,0xd8,0x46,0x65,
0x22,0xf4,0xb9,0xa2,0x6f,0x12,0x1b,0x14,0x45,0xc7,0x87,0x31,0x60,0x29,0xf7,0x73,
0x2c,0x97,0x72,0xcd,0x89,0xa6,0x88,0x4c,0xe8,0x83,0xeb,0x59,0xca,0x50,0x3f,0x27,
0x4e,0xae,0x43,0xd5,0x6e,0xd0,0x99,0x7b,0x7c,0x40,0x0c,0x52,0x86,0xc1,0x46,0x12,
0x5a,0x28,0xa8,0xbb,0xcb,0xf0,0x11,0x95,0x26,0x0d,0x34,0x66,0x22,0x18,0x6f,0x51,
0x9b,0x3b,0xda,0xec,0x5e,0x00,0x2a,0xf5,0x8f,0x61,0xba,0x96,0xb3,0xd1,0x30,0xdc,
0x33,0x75,0xe9,0x6d,0xc8,0xa1,0x3a,0x3e,0x5f,0x9d,0xfd,0xa9,0x31,0x9f,0xaa,0x85,
0x2f,0x92,0xaf,0x67,0x78,0xa5,0xab,0x03,0x21,0x4f,0xb9,0xad,0xfe,0xf3,0x42,0xfc,
0x17,0xd7,0xee,0xa3,0xd8,0x80,0x14,0x2e,0xa0,0x47,0x55,0xc4,0xff,0xe5,0x13,0x3f,
0x81,0xb6,0x7a,0x94,0xd0,0xb5,0x54,0xbf,0x91,0xa7,0x37,0xf1,0x6b,0xc9,0x1b,0xb1,
0x3c,0xb6,0xd9,0x32,0x24,0x8d,0xf2,0x82,0xb4,0xf9,0xdb,0x7d,0x44,0xfb,0x1e,0xd4,
0xea,0x5d,0x35,0x69,0x23,0x71,0x57,0x01,0x06,0xe4,0x55,0x9a,0xa4,0x58,0x56,0xc7,
0x4a,0x8c,0x8a,0xd6,0x6a,0x49,0x70,0xc5,0x8e,0x0a,0x62,0xdc,0x29,0x4b,0x42,0x41,
0xcb,0x2b,0xb7,0xce,0x08,0xa1,0x76,0x1d,0x1a,0xb8,0xe3,0xcc,0x7e,0x48,0x20,0xe6,
0xf8,0x45,0x93,0xde,0xc3,0x63,0x0f,0xb0,0xac,0x5c,0xba,0xdf,0x07,0x77,0xe7,0x4e,
0x1f,0x28,0x10,0x6c,0x59,0xd3,0xdd,0x2d,0x65,0x39,0xb2,0x74,0x84,0x3d,0xf4,0xbd,
0xc7,0x79,0x60,0x0b,0x4d,0x33,0x36,0x25,0xbc,0xe0,0x09,0xcf,0x5b,0xe2,0x38,0x9e,
0xc0,0xef,0xd2,0x16,0x05,0xbe,0x53,0xf7,0xc2,0xc6,0xa2,0x24,0x98,0x1c,0xad,0x04};

u32 diffusion[32]={
0xf26cb481,0x16a5dc92,0x3c5ba924,0x79b65248,0x2fc64b18,0x615acd29,0xc3b59a42,0x976b2584,
0x6cf281b4,0xa51692dc,0x5b3c24a9,0xb6794852,0xc62f184b,0x5a6129cd,0xb5c3429a,0x6b978425,
0xb481f26c,0xdc9216a5,0xa9243c5b,0x524879b6,0x4b182fc6,0xcd29615a,0x9a42c3b5,0x2584976b,
0x81b46cf2,0x92dca516,0x24a95b3c,0x4852b679,0x184bc62f,0x29cd5a61,0x429ab5c3,0x84256b97};

u8 input[32]={
//change only this :
0x66,0xd5,0x4e,0x28,0x5f,0xff,0x6b,0x53,0xac,0x3b,0x34,0x14,0xb5,0x3c,0xb2,0xc6,
0xa4,0x85,0x1e,0x0d,0x86,0xc7,0x4f,0xba,0x75,0x5e,0xcb,0xc3,0x6e,0x48,0x79,0x8f};

void Forward(u8 c[32],u8 d[32],u8 s[512],u32 p[32]){
    for(u32 i=0;i<256;i++) {
        for(u8 j=0;j<32;j++) {
            d[j]=s[c[j]];
            c[j]=0;
        }

        for(u8 j=0;j<32;j++)
            for(u8 k=0;k<32;k++)
                c[j]^=d[k]*((p[j]>>k)&1);
    }
    for(u8 i=0;i<16;i++)
        d[i]=s[c[i*2]]^s[c[i*2+1]+256];
}

int main(int argc, char* argv[])
{
    u8 target[]="Hire me!!!!!!!!";
    u8 output[32];

    Forward(input,output,confusion,diffusion);

    return memcmp(output,target,16); // => contact jobs(at)nerd.nintendo.com
}
```
## Program disection - python

To make the challenge more manageable, we will first rewrite the c-program into python in the file [`forward/original.py`](python/forward/original.py) and the program also self-tests with passing the same input through the C-program.

Then in the program [`forward/split.py`](python/forward/split.py) the original program is split up into several primitive chunks: forward1, forward2 and forward3 and the very basic program structure can be broken down to this:
```python
def forward_full_b(c):
    for _ in range(256):
        d = forward1(c)
        c = forward2(d)
    return forward3(c)
```

Where forward1 describes the first s-box, forward2 describes the d-box and forward3 describes the 2nd d-box.

Furthermore the libaries BitVector, numpy (typically imported as np) and numba - common tools for efficient computation in python - will be used, as well as a handy little utility for converting bits to bytes and vice versa you can find in [`binary_matrix_inversion.py`](python/reverse/binary_matrix_inversion.py):

```python
def b2b(b):
    """ Helper: bits-to-bytes and bytes-to-bits, b2b(b2b(x)) == x"""
    if type(b) == list or type(b) == tuple:
        b = bytes(b)
    if type(b) == bytes:
        return np.array(BitVector(rawbytes=b), dtype=np.uint8)
    elif type(b) == np.ndarray:
        return tuple(bytes.fromhex(BitVector(bitlist=b).get_bitvector_in_hex()))
```

## Basic Blocks - Cryptography and Logic
The challenge contains two basic building blocks from traditional symmetric cryptography: [S-Boxes](https://en.wikipedia.org/wiki/S-box) for substituion and the less comonly known [D-Boxes](https://en.wikipedia.org/wiki/Confusion_and_diffusion) for diffusion. Both provide the Shannon-property of confusion/diffusion - this is why the arrays in the C-Program are called `confusion` and `diffusion`.

Mathematically speaking, both are just a deterministic, stateless linear mapping from one bit-space to another. 

### Forward 1: S-Box 1
The forward-code of the first s-box can be simplified in python to:
```python
for j in range(32):
    d[j] = confusion[c[j]]
```


The S-Box in the challenge is working on byte-level, so it maps 8 bits to other 8 bits. The D-Box is working on the whole input, so it maps a 256 bit space to another 256 bit space. As the input and output spaces are equally sized, one could expect simple, reversible linear mappings - but analyzing the D-Box shows that some subsets of inputs map to the same state and some output-states are not mapped to at all.

This can be found with the simple python program.
```python
rfw1 = defaultdict(list)
for i in range(256):
    rfw1[confusion[i]].append(i)
print("rfw1 double values", {k: v for k, v in rfw1.items() if len(v) > 1})
print("rfw1 missing keys", set(range(256)).difference(rfw1))
```
Printing:
```
rfw1 double values {51: [6, 154], 188: [11, 194], 120: [21, 151], 2: [24, 94], 233: [36, 124], 102: [41, 181], 198: [55, 64], 162: [58, 243], 73: [77, 209], 28: [83, 113], 247: [98, 254], 216: [111, 237], 45: [132, 220], 141: [137, 171], 147: [166, 224], 87: [184, 207]}
rfw1 missing keys {15, 17, 32, 62, 68, 90, 107, 117, 128, 158, 175, 177, 203, 213, 228, 250}
```

### Forward 2 - D-Box
The D-Box can be simplified in python to
```python
for j in range(32):
    for k in range(32):
        c[j] ^= d[k] * ((diffusion[j] >> k) & 1)
```

This can be seen as a kind of matrix multiplication of a diffusion-bit-matrix of size 256x256 and an input of 256 bits. The matrix can be found by passing single bits through the diffusion box like this:

```python
fw2np = np.zeros((256, 256), dtype=np.bool)
for i in range(256):
    one_hot = (int(1)<<(255-i)).to_bytes(32, "big")  # 1 bit at i on, 31 off
    fw2np[i] = b2b(bytes(forward2(one_hot)))
assert b2b(b2b(inp)@fw2np%2) == forward2(inp)  # yields true
```

The code at the end proves, that passing the bits of input `inp` through a dot product (@ in python/numpy) leads to the same result as applying the `forward2` function and taking the mod2 of it.

### Forward 2 - Galois field 2 
A typical integer matrix multiplication would require simple matrix inversion which could be done with algorithms like the gauss inversion.

But in our case, the input and output-fields are single bits, so every bit in the input is XOR'ed with a certain set of bits in the input field. As binary numbers are quite common, there is a special algebraic structure called GF2 ([galois field 2](https://en.wikipedia.org/wiki/GF(2))) which is a special kind of fields that only has two numbers and defines the multiplication as a logical and and the addition as a logical or. It can be seen as a 1-bit-integer-number that can overflow, so the only difference from regular integer numbers is that the addition of 1+1 is 0, not 2 (as this would be outside our field).

The python-based computer algebra system [SageMath](https://www.sagemath.org/) already knows how to work with matrices of GF2 and inverting them is as simple as calling `matrix(GF(2), input_matrix).inverse()`. But as SageMath is not widely available and a pain to install, a numpy-version from [[here]](https://npdeep.github.io/matrix-inversion-gf2.html) using reduced row-echelon form is included.

This provides us with a general solution to all similar D-Box problems, but I later found out that the inverse matrix is the same as the input-matrix, meaning that the diffusion matrix is an [`involutory matrix`](https://en.wikipedia.org/wiki/Involutory_matrix). The proof for this can be found in [`reverse/split.py`](python/reverse/split.py) where we show in the line `assert (fw2np == bw2np).all()
` that the forward and the backward-matrix are equal. This can be verified further, by calling the forward2-function twice on the same input - and it returns the same output. So for this special case of this diffusion matrix, all the work was for nothing, as we can simply use the forward2-function to reverse itself.

### Forward 3 - S-Box 2
The 3rd d-box can be simplified to this python-code
```python
def forward3(c):
    return [confusion[c[i*2]] ^ confusion[c[i*2+1] + 256] for i in range(16)]
```

Unlike the others, this function is only called once at the end, and not 256 times in a loop. Also unlike the others, the input-space here is a 32-byte matrix, but the output is only a 16-byte matrix that has to lead to the bytes `b'Hire me!!!!!!!!\0'`. On further inspection, we see that it takes the input in chunks of size 2 (16 bits), passes both 8-bit-halfs through the confusion matrix from the first s-box, but now also utilizing the indices above 256, not used by the original S-Box.

As the whole mapping is just 2^16 -> 2^8, repeated 16 times, we can just exhaustively compute the inverse 2^8 -> 2^16 mapping like in the first s-box:
```python
rfw3 = defaultdict(list)
for i in range(256):
    for j in range(256):
        rev = confusion[i] ^ confusion[j + 256]
        rfw3[rev].append([i, j])
rfw3 = tuple([tuple(rfw3[i]) for i in range(256)])
```

This means every 16-bit-output we want to reverse deterministically leads to a 32-bit-input of the forward3-function, which means we can easily reverse this function.

### Putting it all together

Using the results previously discussed, that can be found in [`reverse/split.py`](python/reverse/split.py) we can already build an initial solver that is able to find a solution in less than a minute on a modern system with the code you can also find in [`reverse/full.py`](python/reverse/full.py).

```python
def reverse21(options):
    options_next = set()
    for r3o in options:
        r2 = reverse2(r3o)  # unique result
        for r1 in reverse1(r2):  # yields list of options
            options_next.add(r1)
    return options_next


def reverse(output):
    for start_option, r3 in enumerate(reverse3(output)):
        options = {bytes(r3)}
        for round in range(ROUNDS):
            options = reverse21(options)
            if not options:
                break
            else:
                print(start_option, round, len(options))
        for option in options:
            yield option
```

## Optimization

Just like the other solvers in python, the previously discussed solution is rather slow and takes around a minute to find a single solution. The major limitations are:
* the python [global interpreter lock (GIL)](https://realpython.com/python-gil/)
* single-threading
* the general inefficiency of python as an interpreted, dynamically typed language.
* the program-structure that is not optimized for pipelining

So I decided to build 3 extended variants with different computational approaches:
1. A numba-optimized variant that eliminates the GIL and uses multi-threading
2. Two C++/CUDA variants. One using recursion and storing all values on the stack (which doesn't work for CUDA) and a heap-based variant that uses a thread-local job-queue. A job (state) is specified here as the round (timestep out of the 256 top-level iterations), the position inside the reverse1-forking and the actual 32 bytes at this time.
3. A numpy-variant that can also use [CuPy](https://cupy.dev/) a variant of numpy that supports CUDA-computations and is implemented in a way, that requires only a constant number of computations and uses Tensors (higher dimensional arrays) as a basis. I've also tried tensorflow/pytorch but some basic methods were missing or yielded bad performance.

The optimizations used are built on each other, and the code can be found in the respective python or cpp folders: [`python/reverse_fast`](python/reverse_fast), [`cpp/heap`](cpp/heap), [`cpp/stack`](cpp/stack) and [`python/reverse_tensor`](python/reverse_tensor). 

### Numba

Numba is a just-in-time-compiler that can compile python functions into LLVM code. LLVM is an intermediate language that binds higher level languages like C, C++, Rust or Java to any underlying instruction set architecture (typically x86) under it, meaning that these languages can theoretically archive very similar performance. The default implementations of python (CPython) is written in C/C++, which is itself optimized by LLVM. Other interpreters also use JIT-compilation like Jython (Java), IronPython (C#) and PyPy (Python) but none of them reach the speed of CPython or native code.

Additionally, numba allows automatic parallelization of code by adding the parallel-option, as well as a mode that lifts the global interpreter lock.

To maximize the performance, the python code has to be adapted to only use homogenous objects of fixed size (done with numpy objects, which are internally treated like C-arrays) and it can't use functions built into python like itertools.combinations, but as numpy is written in C, numpy-functions can be used.

The according implementations for the `reverse1`, `reverse2` and `reverse3` can be found in [`reverse_fast/split.py`](python/reverse_fast/split.py) and these implementations are tied together by [`reverse_fast/full.py`](python/reverse_fast/full.py).

The major adaptions have been
0) Adding the @njit function decorator annotation to compile functions to C
1) a complete re-implementation of an iterative method of itertools.combinations, a function that takes a list of lists and returns all combinations of all elements in the list.

```python
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
```

2) The reverse1 function can be changed from

```python
def reverse1(output):
    """ elements can be len0, len1 or len2 """
    options = [rfw1[o] for o in output]
    return list(itertools.product(*options))
```

to

```python
@njit()
def reverse1(output, rfw1np):
    return product_jit([rfw1np[a] for a in output])
```

So with minimal changes, we can reuse our python code but this function is now faster by a huge factor.

3) As we have found out, the forward2 and reverse2 function are the same, so we could simply use the bit-matrix-multiplication code we found before, but in fact, the original C-implementation or our python-version of it are far more optimal than a 256x256 matrix multiplication. The difference is roughly 1:10 from experimentation.

```python
@njit()
def reverse2(d):
    c = np.zeros(32, dtype=np.uint8)
    for j in range(32):
        for k in range(32):
            diff = ((diffusion[j] >> k) & 1)
            c[j] ^= d[k] * diff
    return c
```

4) To get even more performance, we can use the python multiprocessing feature on the outermost layer of the program and batch processing to compute multiple values at the same time, that originate from the combinations of reverse3 values. For a progress-display, we use the tqdm libary, and numpy.vstack to merge results, which vertically stacks multiple N x 16 arrays to a longer M x 32 array.

5) The code putting together the computations is [`reverse_full_fast.py`](python/reverse_fast/full.py):

```python
def reverse(output, threads=32):
  generator = reverse3(output)
  pool = Pool(threads)
  while True:
    options = np.array([next(generator) for _ in range(BATCH_SIZE//BATCH_RUNS)], dtype=np.uint8)
    tq = tqdm(range(ROUNDS), leave=False)
    for round in tq:
      options = np.vstack(pool.map(reverse21, np.array_split(options, threads)))
      if not len(options):
        break
      tq.set_postfix_str(f"size {len(options)}")
    yield options
```

To get optimal performance, I chose to do everything in a 16 batches, as the call to np.unique gets more complex with bigger batch sizes, as it's complexity is roughly N*log(N). To fully utilize the 16 hyperthreads, I use 32 threads as some threads with smaller workloads are often done faster.

### C++/CUDA

To gain even better performance than numba, I then tried to re-implement my python code in C/C++. Theoretically CUDA, the parallel computing framework from NVIDIA, should allow even more parallelism as it theoretically has 3072 shader processors and faster DDR6 memory. 

To launch a function like `reverse` in parallel on CUDA on multiple blocks, I can simply use

```cpp
Reverse<<<NR_BLOCKS, NR_THREADS>>>();
cudaDeviceSynchronize();
```

And inside these functions, the threadIdx.[x,y,z] and blockIDX[x,y,z] variables are set to different values. 

Whereas multithreading in regular C++ looks like this:

```cpp
  std::thread threads[NR_THREADS];
  for(int i=0;i<NR_THREADS;i++)
    threads[i] = std::thread(Reverse, i);

  for(int i=0;i<NR_THREADS;i++)
    threads[i].join();
```


In addition to a plain re-implementation I added further optimizations:

0. As CUDA doesn't support simple STL data structures like vector (dynamic list), or pair, I had to create trivial, memory-efficient variants myself:

```cpp
struct state {
  u16 round=0;
  u8* data;
  u8 pos;
};

class vector {
 public:
  int idx = 0;
  state states[BUFFER_SIZE];

  GLOBAL void emplace_back(u16 round, u8* data, u8 pos) {
    if(idx > BUFFER_SIZE)
        printf("out of memory\n");
    states[idx].round = round;
    states[idx].data = data;
    //memcpy(&states[idx].data, data, ILEN);
    states[idx].pos = pos;
    idx++;
  }

  GLOBAL state* back() {
    return &states[idx++];
  }

  GLOBAL state pop() {
    return states[--idx];
  }

  GLOBAL bool empty() {
    return idx == 0;
  }
};
```

To print the state as a hex-value, I generated this function, as a loop would't work in a multi-threaded scenario.
```
GLOBAL void printState(const u8* r) {
  printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n", r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
}
```

Another constraint to CUDA is the limited stack size, which is why the following, described stack-solution that works with recursion doesn't work on CUDA. Also, diagnosing the errors was rather difficult, as CUDA doesn't throw normal errors/exceptions, and even [CUDA error checking](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api) didn't seem to work.
   
1. Just like with numba, one of the problems is that there is no combinations-function that can do the variable-output-size reverse1 - and for optimization reasons, a different implementation for reverse3 was used, as the array has a nice 16x256x2 uniform shape (can be gained by calling the python reverse3(target, raw=true) - so the generation can be done by using the first 8 bits of an index in the first dimension, the next 8 bits in the 2nd, and so on. Theoretically, to cover the whole input-space, a 128 bit (8 bits per byte x 16 bytes) integer would be required or two u64's, but even the u64 space is too huge to compute. The code for reverse3 then looks like this.

```c
CONST u8 rev3[OLEN][256][2] =
{{{0, 153}, {1, 22},  {2, 67},  {3, 63},  {3, 171}, {4, 150}, {5, 141},....

GLOBAL void Reverse3(u64 a, u8* out) {
  for(int i=0;i<OLEN; i++) {
    auto tr = rev3[i][a & 0xFF];
      out[i*2] = tr[0];
      out[i*2+1] = tr[1];
  a = a >> 8;
  }
}
```

2. Instead of the `reverse2` double-nested loop, I first tried to do it as a matrix multiplication with XOR-reduction on the latter dimension, but this wasn't any faster. Then I had the idea of, reformulating the matrix multiplication with a constant matrix into linear code and it resulted in a 3x speedup for this section.

So the `forward2` code (that is also the `reverse2` code as discussed previously) can be rewritten from

```c
      for(u8 j = 0; j < ILEN; j++)
        for(u8 k = 0; k < ILEN; k++)
          c[j] ^= d[k] * ((diffusion[j] >> k) & 1);
```
To
```c
      for(u8 j = 0; j < ILEN; j++)
        for(u8 k = 0; k < ILEN; k++)
          if((diffusion[j] >> k) & 1)
            c[j] ^= d[k];
```
By then slightly modifying this code, we can generate our new, optimized code
```c
for(u8 j = 0; j < 32; j++) {
    printf("c[%d] = ", j);
    for (u8 k = 0; k < 32; k++) {
      u8 dj = (diffusion[j] >> k) & 1;
      if (dj) {
        printf("d[%d]^", k);
        c[j] ^= d[k];
      }
    }
    printf(";\n");
  }
  cout << endl;
```

Printing the i/j's of the innermost calculation allows us to omit all unnecessary ifs, making the code 100% linear. The resulting code looks can be transformed into
```c
GLOBAL void Forward2(u8 *d) {
  u8 c[ILEN];
  c[0] = d[0]^d[7]^d[10]^d[12]^d[13]^d[15]^d[18]^d[19]^d[21]^d[22]^d[25]^d[28]^d[29]^d[30]^d[31];
  c[1] = d[1]^d[4]^d[7]^d[10]^d[11]^d[12]^d[14]^d[15]^d[16]^d[18]^d[21]^d[23]^d[25]^d[26]^d[28];
  c[2] = d[2]^d[5]^d[8]^d[11]^d[13]^d[15]^d[16]^d[17]^d[19]^d[20]^d[22]^d[26]^d[27]^d[28]^d[29];
  c[3] = d[3]^d[6]^d[9]^d[12]^d[14]^d[17]^d[18]^d[20]^d[21]^d[23]^d[24]^d[27]^d[28]^d[29]^d[30];
  c[4] = d[3]^d[4]^d[8]^d[9]^d[11]^d[14]^d[17]^d[18]^d[22]^d[23]^d[24]^d[25]^d[26]^d[27]^d[29];
  c[5] = d[0]^d[3]^d[5]^d[8]^d[10]^d[11]^d[14]^d[15]^d[17]^d[19]^d[20]^d[22]^d[24]^d[29]^d[30];
  c[6] = d[1]^d[6]^d[9]^d[11]^d[12]^d[15]^d[16]^d[18]^d[20]^d[21]^d[23]^d[24]^d[25]^d[30]^d[31];
  c[7] = d[2]^d[7]^d[8]^d[10]^d[13]^d[16]^d[17]^d[19]^d[21]^d[22]^d[24]^d[25]^d[26]^d[28]^d[31];
  c[8] = d[2]^d[4]^d[5]^d[7]^d[8]^d[15]^d[17]^d[20]^d[21]^d[22]^d[23]^d[26]^d[27]^d[29]^d[30];
  c[9] = d[2]^d[3]^d[4]^d[6]^d[7]^d[9]^d[12]^d[15]^d[17]^d[18]^d[20]^d[24]^d[26]^d[29]^d[31];
  c[10] = d[0]^d[3]^d[5]^d[7]^d[10]^d[13]^d[18]^d[19]^d[20]^d[21]^d[24]^d[25]^d[27]^d[28]^d[30];
  c[11] = d[1]^d[4]^d[6]^d[11]^d[14]^d[16]^d[19]^d[20]^d[21]^d[22]^d[25]^d[26]^d[28]^d[29]^d[31];
  c[12] = d[0]^d[1]^d[3]^d[6]^d[11]^d[12]^d[16]^d[17]^d[18]^d[19]^d[21]^d[25]^d[26]^d[30]^d[31];
  c[13] = d[0]^d[2]^d[3]^d[6]^d[7]^d[8]^d[11]^d[13]^d[16]^d[21]^d[22]^d[25]^d[27]^d[28]^d[30];
  c[14] = d[1]^d[3]^d[4]^d[7]^d[9]^d[14]^d[16]^d[17]^d[22]^d[23]^d[24]^d[26]^d[28]^d[29]^d[31];
  c[15] = d[0]^d[2]^d[5]^d[10]^d[15]^d[16]^d[17]^d[18]^d[20]^d[23]^d[24]^d[25]^d[27]^d[29]^d[30];
  c[16] = d[2]^d[3]^d[5]^d[6]^d[9]^d[12]^d[13]^d[14]^d[15]^d[16]^d[23]^d[26]^d[28]^d[29]^d[31];
  c[17] = d[0]^d[2]^d[5]^d[7]^d[9]^d[10]^d[12]^d[17]^d[20]^d[23]^d[26]^d[27]^d[28]^d[30]^d[31];
  c[18] = d[0]^d[1]^d[3]^d[4]^d[6]^d[10]^d[11]^d[12]^d[13]^d[18]^d[21]^d[24]^d[27]^d[29]^d[31];
  c[19] = d[1]^d[2]^d[4]^d[5]^d[7]^d[8]^d[11]^d[12]^d[13]^d[14]^d[19]^d[22]^d[25]^d[28]^d[30];
  c[20] = d[1]^d[2]^d[6]^d[7]^d[8]^d[9]^d[10]^d[11]^d[13]^d[19]^d[20]^d[24]^d[25]^d[27]^d[30];
  c[21] = d[1]^d[3]^d[4]^d[6]^d[8]^d[13]^d[14]^d[16]^d[19]^d[21]^d[24]^d[26]^d[27]^d[30]^d[31];
  c[22] = d[0]^d[2]^d[4]^d[5]^d[7]^d[8]^d[9]^d[14]^d[15]^d[17]^d[22]^d[25]^d[27]^d[28]^d[31];
  c[23] = d[0]^d[1]^d[3]^d[5]^d[6]^d[8]^d[9]^d[10]^d[12]^d[15]^d[18]^d[23]^d[24]^d[26]^d[29];
  c[24] = d[1]^d[4]^d[5]^d[6]^d[7]^d[10]^d[11]^d[13]^d[14]^d[18]^d[20]^d[21]^d[23]^d[24]^d[31];
  c[25] = d[1]^d[2]^d[4]^d[8]^d[10]^d[13]^d[15]^d[18]^d[19]^d[20]^d[22]^d[23]^d[25]^d[28]^d[31];
  c[26] = d[2]^d[3]^d[4]^d[5]^d[8]^d[9]^d[11]^d[12]^d[14]^d[16]^d[19]^d[21]^d[23]^d[26]^d[29];
  c[27] = d[0]^d[3]^d[4]^d[5]^d[6]^d[9]^d[10]^d[12]^d[13]^d[15]^d[17]^d[20]^d[22]^d[27]^d[30];
  c[28] = d[0]^d[1]^d[2]^d[3]^d[5]^d[9]^d[10]^d[14]^d[15]^d[16]^d[17]^d[19]^d[22]^d[27]^d[28];
  c[29] = d[0]^d[5]^d[6]^d[9]^d[11]^d[12]^d[14]^d[16]^d[18]^d[19]^d[22]^d[23]^d[24]^d[27]^d[29];
  c[30] = d[0]^d[1]^d[6]^d[7]^d[8]^d[10]^d[12]^d[13]^d[15]^d[17]^d[19]^d[20]^d[23]^d[25]^d[30];
  c[31] = d[0]^d[1]^d[2]^d[4]^d[7]^d[8]^d[9]^d[11]^d[13]^d[14]^d[16]^d[18]^d[21]^d[26]^d[31];
  memcpy(d, c, ILEN);
}
```
As the repeated shifting/bitmasking and branching costs a lot of performance, whereas this is a linear time operation that is very cache optimal, as the 32 byte d-chunk is just loaded into code.

3. To get an efficient reverse1, I first built a vector from the python `rfw1np` from [`reverse_fast/split.py`](python/reverse_fast/split.py) that has a dimension of 256 x [0 | 1 | 2] so that values with dimension 0 are 0, the values with dimension 1 are 1+value, and the values with 2 dimensions are stored in the most and least significant byte as 1+value+value2<<8. The limitation to this method is, that value1 and value2 can't both be 255, as it would overflow, but the values can't match anyways, so this limitation never applies.
   
The generator code for this, assuming our rfw1np from python is the following:

```python
rfw1np = [[106], [26], [24, 94], [89], [56], [86], [227], [195], [228], [95], [188], [144], [47], [223], [70], [], [88], [], [245], [42], [247], [191], [128], [10], [204], [221], [23], [246], [83, 113], [131], [129], [4], [], [130], [240], [145], [48], [2], [112], [158], [7], [253], [232], [9], [118], [132, 220], [222], [176], [81], [251], [200], [6, 154], [140], [115], [99], [210], [110], [68], [197], [216], [148], [218], [], [230], [161], [162], [37], [117], [], [248], [238], [100], [18], [77, 209], [43], [33], [125], [67], [146], [183], [189], [92], [82], [39], [149], [14], [104], [184, 207], [193], [75], [], [215], [59], [15], [123], [198], [252], [121], [22], [40], [91], [239], [41, 181], [214], [196], [80], [34], [], [62], [155], [51], [244], [233], [31], [32], [255], [119], [], [49], [127], [21, 151], [157], [174], [192], [8], [236], [226], [103], [], [60], [165], [30], [142], [45], [65], [250], [185], [153], [72], [205], [71], [137, 171], [96], [16], [186], [116], [61], [166, 224], [3], [134], [76], [93], [180], [213], [234], [87], [187], [122], [], [201], [69], [53], [58, 243], [172], [152], [50], [168], [150], [90], [133], [28], [101], [0], [97], [173], [], [13], [], [85], [5], [120], [219], [175], [208], [135], [242], [66], [217], [11, 194], [178], [229], [126], [38], [44], [57], [136], [159], [164], [55, 64], [249], [235], [27], [105], [], [203], [170], [102], [46], [73], [1], [17], [20], [19], [], [143], [52], [111, 237], [231], [63], [139], [114], [167], [211], [163], [84], [141], [206], [107], [], [179], [78], [190], [35], [36, 124], [79], [202], [54], [108], [199], [169], [29], [109], [25], [182], [241], [212], [12], [98, 254], [138], [160], [], [156], [225], [177], [74], [147]]
def reduce(arr):
     rfw1np = 0
     if not arr: return 0
     while arr: rfw1np = rfw1np << 8 | arr.pop()
     return rfw1np+1
print(list(map(reduce, rfw1np))) 
```

Resulting in the u16 array
```
CONST u16 rev1[256] = {107, 27, 24089, 90, 57, 87, 228, 196, 229, 96, 189, 145, 48, 224, 71, 0, 89, 0, 246, 43, 248, 192, 129, 11, 205, 222, 24, 247, 29012, 132, 130, 5, 0, 131, 241, 146, 49, 3, 113, 159, 8, 254, 233, 10, 119, 56453, 223, 177, 82, 252, 201, 39431, 141, 116, 100, 211, 111, 69, 198, 217, 149, 219, 0, 231, 162, 163, 38, 118, 0, 249, 239, 101, 19, 53582, 44, 34, 126, 68, 147, 184, 190, 93, 83, 40, 150, 15, 105, 53177, 194, 76, 0, 216, 60, 16, 124, 199, 253, 122, 23, 41, 92, 240, 46378, 215, 197, 81, 35, 0, 63, 156, 52, 245, 234, 32, 33, 256, 120, 0, 50, 128, 38678, 158, 175, 193, 9, 237, 227, 104, 0, 61, 166, 31, 143, 46, 66, 251, 186, 154, 73, 206, 72, 43914, 97, 17, 187, 117, 62, 57511, 4, 135, 77, 94, 181, 214, 235, 88, 188, 123, 0, 202, 70, 54, 62267, 173, 153, 51, 169, 151, 91, 134, 29, 102, 1, 98, 174, 0, 14, 0, 86, 6, 121, 220, 176, 209, 136, 243, 67, 218, 49676, 179, 230, 127, 39, 45, 58, 137, 160, 165, 16440, 250, 236, 28, 106, 0, 204, 171, 103, 47, 74, 2, 18, 21, 20, 0, 144, 53, 60784, 232, 64, 140, 115, 168, 212, 164, 85, 142, 207, 108, 0, 180, 79, 191, 36, 31781, 80, 203, 55, 109, 200, 170, 30, 110, 26, 183, 242, 213, 13, 65123, 139, 161, 0, 157, 226, 178, 75, 148};
```

4. To reduce the number of unnecessary computations, a prior check if a lines is even valid (all lookup-values must be >0) can then be done with:

```cpp
if(pos==0) // only do this check once, not for recursive calls
  for (u8 pos_copy = 0; pos_copy < ILEN; pos_copy++) {
  if (!rev1[r[pos_copy]]) {
    free(r);
    return;
  }
}
```

5. Then we tried two different approaches: 

5. a. A stack-based, recursive solution that minimizes memory usage and increases cache-locality, and is therefore faster but easily recurses >500 times and stores all intermediate states on the stack. THe full version can be found in [`stack/main.cu`](cpp/stack/main.cu)
```
void Reverse1Fork(state& state1, u16 extra) {
  state state2;
  u8 data2[ILEN];
  memcpy(data2, state1.data, ILEN);
  data2[state1.pos] = extra;
  state2.data = data2;
  state2.pos = state1.pos+1;
  state2.round = state1.round;
  Reverse1(state2);
}

void Reverse1(state& state) {
  // Removal code from 4
  for(;state.pos < ILEN; state.pos++) {
    u16 tmp = rev1[state.data[state.pos]]-1;
    if(tmp > 256)
      Reverse1Fork(state, tmp >> 8);
    state.data[state.pos] = tmp&0xFF;
  }
  state.round++;
  state.pos = 0;
  Reverse21(state);
}
```

5. b. A heap based solution that works on CUDA-devices that can be found in the file [`heap/main.cu`](cpp/heap/main.cu)
```cpp
GLOBAL void Reverse1(state& state, vector& v) {
  u8* r = state.data;
  u8 pos = state.pos;
  // Removal code from 4
  for(;pos < ILEN; pos++) {
    u16 tmp = rev1[r[pos]]-1;
    if(tmp > 256) {
      u8* rSpecial = (u8*) malloc(ILEN);
      memcpy(rSpecial, r, ILEN);
      rSpecial[pos] = tmp >> 8;
      v.emplace_back(state.round, rSpecial, pos+1);
    }
    r[pos] = tmp&0xFF;
  }
  v.emplace_back(state.round+1, r, 0);
}
```

The slightly different variants of the calling Reverse21 can be found in the according files. The heap-variant pulls a new job from the v-vector, whereas the stack-variant recursively calls sub-problems while storing the other problem on the stack in the Reverse1Fork method.

### CuPy

This code fully builds on the findings from the C++ variant. So the explanations from before apply here too.

Optimizations:

1) Similar to the C++-variant, the reverse2 can be done as a constant time operation with a limited number of XOR's, but this time we extract vectors instead of numbers and do a full batch_size of XOR's at the same time.

```python
def reverse2(d):
    d = d.T
    c = np.zeros_like(d)
    c[0] = d[0]^d[7]^d[10]^d[12]^d[13]^d[15]^d[18]^d[19]^d[21]^d[22]^d[25]^d[28]^d[29]^d[30]^d[31]
    c[1] ...
    return c.T
```

2. To maximize our parallelism, having a batch-size as big as possible is advisable. One state has 32 bytes and for the optimization proposed in 4) we also need to store it as an uint16 at some point in time. Additionally we require out-of-place computations/masking/copying, so this array has to exist 2^2 times in memory. Multiplying up these factors, we can determine that our batch_size is roughly (2^8+N), meaning that a GPU with 2^33 bytes of RAM (8GB) can hold at most 2^25 entries. Experimentation roughly confirmed this: The GPU ran out of memory for batch size 2^24, but worked fine with 2^23.

To archive these upper limits, it's required to frequently free memory, as cupy doesn't do that automatically. To maintain numpy-compatibility, we can use this method. 

```c
def free():
    if np.__name__ == 'cupy':
        np._default_memory_pool.free_all_blocks()
```

3. To get a fully parallelized variant of reverse1, the problem first can be filtered as we did in the C++ variant 4, then split into easy sub-problems (no expansion required) and hard sub-problems. Then for every byte-position, the problems are split into problems that need expansion and problems that don't. So it's a constant computation of 256*32 problems.

```python
def reverse1(d):
    d = rev1.take(d).astype(np.uint16)  # replace values in d by rev1[d]
    d = d[d.min(1) != 0, :] - 1  # remove all batch-items that contain a 0
    mask = d.max(1) > 0xFF
    easy = d[~mask]  # filter out trivial items that need no processing
    hard = d[mask]
    del d  # this clears out a lot of memory!
    for i in range(32):
        free()
        mask = hard[:, i] > 0xFF
        ignore = hard[~mask]
        low = hard[mask]
        high = hard[mask]
        low[:, i] = low[:, i] & 0xFF
        high[:, i] = high[:, i] >> 8
        hard = np.vstack([ignore, low, high])
    return np.vstack([easy, hard]).astype(np.uint8)
```

This solution can be found in [`reverse_tensor/full.py`](reverse_tensor/full.py)

## Methodology
All tests have been executed on a private system, a Ryzen1700 (3.2ghz, 8 cores, 2 hyperthreads per core), 64GB DDR4 3200MHz RAM, Geforce 2070 Super 8GB system that is representative of the SOTA in 2020. In the CPU-multithreaded versions, the scaling was nearly linear, leading to the conclusion that an increased number of cores should lead to a near-linear speedup meaning that a 1000€ 64-thread CPU could perform 4x as good. The system environment was a fully updated Arch Linux / Manjaro at 2.1.21, so Python 3.9, Numba 0.52, Cuda 11.2, gcc 10.2 and all optimizers on -O3. 

Most benchmarks do 2^24 computations, resulting in a 1.3GB results file.

## Results

★ For the very slow approaches, only 2^14 runs are done and the results are multiplied with 2^10 to interpolate the result. The solution-file of the 1^14 run is roughly 0.9MB, so this suggests that the work done could be less than 1/1^10, making the time-estimates rather conservative, even including overhead.

★★ To further aid this, for the GPU-solutions, the printing isn't enabled in the evaluation-runs (only in the test-runs to verify validity) as printing can synchronize GPUs and severely throttle performance. In a real world scenario, the results would be written into a shared memory space and read by the CPU later.

Program | Device | Hyperparameters | Time in ms
--- | --- | ---: | ---:
native python solution | CPU | 1 Thread | ★8.515.411.968
numba optimized python | CPU | 16 Threads | 3.183.405
numba + C++ optimization 2 | CPU | 32 Threads | 2.165.003
heap-based CUDA | GPU | 2 Blocks, 128 Threads | ★★350.256.128
heap-based CUDA | GPU | 16 Blocks, 16 Threads | ★★201.165.824
heap-based C++ | CPU | 32 Threads | 197.342
heap-based C++ | CPU | 16 Threads | 206.011
stack-based C++ | CPU | 32 Threads | 187.158
stack-based C++ | CPU | 16 Threads | 198.909
NumPy based | CPU | 32 Threads | 3.994.456
CuPy based | GPU | 16 Batches | 1.276.221
CuPy based | GPU | 1 Batch | 1.169.682

## Discussion

So the fastest options are the stack-based C++ variant for CPU and the CuPy variant on GPU. As the problem is quite easily dividable, so computation could be done on CPU and GPU parallel to reach slightly faster total computation time.

Some methods turned out rather disappointing, especially the GPU-Variants as GPUs require the data to be in a very specific format, and if the code requires any branching, it gets horribly slow.

Writing native CUDA-Code and diagnosing it is really painfull and error-prone, as debugging-options are limited, and you are required to write CPU-compatible code with ugly preprocessor code. And you are mostly limited to basic C-capabilities and libraries like CUDA-Thrust don't offer things like on-GPU-lists. Even CuPy code requires extra work like manually deallocating, otherwise you run out of memory very quickly.

The most relevant speedup is probably the 100x speedup from python to numba (including the 10x speedup you get from multithreading) as it mostly requires rewriting code. For this problem, numba even outperforms CuPy by a factor of 10x.

To get the best performance, a near-total rewrite of the problems in C++ or CuPy is required, so this is only applicable for very specific problems.

## Conclusion

Whilst the theoretical solving of the problem to get ANY solution was rather easy with some basics in logic, mathematics and cryptography. The real challenge was finding the best optimization and optimization strategy, for problem-sizes of 2^20+.

As there is a lot of conflicting advice out there, the answer to what optimization strategy works best is really problem dependant, but the big findings for this problem seem to be:
* Use a performant language. The overhead from python is massive.
* Optimizing bottlenecks with many memory-accesses / branches like the C++/CUDA reverse2 to constant time was one of the most significant improvements.
* Cache-friendly, parallelized, highly optimized C/C++ code on a 200€ CPU still beats batch-processing optimized, constant-instructions CUDA Code with expensive GDDR6 memory on a 500€ GPU. Even if the problem doesn't have the classic machine-learning problem of requiring huge CPU<->GPU data transfers, using the GPU here wasn't the optimal solution.
* The branching nature of the reverseal-problem in reverse1 make all optimizations very tedious, so even weak non-unique s-boxes make reversing problems incredibly hard and can be seen as the primary protection-mechanism of the challenge.