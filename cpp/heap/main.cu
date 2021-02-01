#include <iostream>
#include <cstdio>
#include <chrono>
#include <thread>
#include "main.h"

u64 time() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

GLOBAL void printState(const u8* r) {
  printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n", r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
}

GLOBAL void Forward1(u8 *c) {
  u8 d[ILEN];
  for(u8 j = 0; j < ILEN; j++)
    d[j] = confusion[c[j]];
  memcpy(c, d, ILEN);
}

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

GLOBAL bool Forward3(const u8 *c) {
  for(u8 i = 0; i < OLEN; i++) {
    u8 fw = confusion[c[i * 2]] ^ confusion[c[i * 2 + 1] + 256];
    if(fw != target[i])
      return false;
  }
  return true;
}

#ifndef CUDA
  __host__ void Forward(u32 index) {
#else
  __global__ void Forward() {
  u32 index = threadIdx.x;
#endif
  u8 c[ILEN];
  u64 start = index * BATCH_SIZE;
  u64 end = start + BATCH_SIZE;

  for(u64 batch = start; batch < end; batch++) {
    memcpy(c, &batch, sizeof(u64));
    for(u32 i = 0; i < ROUNDS; i++) {
      Forward1(c);
      Forward2(c);
    }
    if(Forward3(c))
      printf("%ld\n", batch);
  }
}

GLOBAL void Reverse3(uint64_t a, u8* out) {
  for(int i=0;i<OLEN; i++) {
    auto tr = rev3[i][a & 0xFF];
    out[i*2] = tr[0];
    out[i*2+1] = tr[1];
    a = a >> 8;
  }
}

GLOBAL void Reverse1(state& state, vector& v) {
  u8* r = state.data;
  u8 pos = state.pos;
  if(pos==0) // only do this check once, not for recursive calls
    for (u8 pos_copy = 0; pos_copy < ILEN; pos_copy++) {
      if (!rev1[r[pos_copy]]) {
        free(r);
        return;
      }
    }

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

GLOBAL void Reverse21(vector& v) {
    while(!v.empty()) {
      auto state = v.pop();
      if (state.round < ROUNDS) {
        if(!state.pos)
          Forward2(state.data);
        Reverse1(state, v);
      } else
        printState(state.data);
    }
}

#ifndef CUDA
__host__ void Reverse(u32 index) {
#else
__global__ void Reverse() {
  u32 index = threadIdx.x;
#endif
  u64 start = index * BATCH_SIZE;
  u64 end = start + BATCH_SIZE;
  vector* v = new vector();

  for (u64 batch = start; batch < end; batch++) {
    u8* r = (u8*) malloc(ILEN);
    Reverse3(batch, r);
    v->emplace_back(0, r, 0);
    Reverse21(*v);
  }
}


int main() {
  u64 start = time();

#ifdef CUDA
  Reverse<<<1, NR_CORES>>>();
    cudaDeviceSynchronize();
#else
  std::thread threads[NR_CORES];
  for(int i=0;i<NR_CORES;i++)
    threads[i] = std::thread(Reverse, i);

  for(int i=0;i<NR_CORES;i++)
    threads[i].join();
#endif
  printf("Execution time = %ld; cnt: %d\n", time()-start, cnt);
}
