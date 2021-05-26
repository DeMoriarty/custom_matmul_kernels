#define _VOLATILE_  

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)
#define isnan(x) ( x != x )
#if (__CUDA_ARCH__ < 700)
__device__ void __nanosleep(unsigned int ns){
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < ns)
  {
    clock_offset = clock() - start_clock;
  }
}
#endif 

typedef long long ll_t;
typedef unsigned long long ull_t;

typedef struct __builtin_align__(32) {
  float s0, s1, s2, s3, s4, s5, s6, s7;
} _float8;

typedef union {
  _float8 f8;
  float val[8];
} float8;

__device__ void mutex_lock(
  unsigned int *mutex
) {
  unsigned int ns = 8;
  __syncthreads();
  if (threadIdx.x == 0 ){
    while (atomicCAS(mutex, 0, 1) == 1) {
      __nanosleep(ns);
      if (ns < 256) {
        ns *= 2;
      }
    }
  }
  __syncthreads();
}

__device__ void mutex_lock_noop(
) {
  __syncthreads();
}

__device__ void mutex_unlock(
  unsigned int *mutex
) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0){
    atomicExch(mutex, 0);
    __threadfence();
  }
  __syncthreads();
}

__device__ void mutex_unlock_noop(){
  __syncthreads();
  __syncthreads();
}

__device__ __forceinline__ unsigned int bfe(
  unsigned int source,
  unsigned int bitIndex
) {
  unsigned int bit;
  asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bit) : "r"((unsigned int) source), "r"(bitIndex), "r"(1));
  return bit;
}

__device__ __forceinline__ void warpComparator(
  float &value,
  float &index,
  const int stride,
  const int direction
){
  const float other_value = __shfl_xor_sync(0xFFFFFFFF, value, stride);
  const float other_index = __shfl_xor_sync(0xFFFFFFFF, index, stride);
  bool condition = value < other_value == direction;
  index = condition ? other_index : index;
  value = condition ? other_value : value;
}

__device__ __forceinline__ void blockComparator(
  float &value,
  float &index,
  const int stride,
  const int direction,
  const int laneID,
  _VOLATILE_ float valSM[128+4],
  _VOLATILE_ float idxSM[128+4]
){
  valSM[laneID] = value;
  idxSM[laneID] = index;
  __syncthreads();

  float other_value = valSM[laneID ^ stride];
  float other_index = idxSM[laneID ^ stride];
  __syncthreads();

  bool condition = value < other_value == direction;
  index = condition ? other_index : index;
  value = condition ? other_value : value;
}

__device__ void bitonicSort128(
  float &value,
  float &index,
  _VOLATILE_ float valSM[128+4],
  _VOLATILE_ float idxSM[128+4]
) {
  unsigned int laneID = threadIdx.x % 128;
  warpComparator(value, index, 1, bfe(laneID, 1) ^ bfe(laneID, 0));

  warpComparator(value, index, 2, bfe(laneID, 2) ^ bfe(laneID, 1));
  warpComparator(value, index, 1, bfe(laneID, 2) ^ bfe(laneID, 0));

  warpComparator(value, index, 4, bfe(laneID, 3) ^ bfe(laneID, 2));
  warpComparator(value, index, 2, bfe(laneID, 3) ^ bfe(laneID, 1));
  warpComparator(value, index, 1, bfe(laneID, 3) ^ bfe(laneID, 0));

  warpComparator(value, index, 8, bfe(laneID, 4) ^ bfe(laneID, 3));
  warpComparator(value, index, 4, bfe(laneID, 4) ^ bfe(laneID, 2));
  warpComparator(value, index, 2, bfe(laneID, 4) ^ bfe(laneID, 1));
  warpComparator(value, index, 1, bfe(laneID, 4) ^ bfe(laneID, 0));

  warpComparator(value, index, 16, bfe(laneID, 5) ^ bfe(laneID, 4));
  warpComparator(value, index, 8, bfe(laneID, 5) ^ bfe(laneID, 3));
  warpComparator(value, index, 4, bfe(laneID, 5) ^ bfe(laneID, 2));
  warpComparator(value, index, 2, bfe(laneID, 5) ^ bfe(laneID, 1));
  warpComparator(value, index, 1, bfe(laneID, 5) ^ bfe(laneID, 0));

  blockComparator(value, index, 32, bfe(laneID, 6) ^ bfe(laneID, 5), laneID, valSM, idxSM);
  warpComparator(value, index, 16, bfe(laneID, 6) ^ bfe(laneID, 4));
  warpComparator(value, index, 8, bfe(laneID, 6) ^ bfe(laneID, 3));
  warpComparator(value, index, 4, bfe(laneID, 6) ^ bfe(laneID, 2));
  warpComparator(value, index, 2, bfe(laneID, 6) ^ bfe(laneID, 1));
  warpComparator(value, index, 1, bfe(laneID, 6) ^ bfe(laneID, 0));

  blockComparator(value, index, 64, bfe(laneID, 6), laneID, valSM, idxSM);
  blockComparator(value, index, 32, bfe(laneID, 5), laneID, valSM, idxSM);
  warpComparator(value, index, 16, bfe(laneID, 4));
  warpComparator(value, index, 8, bfe(laneID, 3));
  warpComparator(value, index, 4, bfe(laneID, 2));
  warpComparator(value, index, 2, bfe(laneID, 1));
  warpComparator(value, index, 1, bfe(laneID, 0));
}

__device__ void bitonicSort256(
  float &value,
  float &index,
  float* gValue,
  ll_t* gIndex,
  float valSM[128+4],
  float idxSM[128+4],
  int Q
){
  int laneID = threadIdx.x % 128;
  float other_value = gValue[0];
  float other_index = gIndex[0];
  
  bool condition = value > other_value == 0;
  if (condition){
    // swap values without 3rd variable
    value = value + other_value;
    index = index + other_index;
    other_value = value - other_value;
    other_index = index - other_index;
    value = value - other_value;
    index = index - other_index;
  }

  blockComparator(value, index, 64, !bfe(laneID, 6), laneID, valSM, idxSM);
  blockComparator(value, index, 32, !bfe(laneID, 5), laneID, valSM, idxSM);
  warpComparator(value, index, 16, !bfe(laneID, 4));
  warpComparator(value, index, 8, !bfe(laneID, 3));
  warpComparator(value, index, 4, !bfe(laneID, 2));
  warpComparator(value, index, 2, !bfe(laneID, 1));
  warpComparator(value, index, 1, !bfe(laneID, 0));

  if ( laneID < Q){
    gValue[0] = value;
    gIndex[0] = index;
  }
}

__device__ void topk_dim_1(
  float8 cCache[8],
  _VOLATILE_ float valSM[16][128+4],
  _VOLATILE_ float idxSM[16][128+4],
  float* values,
  ll_t* indices,
  unsigned int* mutex,
  int gStartx, int gStarty, int bid,
  int M, int N, int Q
){
  int tid = threadIdx.x;
  int vx = tid % 16;
  int vy = tid / 16;
  int hx = tid % 128;
  int hy = tid / 128;
  #pragma unroll
  for (int ni=0; ni<8; ni++){
    if (gStartx + vx*8 + ni >= N)
      break;

    // Store cCache to cSM
    #pragma unroll
    for (int mi=0; mi<8; mi++){
      int iM = gStarty + vy*8 + mi;
      if (likely(iM < M)){
        valSM[vx][vy*8 + mi] = cCache[mi].val[ni];
        idxSM[vx][vy*8 + mi] = iM;
      } else {
        valSM[vx][vy*8 + mi] = -INFINITY;
        idxSM[vx][vy*8 + mi] = iM;
      }
    }
    __syncthreads();
    // Load from cSM to cCache
    #pragma unroll
    for (int i=0; i<8; i++){
      float value = valSM[hy*8 + i][hx];
      float index = idxSM[hy*8 + i][hx];
      bitonicSort128(
        value, index,
        valSM[hy*8 + i], idxSM[hy*8 + i]
      );
      int iN = gStartx + (hy*8 + i)*8 + ni;
      mutex_lock( &mutex[(bid)*N + iN] );
      bitonicSort256(
        value, index, 
        &values[(bid)*N*Q + iN*Q + hx],
        &indices[(bid)*N*Q + iN*Q + hx], 
        valSM[hy*8+i], idxSM[hy*8+i],
        Q
      );
      mutex_unlock( &mutex[(bid)*N + iN] );
    }
  }
}

__device__ void topk_dim_2(
  float8 cCache[8],
  _VOLATILE_ float valSM[16][128+4],
  _VOLATILE_ float idxSM[16][128+4],
  float* values,
  ll_t* indices,
  unsigned int* mutex,
  int gStartx, int gStarty, int bid,
  int M, int N, int Q
){
  int tid = threadIdx.x;
  int vx = tid % 16;
  int vy = tid / 16;
  int hx = tid % 128;
  int hy = tid / 128;
  #pragma unroll
  for (int mi=0; mi<8; mi++){
    if (gStarty + vy*8 + mi >= M)
      break;

    // Store cCache to cSM
    #pragma unroll
    for (int ni=0; ni<8; ni++){
      int iN = gStartx + vx*8 + ni;
      if (likely(iN < N)){
        valSM[vy][vx*8 + ni] = cCache[mi].val[ni];
        idxSM[vy][vx*8 + ni] = iN;
      } else {
        valSM[vy][vx*8 + ni] = -INFINITY;
      }
    }
    __syncthreads();
    // Load from cSM to cCache
    #pragma unroll
    for (int i=0; i<8; i++){
      float value = valSM[hy*8 + i][hx];
      float index = idxSM[hy*8 + i][hx];
      bitonicSort128(
        value, index,
        valSM[hy*8 + i], idxSM[hy*8 + i]
      );
      int iM = gStarty + (hy*8 + i)*8 + mi;
      mutex_lock( &mutex[(bid)*M + iM] );
      bitonicSort256(
        value, index, 
        &values[(bid)*M*Q + iM*Q + hx],
        &indices[(bid)*M*Q + iM*Q + hx], 
        valSM[hy*8+i], idxSM[hy*8+i],
        Q
      );
      mutex_unlock( &mutex[(bid)*M + iM] );
    }
  }
}

__device__ void init_cCache(
  float8 cCache[8]
) {
  #pragma unroll
  for (int i=0; i<8; i++){
    #pragma unroll
    for (int j=0; j<8; j++){
      cCache[i].val[j] = 0.f;
    }
  }
}

__device__ void thread_matmul_v4(
  _VOLATILE_ float aSM[8][128+4],
  _VOLATILE_ float bSM[8][128+4],
  float8 cCache[8],
  int vx, int vy
) {
  float aCache1[8];
  float aCache2[8];
  #pragma unroll
  for (int mi=0; mi<8; mi++){
    aCache1[mi] = aSM[0][8*vy + mi];
  }

  #pragma unroll
  for (int ki=0; ki<8; ki++){
    int is_odd = ki & 1;
    if (is_odd == 0){
      if (likely(ki < 7)){
        #pragma unroll
        for (int mi=0; mi<8; mi++){
          aCache2[mi] = aSM[ki+1][8*vy + mi];
        }
      }
      #pragma unroll
      for (int ni=0; ni<8; ni++){
        float b = bSM[ki][vx/4 + 8*vx + ni];
        #pragma unroll
        for (int mi=0; mi<8; mi++){
          float a = aCache1[mi];
          cCache[mi].val[ni] = fmaf(a, b, cCache[mi].val[ni]);
        }
      }
    } else {
      if (likely(ki < 7)){
        #pragma unroll
        for (int mi=0; mi<8; mi++){
          aCache1[mi] = aSM[ki+1][8*vy + mi];
        }
      }
      #pragma unroll
      for (int ni=0; ni<8; ni++){
        float b = bSM[ki][vx/4 + 8*vx + ni];
        #pragma unroll
        for (int mi=0; mi<8; mi++){
          float a = aCache2[mi];
          cCache[mi].val[ni] = fmaf(a, b, cCache[mi].val[ni]);
        }
      }
    }
  }
}

__device__ void thread_matmul_v3(
  _VOLATILE_ float aSM[16][128+4],
  _VOLATILE_ float bSM[16][128+4],
  float8 cCache[8],
  int vx, int vy
) {
  float aCache[8];

  #pragma unroll
  for (int ki=0; ki<16; ki++){
    #pragma unroll
    for (int mi=0; mi<8; mi++){
      aCache[mi] = aSM[ki][8*vy + mi];
    }
    #pragma unroll
    for (int ni=0; ni<8; ni++){
      float b = bSM[ki][vx/4 + 8*vx + ni];
      #pragma unroll
      for (int mi=0; mi<8; mi++){
        float a = aCache[mi];
        cCache[mi].val[ni] = fmaf(a, b, cCache[mi].val[ni]);
      }
    }
  }
}

// Unsafe
__device__ void write_c(
  float8 cCache[8],
  float* C,
  int gStartx, int gStarty,
  int vx, int vy, int bid,
  int M, int N
) {
  #pragma unroll
  for (int i=0; i<8; i++){
    int iM = gStarty + vy*8 + i;
    if (likely(iM < M)){
      int iN_start = gStartx + vx*8;
      reinterpret_cast<float8*>(C + (bid)*M*N + (iM)*N + (iN_start))[0] = cCache[i];
    }
  }
}

__device__ void write_c_v3(
  float8 cCache[8],
  float* C,
  int gStartx, int gStarty,
  int vx, int vy, int bid,
  int M, int N
) {
  __shared__ volatile float cSM[16][128];
  #pragma unroll
  for (int mi=0; mi<8; mi++){
    int iM = gStarty + vy*8 + mi;
    // Store 1 row from cCache to cSM
    if (iM < M){
      #pragma unroll
      for (int ni=0; ni<8; ni++){
        cSM[vy][vx*8 + ni] = cCache[mi].val[ni];
      }
      // Store to C
      #pragma unroll
      for (int ni=0; ni<8; ni++){
        int iN = gStartx + 16*ni + vx;
        if (iN < N){
          float cVal = cSM[vy][16*ni + vx];
          store(C+(bid)*M*N + (iM)*N + (iN), cVal);
        }
      }
    }
  } 
}

extern "C"
__global__ void topk_bmm_tn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
){
}

extern "C"
__global__ void topk_bmm_nt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
){
}

extern "C"
__global__ void topk_bmm_nn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* values,
  ll_t* indices,
  unsigned int* mutex,
  int M, int N, int K, int DIM, int Q
){
  int tid = threadIdx.x;     // thread idx
  int bid = blockIdx.z;      // batch idx

  // Neighboring blocks are grouped into PN x PM block groups in order to increase
  // L1 cache hit rate
  // There are ceil(M/PM) x ceil(N/PN) block groups in total.
  // Blocks within block groups are indexed with blockIdx.x % PN and blockIdx.x / PN
  int px = blockIdx.x % _PN_;
  int py = blockIdx.x / _PN_;
  int bDimX = (N + (128*_PN_) - 1) / (128*_PN_); 
  int bDimY = (M + (128*_PM_) - 1) / (128*_PM_); 
  int bIdxX = (blockIdx.y % bDimX) * _PN_ + px;
  int bIdxY = (blockIdx.y / bDimX) * _PM_ + py;
  int gStartx = bIdxX * 128;   // starting index of block on N axis
  int gStarty = bIdxY * 128;   // starting index of block on M axis
  if (gStartx > N || gStarty > M){
    return;
  }
  // These are used to re-arrange threads into different shapes
  // for example: (256) -> (16, 16) -> (8, 32) -> (32, 8)
  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ _VOLATILE_ float aSM[16][128+4];
  __shared__ _VOLATILE_ float bSM[16][128+4];

  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  // Load initial 16 x 128 tile of A and B to buffer1 and buffer2
  #pragma unroll
  for (int i=0; i<4; i++){
    int iM = gStarty + dy + i*32;
    int iN = gStartx + wx + i*32;
    if (likely(iM < _M_)){
      if (likely(dx < _K_)){
        aBuffer1[i] = load(A + (bid)*_M_*_K_ + (iM)*_K_ + (dx));
      } else {
        aBuffer1[i] = 0.f;
      }
      if (likely(dx+8 < _K_)){
        aBuffer2[i] = load(A + (bid)*_M_*_K_ + (iM)*_K_ + (dx+8));
      } else {
        aBuffer2[i] = 0.f;
      }
    }
    if (likely(iN < N)){
      if (likely(wy < _K_)){
        bBuffer1[i] = load(B + (bid)*_N_*_K_ + (wy)*_N_ + (iN));
      } else {
        bBuffer1[i] = 0.f;
      }
      if (likely(wy+8 < _K_)){
        bBuffer2[i] = load(B + (bid)*_N_*_K_ + (wy+8)*_N_ + (iN));
      } else {
        bBuffer2[i] = 0.f;
      }
    }
  }

  // Number of main loop iterations is ceil(k/16)
  int nIt = (_K_ + 16 - 1) / 16;
  #pragma unroll
  for (int itr=0; itr<nIt; itr++){
    int gStartk = itr * 16;

    // Index on K axis of A and B
    int iKA = gStartk + 16 + dx;
    int iKB = gStartk + 16 + wy;

    #pragma unroll
    for (int i=0; i<4; i++){
      // Store buffered tiles into shared memory
      aSM[dx][dy+i*32] = aBuffer1[i];
      bSM[wy][wx+i*32+i] = bBuffer1[i];
      aSM[8 + dx][dy+i*32] = aBuffer2[i];
      bSM[8 + wy][wx+i*32+i] = bBuffer2[i];

      // Start loading next 16*128 tile of A and B to buffer1 and buffer2.
      // Don't load anything on the last iteration.
      // Loading from global memory will not block thread_matmul
      if (likely(itr < nIt - 1)){
        int iM = gStarty + i*32 + dy;
        int iN = gStartx + i*32 + wx;
        
        if (likely(iM < _M_)){
          if (likely(iKA < _K_)){
            aBuffer1[i] = load(A + (bid)*_M_*_K_ + (iM)*_K_ + (iKA));
          } else {
            aBuffer1[i] = 0.f;
          }
          if (likely(iKA+8 < _K_)){
            aBuffer2[i] = load(A + (bid)*_M_*_K_ + (iM)*_K_ + (iKA+8));
          } else {
            aBuffer2[i] = 0.f;
          }
        }

        if (likely(iN < _N_)){
          if (likely(iKB < _K_)){
            bBuffer1[i] = load(B + (bid)*_N_*_K_ + (iKB)*_N_ + (iN));
          } else {
            bBuffer1[i] = 0.f;
          }
          if (likely(iKB+8 < _K_)){
            bBuffer2[i] = load(B + (bid)*_N_*_K_ + (iKB+8)*_N_ + (iN));
          } else {
            bBuffer2[i] = 0.f;
          }
        }
      }
    }
    // synchroznie threads in order make sure tiles of A and B are fully
    // loaded to shared memory.
    __syncthreads();

    thread_matmul_v3(aSM, bSM, cCache, vx, vy);

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }

  // TopK sort along DIM
  if (DIM == 1){
    topk_dim_1(
      cCache, aSM, bSM,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  } else if (DIM == 2){
    topk_dim_2(
      cCache, aSM, bSM,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  }
}

extern "C"
__global__ void topk_bmm_tt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
){
}