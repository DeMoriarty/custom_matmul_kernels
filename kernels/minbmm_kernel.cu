#define _VOLATILE_  

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)

typedef long long ll_t;
typedef unsigned long long ull_t;

typedef struct __builtin_align__(32) {
  float s0, s1, s2, s3, s4, s5, s6, s7;
} _float8;

typedef union {
  _float8 f8;
  float val[8];
} float8;

__device__ __forceinline__ float atomicMin(float *address, float val)
{
  int ret = __float_as_int(*address);
  while(val < __int_as_float(ret))
  {
    int old = ret;
    if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
      break;
  }
  return __int_as_float(ret);
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

__device__ void min_dim_1(
  float8 cCache[8],
  _VOLATILE_ float valSM[16][128+4],
  _VOLATILE_ float idxSM[16][128+4],
  float* values,
  ll_t* indices,
  int gStartx, int gStarty, int tid, int bid,
  int M, int N
){
  int vx = tid % 16;
  int vy = tid / 16;

  #pragma unroll
  for (int ni = 0; ni < 8; ni++){
    // initialize with first value
    float value = cCache[0].val[ni];
    float index = vy*8;

    // Reduce within thread
    #pragma unroll
    for (int mi = 1; mi < 8; mi++){
      float temp = cCache[mi].val[ni];
      int iM = gStarty + vy*8 + mi;
      if (likely(iM < M)){
        if (temp < value){
          value = temp;
          index = vy*8 + mi;
        }
      } else {
        value = INFINITY;
      }
    }

    // Store reduced values and indices in shared memory
    valSM[vy][vx * 8 + ni] = value;
    idxSM[vy][vx * 8 + ni] = index;
  }
  __syncthreads();

  // first 128 threads do block wise reduction
  if (tid < 128){
    float value = valSM[0][tid];
    float index = idxSM[0][tid];
    
    #pragma unroll
    for (int i=1; i<16; i++){
      float temp = valSM[i][tid];
      if (temp < value){
        value = temp;
        index = idxSM[i][tid];
      }
    }
    
    // global reduction
    int iN = gStartx + tid;
    if (iN < N){
      atomicMin(values + (bid) * N + iN, value);
      if (value <= values[(bid) * N + iN]){
        indices[(bid) * N + iN] = ll_t(index) + gStarty;
      }
    }
    /*
    */
  }
}

__device__ void min_dim_2(
  float8 cCache[8],
  _VOLATILE_ float valSM[16][128+4],
  _VOLATILE_ float idxSM[16][128+4],
  float* values,
  ll_t* indices,
  int gStartx, int gStarty, int tid, int bid,
  int M, int N
){
  int vx = tid % 16;
  int vy = tid / 16;

  #pragma unroll
  for (int mi = 0; mi < 8; mi++){
    // initialize with first value
    float value = cCache[mi].val[0];
    float index = vx*8;

    // Reduce within thread
    #pragma unroll
    for (int ni = 1; ni < 8; ni++){
      float temp = cCache[mi].val[ni];
      int iN = gStartx + vx*8 + ni;
      if (likely(iN < N)){
        if (temp < value){
          value = temp;
          index = vx*8 + ni;
        }
      } else {
        value = INFINITY;
      }
    }

    // Store reduced values and indices in shared memory
    valSM[vx][vy * 8 + mi] = value;
    idxSM[vx][vy * 8 + mi] = index;
  }
  __syncthreads();

  // first 128 threads do block-wise reduction
  if (tid < 128){
    float value = valSM[0][tid];
    float index = idxSM[0][tid];
    #pragma unroll
    for (int i = 1; i < 16; i++){
      float temp = valSM[i][tid];
      if (temp < value){
        value = temp;
        index = idxSM[i][tid];
      }
    }

    // global reduction
    int iM = gStarty + tid;
    if (iM < M){
      atomicMin(values + (bid) * M + iM, value);
      if (value <= values[(bid) * M + iM]){
        indices[(bid) * M + iM] = ll_t(index) + gStartx;
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
      /*
      if (likely(iN_start + 7 < N)){
        reinterpret_cast<float8*>(C + (bid)*M*N + (iM)*N + (iN_start))[0] = cCache[i];
      } else {
        #pragma unroll
        for (int j=0; j<8; j++){
          int iN = iN_start + j;
          if (iN < N){
            C[(bid)*M*N + (iM)*N + (iN)] = cCache[i].val[j];
          }
        }
      }
      */
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
__global__ void min_bmm_tn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
){
}

extern "C"
__global__ void min_bmm_nt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
){
}

extern "C"
__global__ void min_bmm_nn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
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

  // Reduce along DIM
  if (DIM == 1){
    min_dim_1(
      cCache, aSM, bSM, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  } else if (DIM == 2){
    min_dim_2(
      cCache, aSM, bSM, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  }
}

extern "C"
__global__ void min_bmm_tt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
){
}