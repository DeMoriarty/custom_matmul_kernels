typedef long long ll_t;
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

/*
mutex lock code from:
https://stackoverflow.com/questions/18963293/cuda-atomics-change-flag/18968893#18968893
*/

__device__ void mutex_lock_v2(
  unsigned int *mutex
) {
  unsigned int ns = 8;
  __syncthreads();
  if (threadIdx.x == 0){
    while (atomicCAS(mutex, 0, 1) == 1) {
      __nanosleep(ns);
      if (ns < 256) {
        ns *= 2;
      }
    }
  }
  __syncthreads();
}

__device__ void mutex_lock(
  unsigned int *mutex,
  unsigned int blockMutex[1]
) {
  unsigned int ns = 8;
  float old_value;
  if (threadIdx.x == 0){
    old_value = atomicCAS(mutex, 0, 1);
    blockMutex[0] = old_value;
  }
  __syncthreads();
  old_value = blockMutex[0];
  while (old_value == 1) {
    __nanosleep(ns);
    if (ns < 256) {
      ns *= 2;
    }

    if (threadIdx.x == 0){
      old_value = atomicCAS(mutex, 0, 1);
      blockMutex[0] = old_value;
    }
    __syncthreads();
    old_value = blockMutex[0];
    __syncthreads();
  }
}

__device__ void mutex_unlock_v2(unsigned int *mutex) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0){
    atomicExch(mutex, 0);
    __threadfence();
  }
  __syncthreads();
}

__device__ void mutex_unlock(unsigned int *mutex) {
  atomicExch(mutex, 0);
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
  float valSM[128],
  float idxSM[128]
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

__device__ void bitonicSort256(
  float &value,
  float &index,
  float* values,
  ll_t* indices,
  float valSM[128],
  float idxSM[128],
  int gStartx, int Q
){
  float other_value = values[threadIdx.x];
  float other_index = indices[threadIdx.x] - gStartx;
  
  bool condition = value > other_value == 0;
  if (condition){
    float temp_value = value;
    float temp_index = index;
    value = other_value;
    index = other_index;
    other_value = temp_value;
    other_index = temp_index;
  }

  int laneID = threadIdx.x % 128;
  int i = 7;
  for (int j = 6; j >= 0; j--){
    unsigned int direction = bfe(laneID, 8) ^ bfe(laneID, j);
    int stride = pow(2, j);
    if (stride < 32){
      warpComparator(value, index, stride, !direction);
    } else {
      blockComparator(value, index, stride, !direction, laneID, valSM, idxSM);
    }
  }

  if (threadIdx.x < Q){
    values[threadIdx.x] = value;
    indices[threadIdx.x] = index + gStartx;
  }
}

__device__ void bitonicSort(
  float &value,
  float &index,
  float valSM[128],
  float idxSM[128]
) {
  unsigned int laneID = threadIdx.x % 128;
  for (int i=0; i < 7; i++){
    for (int j=i; j >= 0; j--){
      unsigned int direction = bfe(laneID, i + 1) ^ bfe(laneID, j);
      int stride = pow(2, j);
      if (stride < 32){
        warpComparator(value, index, stride, direction);
      } else {
        blockComparator(value, index, stride, direction, laneID, valSM, idxSM);
      }
    }
  }
}

extern "C"
__global__ void bitonic_sort(
   const float* __restrict__ arr,
   float* values,
   ll_t* indices,
   unsigned int* mutex,
   int L, int Q
){
  int gStartx = blockIdx.x * 128;
  int tid = threadIdx.x;
  __shared__ float valSM[128];
  __shared__ float idxSM[128];
  
  float value;
  float index;
  int iL = gStartx + tid;
  if (iL < L){
    value = arr[iL];
    index = tid;
  } else {
    value = -INFINITY;
  }
  
  bitonicSort(value, index, valSM, idxSM);

  __shared__ unsigned int blockMutex[1];
  mutex_lock_v2(mutex);

  bitonicSort256(
    value, index, values, indices,
    valSM, idxSM, gStartx, Q
  );
  
  mutex_unlock_v2(mutex);
}