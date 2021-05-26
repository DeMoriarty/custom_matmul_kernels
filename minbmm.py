import torch
import cupy as cp
import numpy as np
import math
from custom_kernel import CustomKernel

class MinBMMCUDA(CustomKernel): 
  def __init__(self, m=None, n=None, k=None, patch_m=4, patch_n=4):
    super(MinBMMCUDA, self).__init__()
    self.m = m
    self.n = n
    self.k = k
    self.patch_m = patch_m
    self.patch_n = patch_n
    
    with open("kernels/minbmm_kernel.cu",'r') as f: ###
      self.kernel = f.read()
      
    self.kernel = (self.kernel
      .replace("_M_", str(m) if m else "M")
      .replace("_N_", str(n) if n else "N")
      .replace("_K_", str(k) if k else "K")
      .replace("_PM_", str(self.patch_m))
      .replace("_PN_", str(self.patch_n))
    )
    
    self._fn_tt = cp.RawKernel(
      code=self.kernel,
      name="min_bmm_tt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nn = cp.RawKernel(
      code=self.kernel,
      name="min_bmm_nn",
      backend='nvcc',
      options=(
        '--maxrregcount=128',
        '--use_fast_math',
        #'-Xptxas',
        #'-dlcm=cg',
      )
    )
    # print(self._fn_nn.attributes)
    self._fn_tn = cp.RawKernel(
      code=self.kernel,
      name="min_bmm_tn",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nt = cp.RawKernel(
      code=self.kernel,
      name="min_bmm_nt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )

  def _call_nn(self, A, B, dim):
    assert A.shape[0] == B.shape[0]
    assert A.shape[2] == B.shape[1]
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)
    assert dim in [1, 2]
    
    l, m, k = A.shape
    l, k, n = B.shape

    if self.m is not None: assert m == self.m
    if self.n is not None: assert n == self.n
    if self.k is not None: assert k == self.k

    if dim == 1:
      values = torch.empty([l, n], device="cuda:0", dtype=A.dtype)
      indices = torch.empty([l, n], device="cuda:0", dtype=torch.int64)
    elif dim == 2:
      values = torch.empty([l, m], device="cuda:0", dtype=A.dtype)
      indices = torch.empty([l, m], device="cuda:0", dtype=torch.int64)
    values.fill_(float("inf"))

    threads_per_block = (256,)
    #blocks_per_grid = (math.ceil(n/128), math.ceil(m/128), l)
    
    n_ = math.ceil(n/(128*self.patch_n))
    m_ = math.ceil(m/(128*self.patch_m))
    blocks_per_grid = (self.patch_n*self.patch_m, n_ * m_, l)
    # print(blocks_per_grid, m_, n_)

    self._fn_nn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        m, n, k, dim,
      ],
      stream=self.stream
    )
    return values, indices

  def _call_tt(self, A, B, dim):
    raise NotImplementedError

  def _call_tn(self, A, B, dim):
    raise NotImplementedError

  def _call_nt(self, A, B, dim):
    raise NotImplementedError

  def __call__(self, A, B, dim=1, mode="nn"):
    """
      Performs C = min(f(A) @ g(B)), argmin(f(A) @ g(B))

      A:
        torch.Tensor
        shape : [m, k] or [k, m] or [l, m, k] or [l, k, m]
        dtype : float32

      B:
        torch.Tensor
        shape : [n, k] or [k, n] or [l, n, k] or [l, k, n]
        dtype : float32

      returns: (values, indices)
        values:
          torch.Tensor
          shape : [m] or [l, m] or [n] or [l, n] depending on *dim*
          dtype : float32

        indices:
          torch.Tensor
          shape : [m] or [l, m] or [n] or [l, n] depending on *dim*
          dtype : int64

      dim: {0, 1, 2}
        the dimention to sort along
        when inputs have 2 dimentions, *dim* can be 0 or 1
        when inputs have 3 dimentions, *dim* can be 1 or 2, 0 is not supported.

      mode: {"nn", "tn", "nt", "tt"}, default: "nn"

      Notes:
        f() and g() are determined by *mode*
        "nn" --> A @ B
        "tt" --> A.T @ B.T
        "nt" --> A @ B.T
        "tn" --> A.T @ B
    """
    assert len(A.shape) == len(B.shape)
    A = A.contiguous()
    B = B.contiguous()
    if len(A.shape) == 2 and len(B.shape) == 2:
      A2 = A[None]
      B2 = B[None]
      dim += 1
    elif len(A.shape) == 3 and len(B.shape) == 3:
      A2 = A
      B2 = B
    else:
      raise ValueError("shape of A and B need to be 2d or 3d")

    if mode == "nn":
      values, indices = self._call_nn(A2, B2, dim)
    elif mode == "tt":
      values, indices = self._call_tt(A2, B2, dim)
    elif mode == "tn":
      values, indices = self._call_tn(A2, B2, dim)
    elif mode == "nt":
      values, indices = self._call_nt(A2, B2, dim)

    if len(A.shape) == 2 and len(B.shape) == 2:
      indices = indices[0]
      values = values[0]

    return values, indices