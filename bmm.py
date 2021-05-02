import torch
import cupy as cp
import numpy as np
import math
from custom_kernel import CustomKernel

class BMM(CustomKernel): 
  def __init__(self, m=None, n=None, k=None):
    super(BMM, self).__init__()
    self.m = m
    self.n = n
    self.k = k
    with open("kernels/bmm_kernel.cu",'r') as f: ###
      self.kernel = f.read()
      
    self.kernel = (self.kernel
      .replace("_M_", str(m) if m else "M")
      .replace("_N_", str(n) if n else "N")
      .replace("_K_", str(k) if k else "K")
    )
    
    self._fn_tt = cp.RawKernel(
      code=self.kernel,
      name="bmm_tt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nn = cp.RawKernel(
      code=self.kernel,
      name="bmm_nn",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_tn = cp.RawKernel(
      code=self.kernel,
      name="bmm_tn",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nt = cp.RawKernel(
      code=self.kernel,
      name="bmm_nt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )

  def _call_nn(self, A, B):
    """
      Performs C = A @ B
      A: shape = [l, m, k]
      B: shape = [l, k, n]
      returns C: shape = [l, m, n]
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[2] == B.shape[1]
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)
    
    l, m, k = A.shape
    l, k, n = B.shape

    if self.m is not None: assert m == self.m
    if self.n is not None: assert n == self.n
    if self.k is not None: assert k == self.k

    C = torch.zeros([l, m, n], device="cuda:0", dtype=A.dtype)

    threads_per_block = (256,)
    blocks_per_grid = (l, math.ceil(n/128), math.ceil(m/128))

    self._fn_nn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        m, n, k
      ],
      stream=self.stream
    )
    return C

  def _call_tt(self, A, B):
    """
      Performs C = A.t @ B.t
      A: shape = [l, k, m]
      B: shape = [l, n, k]
      returns C: shape = [l, m, n]
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[2]
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)
    
    l, k, m = A.shape
    l, n, k = B.shape

    if self.m is not None: assert m == self.m
    if self.n is not None: assert n == self.n
    if self.k is not None: assert k == self.k


    C = torch.zeros([l, m, n], device="cuda:0", dtype=A.dtype)

    threads_per_block = (256,)
    blocks_per_grid = (l, math.ceil(n/128), math.ceil(m/128))

    self._fn_tt(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        m, n, k
      ],
      stream=self.stream
    )
    return C

  def _call_tn(self, A, B):
    """
      Performs C = A.t @ B
      A: shape = [l, k, m]
      B: shape = [l, k, n]
      returns C: shape = [l, m, n]
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[1]
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)

    l, k, m = A.shape
    l, k, n = B.shape

    if self.m is not None: assert m == self.m
    if self.n is not None: assert n == self.n
    if self.k is not None: assert k == self.k

    C = torch.zeros([l, m, n], device="cuda:0", dtype=A.dtype)
    
    threads_per_block = (256,)
    blocks_per_grid = (l, math.ceil(n/128), math.ceil(m/128))

    self._fn_tn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        m, n, k
      ],
      stream=self.stream,
    )
    return C

  def _call_nt(self, A, B):
    """
      Performs C = A @ B.t
      A: shape = [l, m, k]
      B: shape = [l, n, k]
      returns C: shape = [l, m, n]
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[2] == B.shape[2]
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)

    l, m, k = A.shape
    l, n, k = B.shape

    if self.m is not None: assert m == self.m
    if self.n is not None: assert n == self.n
    if self.k is not None: assert k == self.k

    C = torch.zeros([l, m, n], device="cuda:0", dtype=A.dtype)

    threads_per_block = (256,)
    blocks_per_grid = (l, math.ceil(n/128), math.ceil(m/128))

    self._fn_nt(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        m, n, k
      ],
      stream=self.stream
    )
    return C

  def __call__(self, A, B, mode="nn"):
    """
      Performs C = f(A) @ f(B)
      A: torch.Tensor, shape : [l, m, k] or [l, k, m]
      B: torch.Tensor, shape : [l, n, k] or [l, k, n]
      returns C: torch.Tensor, shape : [l, m, n]
      mode: str, default: "nn"
      Notes:
        f() and g() are determined by mode
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
    elif len(A.shape) == 3 and len(B.shape) == 3:
      A2 = A
      B2 = B
    else:
      raise ValueError("shape of A and B need to be 2d or 3d")

    if mode == "nn":
      C = self._call_nn(A2, B2)
    elif mode == "tt":
      C = self._call_tt(A2, B2)
    elif mode == "tn":
      C = self._call_tn(A2, B2)
    elif mode == "nt":
      C = self._call_nt(A2, B2)

    if len(A.shape) == 2 and len(B.shape) == 2:
      C = C[0]
    return C