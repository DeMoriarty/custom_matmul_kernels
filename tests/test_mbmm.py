import torch
import numpy as np
from time import time
from .mbmm import MBMM

def test_mbmm(l, m, n, k, mode="nn", n_iter=1, share_mask=True):
  """
    compares torch.bmm and MBMM
    C = (A @ B) * M
    where:
      A: shape = [l, m, k], dtype = float32
      B: shape = [l, k, n], dtype = float32
      C: shape = [l, m, n], dtype = float32
      M: shape = [l, m, m], dtype = uint8
      @ means batch dot product
      * means hadamart product (elementwise product)
    mode: {"nn", "tt, "nt", "tn"}, default : "nn"
    n_iter: time cost is averaged over n_iter runs
    share_mask:
      if True, all [m, n] matrices in batch shares same mask
      if False, all matrices have unique masks
  """
  print(f"l={l}  m={m}  n={n}  k={k}")
  if mode[0] == "n":
    A = torch.randn(l, m, k, device="cuda:0")
  elif mode[0] == "t":
    A = torch.randn(l, k, m, device="cuda:0")
  
  if mode[1] == "n":
    B = torch.randn(l, k, n, device="cuda:0")
  elif mode[1] == "t":
    B = torch.randn(l, n, k, device="cuda:0")
  custom_mbmm = MBMM(write_float8=True, share_mask=share_mask)

  if share_mask:
    final_mask = torch.ones(m, n, device="cuda")
    final_mask = torch.tril(final_mask).to("cuda").bool()
    thread_mask = final_mask.view(math.ceil(m/8), 8, math.ceil(n/8), 8)
    thread_mask = thread_mask.sum(dim=1).sum(dim=-1)
    block_mask = final_mask.view(math.ceil(m/128), 128, math.ceil(n/128), 128)
    block_mask = block_mask.sum(dim=1).sum(dim=-1).bool()
  else:
    final_mask = torch.ones(l, m, n, device="cuda")
    final_mask = torch.tril(final_mask).to("cuda").bool()
    thread_mask = final_mask.view(l, math.ceil(m/8), 8, math.ceil(n/8), 8)
    thread_mask = thread_mask.sum(dim=2).sum(dim=-1)
    block_mask = final_mask.view(l, math.ceil(m/128), 128, math.ceil(n/128), 128)
    block_mask = block_mask.sum(dim=2).sum(dim=-1).bool()

  thread_mask = thread_mask.to(torch.uint8)
  block_mask = block_mask.to(torch.uint8)
  element_mask = final_mask.to(torch.uint8)
  ###
  batch_index = 0

  mask = ~final_mask.bool()
  del final_mask
  torch.cuda.synchronize()

  tm = time()
  for i in range(n_iter):
    C1 = custom_mbmm(A, B, block_mask, thread_mask, element_mask, mode=mode)
    torch.cuda.synchronize()
  time_cost_1 = (time() - tm) / n_iter
  print("time spent for custom_mbmm:", time_cost_1)
  
  del C1
  if mode[0] == "t":
    A = A.transpose(1, 2)
  if mode[1] == "t":
    B = B.transpose(1, 2)

  tm = time()
  for i in range(n_iter):
    C2 = torch.bmm(A, B)
    C2.masked_fill_(mask = mask, value=0)
    torch.cuda.synchronize()
  time_cost_2 = (time() - tm) / n_iter
  print("time spent for torch.bmm:", time_cost_2)
  
  error = (C1 - C2).abs().sum()
  print("Error:", error)
  return time_cost_1, time_cost_2
