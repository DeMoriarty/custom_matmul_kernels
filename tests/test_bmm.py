import torch
import numpy as np
from time import time
from .bmm import BMM

def test_bmm(l, m, n, k, mode="nn", n_iter=1):
  """
  compares torch.bmm and BMM
    C = A @ B
    where:
      A: shape = [l, m, k], dtype = float32
      B: shape = [l, k, n], dtype = float32
      C: shape = [l, m, n], dtype = float32
      @ means batch dot product
    mode: {"nn", "tt, "nt", "tn"}, default : "nn"
    n_iter: time cost is averaged over n_iter runs
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
  custom_bmm = BMM()

  tm = time()
  for i in range(n_iter):
    C1 = custom_bmm(A, B, mode=mode)
    torch.cuda.synchronize()
  time_cost_1 = (time() - tm) / n_iter
  print("time spent for custom_bmm:", time_cost_1)
  
  del C1
  if mode[0] == "t":
    A = A.transpose(1, 2)
  if mode[1] == "t":
    B = B.transpose(1, 2)
  tm = time()
  for i in range(n_iter):
    C2 = torch.bmm(A, B)
    torch.cuda.synchronize()
  time_cost_2 = (time() - tm) / n_iter
  print("time spent for torch.bmm:", time_cost_2)
  error = (C1 - C2).abs().sum()
  print("Error:", error)
  return time_cost_1, time_cost_2