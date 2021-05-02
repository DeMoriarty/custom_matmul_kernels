# Custom Matmul Kernels
This repository contains custom matrix multiplication CUDA kernels and their pytorch wrappers.  
  
## Batched Matrix Multiplication (BMM)
```C = A * B```  
A : shape = ```[b, m, k]```, dtype = float32  
B : shape = ```[b, k, n]```, dtype = float32  
C : shape = ```[b, m, n]```, dtype = float32  
### Benchmark
#### Square matrices (```m = n = k```)
image here

#### Tall and skinny matrices (```k < m = n```)
image here

## Masked Batched Matrix Multiplication (MBMM)
``` C = (A * B) ⊙ M```
A : shape = ```[b, m, k]```, dtype = float32  
B : shape = ```[b, k, n]```, dtype = float32  
C : shape = ```[b, m, n]```, dtype = float32  
M : shape = ```[b, m, n]```, dtype = boolean  
#### Mask Pattern
### Benchmark
#### b, k fixed
#### m, n, k fixed
#### b, m, n fixed
