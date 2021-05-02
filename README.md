# Custom Matmul Kernels
This repository contains custom matrix multiplication CUDA kernels and their pytorch wrappers.  
  
## Batched Matrix Multiplication (BMM)
```C = A ⋅ B```  
A : shape = ```[b, m, k]```, dtype = float32  
B : shape = ```[b, k, n]```, dtype = float32  
C : shape = ```[b, m, n]```, dtype = float32  
### Benchmark
#### Square matrices (```m = n = k```)
<p float="left">
  <img src="/imgs/bmm_2.png" width="70%"/>
</p>  

#### Tall and skinny matrices (```k < m = n```)
<p float="left">
  <img src="/imgs/bmm_1.png" width="70%"/>
</p>  

## Masked Batched Matrix Multiplication (MBMM)
``` C = (A ⋅ B) ⊙ M```  
A : shape = ```[b, m, k]```, dtype = float32  
B : shape = ```[b, k, n]```, dtype = float32  
C : shape = ```[b, m, n]```, dtype = float32  
M : shape = ```[b, m, n]```, dtype = boolean  
#### Mask Pattern
<p float="left">
  <img src="/imgs/mask2.png" width="20%"/>
</p>  
  
### Benchmark
#### fixed b, k, varying m, n (m=n)  
<p float="left">
  <img src="/imgs/mbmm_1.png" width="70%"/>
</p>  

#### fixed m, n, k, varying b  
<p float="left">
  <img src="/imgs/mbmm_2.png" width="70%"/>
</p>  
<p float="left">
  <img src="/imgs/mbmm_3.png" width="70%"/>
</p>  

#### fixed b, m, n, varying k  
<p float="left">
  <img src="/imgs/mbmm_4.png" width="70%"/>
</p>  
