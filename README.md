# Custom Matmul Kernels
This repository contains custom matrix multiplication CUDA kernels and their pytorch wrappers.  
  
## Things you can do with a custom matmul kernel
- [Masked Batch Matrix Multiplication (MBMM)](#masked-batch-matrix-multiplication-mbmm)
- [Fused Reduce & BMM](#fused-reduce--bmm)
- [Sampled Softmax](#sampled-softmax)

## Batch Matrix Multiplication (BMM)
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

**Takeaway**: custom kernel is about 25 ~ 50% slower than the cuBLAS SGEMM pytorch uses.  

## Masked Batch Matrix Multiplication (MBMM)
``` C = (A ⋅ B) ⊙ M```  
A : shape = ```[b, m, k]```, dtype = float32  
B : shape = ```[b, k, n]```, dtype = float32  
C : shape = ```[b, m, n]```, dtype = float32  
M : shape = ```[b, m, n]```, dtype = boolean  

This is commonly used in Transformer decoder multi-head
self attention modules, the difference is masked regions are filled with -∞ instead of 0:  
``` python
# ...
# shape of keys : [h, n, d]
# shape of queries: [h, d, n]
# shape of mask: [n, n]
dot = torch.bmm(keys, queries)
dot = dot.masked_fill_(mask=mask, value = float("-inf"))
# ...
```
If we fuse ```masked_fill``` into the matmul kernel, we not only can get rid of ```masked_fill```, but the matmul kernel itself can make use of the sparcity given by the mask (by not computing the masked part), reducing computation even further.  

The BMM kernel splits the output matrix **C** into grids of 128x128 submatrices, each submatrix is assigned to a **thread block**. Each thread block consists of 256 **threads**, each thread computes a 8x8 subsubmatrix of the 128x128 submatrix.  

Now we can try to fuse the ```masked_fill``` into matmul kernel. First we split the mask into grid of 128x128 blocks, and get sum of each block. if the sum of any block is equal to 0, means that block is completely masked, we will skip this thread block entirely inside the kernel (no memory read/write, no math ops). Then we split mask into 8x8 blocks, get sum, then skip the computation of threads where sum equals to 0 (no math ops, no memory write, stil need memory read).

This is similar to the idea of [Block Sparse](https://github.com/openai/blocksparse). The difference is we didn't design a whole new kernel for it, we just modified the existing matmul kernel.

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

We can see that MBMM is nearly 100% faster than the custom BMM + masked_fill, and around 5 ~ 10% faster than torch.bmm + masked_fill. The mask used above has about 50% sparsity. It gets faster when the mask is even more sparse.  

**Takeaway**: custom MBMM kernel is slightly faster than torch.bmm + masked_fill for causal masking.

## Fused Reduce & BMM
Sometimes we want to apply reduction right after a matrix multiplication:
```
C = torch.bmm(A, B)
maxC, argmaxC = torch.max(C, dim=1)
sumC = torch.sum(C, dim=2)
argminC = torch.argmin(C, dim=1)
```
Normally there isn't a problem with doing this, however, when optimizing my implementation of K-means clustering algorithm, this become an issue. One of the main steps of K-means algorithm is computing the distance between every data point and every centroid (cluster center), and get index of the closest cluster for each data point (argmin).  

When the number of data point (n_data) and the number of clusters (n_clusters) are very large, this step will produce a huge (n_data x n_clusters) matrix that might not fit into GPU memory (imagine a 1,000,000 x 10,000 fp32 matrix)   

One workaround is to split this step into multiple tiny steps: in each step only compute the distance between a subset of data points and centroids, (let's say 10,000 x 10,000), asign each data point to the closest cluster, and repeat.  

A better way is to **fuse argmin into the matmul kernel**. Advantages of doing this are:  
1. No need to create huge matrices if the result of argmin is all we care about  
2. No need for loop  
3. Possibily faster (less memory ops).  

The K-Means implementation is [here](https://github.com/DeMoriarty/TorchPQ/tree/main/torchpq/kmeans), 
And [this](https://github.com/DeMoriarty/TorchPQ/blob/main/torchpq/kmeans/kernels/MaxSimKernel.cu) is the fused kernel. Maybe I will put them on this repo later.  

## Sampled Softmax
Usually, on the last layer of neural network language models, the hidden representation of each input token is "compared" to a large number of "output embeddings", or in other words, we compute the dot product of a matrix with shape `[batch_size, hidden_dim]` and another matrix with shape `[hidden_dim, vocab_size]`, the resulting matrix (logits) is then passed into a cross entropy loss function during training, or a softmax function during inference.  

However, when `vocab_size` is large, the process described above gets very expensive. **Sampled softmax** is a method that is proposed to reduce the computation cost of exhaustive softmax. The basic idea is, for each training example, we pick a subset of the output embeddings according to some sampling function and use it to estimate full softmax, this way we only need to compute dot product `(batch_size x hidden_dim) ⋅ (hidden_dim x sample_size)`, where `sample_size` is far smaller than `vocab_size`. I won't go into too much details, you can read this [paper](https://arxiv.org/abs/1412.2007), this [blog post](https://ruder.io/word-embeddings-softmax/), or this [document by tensorflow](https://www.tensorflow.org/extras/candidate_sampling.pdf) if you are interested.  

This method is implemented in [tensorflow](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss),
however, it only performs the sampling for each minibatch, hidden vectors of all tokens inside the minibatch is multiplied to the same set of output embeddings. 

By modifying matmul kernel, we can **multiply the hidden vector of each token in the minibatch to a unique subset of output embeddings**

to be continued...
