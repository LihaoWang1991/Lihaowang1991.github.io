---
layout:     post
title:      How gradients are back propagated in NN
date:       2019-01-30
author:     Lihao Wang
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Machine Learning
---



In modern DL frameworks like Tensorflow and Pytorch, the backpropagation is computed automatically once the forward propagation is built. Nevertheless, it's always a good thing to understand how the gradients are "propagated" layer by layer during backpropagation.

I have taken a 2-layer NN as a simple example to show how the formula is derived.

<img src="https://i.postimg.cc/Xv10F1Tb/Deepin-Capture-cran-zone-de-s-lection-20200730120631.png" style="width:400px;">

As NN convention, the superscript of each variable represents its layer order and sample order and the subscript represents its node order. For example, the second node in layer 1 (hidden layer) of sample 3 is noted as a<sub>2</sub><sup>\[1]\(3)</sup>. And the total number of samples is noted as *m*. 

So the NN vectorized output matrix A<sup>\[2]</sup> is as following:


![](https://latex.codecogs.com/svg.latex?%5Clarge%20%5Cwidehat%7BY%7D%20%3D%20A%5E%7B%5B2%5D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20a_%7B1%7D%5E%7B%5B2%5D%281%29%7D%20%26%20a_%7B1%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20a_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%5C%5C%20%26%20%26%20%26%20%5C%5C%20a_%7B2%7D%5E%7B%5B2%5D%281%29%7D%20%26%20a_%7B2%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20a_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%20%5C%5C%20%26%20%26%20%26%20%5C%5C%20a_%7B3%7D%5E%7B%5B2%5D%281%29%7D%20%26%20a_%7B3%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20a_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%20%5Cend%7Bbmatrix%7D)

And we define the loss function as: 

Now we can begin to calculate the back propogated gradients.

First of all, let's figure out that of the latest group of variables: A<sup>\[2]</sup>

Then for Z<sup>\[2]</sup>, I put the forward propagaton from Z<sup>\[2]</sup> to A<sup>\[2]</sup> here:

![](https://latex.codecogs.com/svg.latex?%5Clarge%20A%5E%7B%5B2%5D%7D%20%3Dg%5E%7B%5B2%5D%7D%28Z%5E%7B%5B2%5D%7D%29%20%3D%20%5Cbegin%7Bbmatrix%7D%20g%5E%7B%5B2%5D%7D%28z_%7B1%7D%5E%7B%5B2%5D%281%29%7D%29%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B1%7D%5E%7B%5B2%5D%282%29%7D%29%20%26%20...%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%29%5C%5C%20%26%20%26%20%26%20%5C%5C%20g%5E%7B%5B2%5D%7D%28z_%7B2%7D%5E%7B%5B2%5D%281%29%7D%29%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B2%7D%5E%7B%5B2%5D%282%29%7D%29%20%26%20...%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%29%20%5C%5C%20%26%20%26%20%26%20%5C%5C%20g%5E%7B%5B2%5D%7D%28z_%7B3%7D%5E%7B%5B2%5D%281%29%7D%29%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B3%7D%5E%7B%5B2%5D%282%29%7D%29%20%26%20...%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%29%20%5Cend%7Bbmatrix%7D)

We can see that each *a* depeds only on its *z* before activation function, so we can simply propagate the gradient by an element-wise production:

<!--
(comments) formula template: 
A^{[2]} =g^{[2]}(Z^{[2]}) = \begin{bmatrix}
g^{[2]}(z_{1}^{[2](1)}) & g^{[2]}(z_{1}^{[2](2)}) & ... & g^{[2]}(z_{1}^{[2](m)})\\ 
&  &  & \\ 
g^{[2]}(z_{2}^{[2](1)}) & g^{[2]}(z_{2}^{[2](2)}) & ... & g^{[2]}(z_{2}^{[2](m)}) \\ 
 &  &  & \\ 
g^{[2]}(z_{3}^{[2](1)}) & g^{[2]}(z_{3}^{[2](2)}) & ... & g^{[2]}(z_{3}^{[2](m)}) 
\end{bmatrix}

12pts, format svg
-->
