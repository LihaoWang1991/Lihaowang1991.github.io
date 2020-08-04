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

As NN convention, the superscript of each variable represents its layer order as well as sample order and the subscript represents its node order. For example, the second node in layer 1 (hidden layer) of sample 3 is noted as a<sub>2</sub><sup>\[1]\(3)</sup>. And the total number of samples is noted as *m*. 

So the NN vectorized output matrix A<sup>\[2]</sup> is:

![](https://latex.codecogs.com/svg.latex?%5Clarge%20%5Cwidehat%7BY%7D%20%3D%20A%5E%7B%5B2%5D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20a_%7B1%7D%5E%7B%5B2%5D%281%29%7D%20%26%20a_%7B1%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20a_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%5C%5C%20%26%20%26%20%26%20%5C%5C%20a_%7B2%7D%5E%7B%5B2%5D%281%29%7D%20%26%20a_%7B2%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20a_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%20%5C%5C%20%26%20%26%20%26%20%5C%5C%20a_%7B3%7D%5E%7B%5B2%5D%281%29%7D%20%26%20a_%7B3%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20a_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%20%5Cend%7Bbmatrix%7D)

And we define the loss function as: 

![](https://latex.codecogs.com/svg.latex?%5Clarge%20J%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5C%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Csum_%7Bj%3D1%7D%5E%7Bn%7DL%28a_%7Bj%7D%5E%7B%5B2%5D%28i%29%7D%2C%20y_%7Bj%7D%5E%7B%28i%29%7D%29)

Now we can begin to calculate the back propogated gradients. In the rest of this article, for any variable x, whether it's a scalar or matrix, we simplify the partial derivative of loss function J w.r.t x (namely, dJ/dx) as dx.

First of all, let's figure out that of the latest group of variables A<sup>\[2]</sup>, it's just the derivative of each matrix element:

![](https://latex.codecogs.com/svg.latex?dA%5E%7B%5B2%5D%7D%20%3D%20%5Cfrac%7BdJ%7D%7BdA%5E%7B%5B2%5D%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20da_%7B1%7D%5E%7B%5B2%5D%281%29%7D%5C%20%26%20da_%7B1%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20da_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%5C%5C%20%26%20%26%20%26%20%5C%5C%20da_%7B2%7D%5E%7B%5B2%5D%281%29%7D%5C%20%26%20da_%7B2%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20da_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%5C%5C%20%26%20%26%20%26%20%5C%5C%20da_%7B3%7D%5E%7B%5B2%5D%281%29%7D%5C%20%26%20da_%7B3%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20da_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%5C%5C%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7BdL%28a_%7B1%7D%5E%7B%5B2%5D%281%29%7D%2C%20y_%7B1%7D%5E%7B%281%29%7D%29%7D%7Bda_%7B1%7D%5E%7B%5B2%5D%281%29%7D%7D%5C%20%26%20%5Cfrac%7BdL%28a_%7B1%7D%5E%7B%5B2%5D%282%29%7D%2C%20y_%7B1%7D%5E%7B%282%29%7D%29%7D%7Bda_%7B1%7D%5E%7B%5B2%5D%282%29%7D%7D%20%26%20...%20%26%20%5Cfrac%7BdL%28a_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%2C%20y_%7B1%7D%5E%7B%28m%29%7D%29%7D%7Bda_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%7D%5C%5C%20%26%20%26%20%26%20%5C%5C%20%5Cfrac%7BdL%28a_%7B2%7D%5E%7B%5B2%5D%281%29%7D%2C%20y_%7B2%7D%5E%7B%281%29%7D%29%7D%7Bda_%7B2%7D%5E%7B%5B2%5D%281%29%7D%7D%5C%20%26%20%5Cfrac%7BdL%28a_%7B2%7D%5E%7B%5B2%5D%282%29%7D%2C%20y_%7B2%7D%5E%7B%282%29%7D%29%7D%7Bda_%7B2%7D%5E%7B%5B2%5D%282%29%7D%7D%20%26%20...%20%26%20%5Cfrac%7BdL%28a_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%2C%20y_%7B2%7D%5E%7B%28m%29%7D%29%7D%7Bda_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%7D%5C%5C%20%26%20%26%20%26%20%5C%5C%20%5Cfrac%7BdL%28a_%7B3%7D%5E%7B%5B2%5D%281%29%7D%2C%20y_%7B3%7D%5E%7B%281%29%7D%29%7D%7Bda_%7B3%7D%5E%7B%5B2%5D%281%29%7D%7D%5C%20%26%20%5Cfrac%7BdL%28a_%7B3%7D%5E%7B%5B2%5D%282%29%7D%2C%20y_%7B3%7D%5E%7B%282%29%7D%29%7D%7Bda_%7B3%7D%5E%7B%5B2%5D%282%29%7D%7D%20%26%20...%20%26%20%5Cfrac%7BdL%28a_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%2C%20y_%7B3%7D%5E%7B%28m%29%7D%29%7D%7Bda_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%7D%5C%5C%20%5Cend%7Bbmatrix%7D)

Then for Z<sup>\[2]</sup>, I put here the forward propagation from Z<sup>\[2]</sup> to A<sup>\[2]</sup>:

![](https://latex.codecogs.com/svg.latex?%5Clarge%20A%5E%7B%5B2%5D%7D%20%3Dg%5E%7B%5B2%5D%7D%28Z%5E%7B%5B2%5D%7D%29%20%3D%20%5Cbegin%7Bbmatrix%7D%20g%5E%7B%5B2%5D%7D%28z_%7B1%7D%5E%7B%5B2%5D%281%29%7D%29%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B1%7D%5E%7B%5B2%5D%282%29%7D%29%20%26%20...%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%29%5C%5C%20%26%20%26%20%26%20%5C%5C%20g%5E%7B%5B2%5D%7D%28z_%7B2%7D%5E%7B%5B2%5D%281%29%7D%29%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B2%7D%5E%7B%5B2%5D%282%29%7D%29%20%26%20...%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%29%20%5C%5C%20%26%20%26%20%26%20%5C%5C%20g%5E%7B%5B2%5D%7D%28z_%7B3%7D%5E%7B%5B2%5D%281%29%7D%29%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B3%7D%5E%7B%5B2%5D%282%29%7D%29%20%26%20...%20%26%20g%5E%7B%5B2%5D%7D%28z_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%29%20%5Cend%7Bbmatrix%7D)

We find that each *a<sub>i</sub><sup>\[2]\(j)</sup>* depends only on its corresponding *z<sub>i</sub><sup>\[2]\(j)</sup>* before activation function, so we can simply back propagate the gradient by an element-wise production (which is noted as * to differentiate from matrix production ·):

![](https://latex.codecogs.com/svg.latex?dZ%5E%7B%5B2%5D%7D%20%3D%20dA%5E%7B%5B2%5D%7D%5C%3B*%5C%3B%7Bg%5E%7B%5B2%5D%7D%7D%27%28Z%5E%7B%5B2%5D%7D%29%20%3D%20%5Cbegin%7Bbmatrix%7D%20da_%7B1%7D%5E%7B%5B2%5D%281%29%7D%5Ccdot%20%7Bg%5E%7B%5B2%5D%7D%7D%27%28z_%7B1%7D%5E%7B%5B2%5D%281%29%7D%29%20%26%20da_%7B1%7D%5E%7B%5B2%5D%282%29%7D%5Ccdot%20%7Bg%5E%7B%5B2%5D%7D%7D%27%28z_%7B1%7D%5E%7B%5B2%5D%282%29%7D%29%20%26%20...%20%26%20da_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%5Ccdot%20%7Bg%5E%7B%5B2%5D%7D%7D%27%28z_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%29%5C%5C%20%26%20%26%20%26%20%5C%5C%20da_%7B2%7D%5E%7B%5B2%5D%281%29%7D%5Ccdot%20%7Bg%5E%7B%5B2%5D%7D%7D%27%28z_%7B2%7D%5E%7B%5B2%5D%281%29%7D%29%20%26%20da_%7B2%7D%5E%7B%5B2%5D%282%29%7D%5Ccdot%20%7Bg%5E%7B%5B2%5D%7D%7D%27%28z_%7B2%7D%5E%7B%5B2%5D%282%29%7D%29%20%26%20...%20%26%20da_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%5Ccdot%20%7Bg%5E%7B%5B2%5D%7D%7D%27%28z_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%29%20%5C%5C%20%26%20%26%20%26%20%5C%5C%20da_%7B3%7D%5E%7B%5B2%5D%281%29%7D%5Ccdot%20%7Bg%5E%7B%5B2%5D%7D%7D%27%28z_%7B3%7D%5E%7B%5B2%5D%281%29%7D%29%20%26%20da_%7B3%7D%5E%7B%5B2%5D%282%29%7D%5Ccdot%20%7Bg%5E%7B%5B2%5D%7D%7D%27%28z_%7B3%7D%5E%7B%5B2%5D%282%29%7D%29%20%26%20...%20%26%20da_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%5Ccdot%20%7Bg%5E%7B%5B2%5D%7D%7D%27%28z_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%29%20%5Cend%7Bbmatrix%7D)




<!--
(comments) formulas: 
A[2] forward propagation:
A^{[2]} =g^{[2]}(Z^{[2]}) = \begin{bmatrix}
g^{[2]}(z_{1}^{[2](1)}) & g^{[2]}(z_{1}^{[2](2)}) & ... & g^{[2]}(z_{1}^{[2](m)})\\ 
&  &  & \\ 
g^{[2]}(z_{2}^{[2](1)}) & g^{[2]}(z_{2}^{[2](2)}) & ... & g^{[2]}(z_{2}^{[2](m)}) \\ 
 &  &  & \\ 
g^{[2]}(z_{3}^{[2](1)}) & g^{[2]}(z_{3}^{[2](2)}) & ... & g^{[2]}(z_{3}^{[2](m)}) 
\end{bmatrix}

total loss function:
J = \frac{1}{m}\ \sum_{i=1}^{m}\sum_{j=1}^{n}L(a_{j}^{[2](i)}, y_{j}^{(i)})

dA[2] (size 10pts):
dA^{[2]} = \frac{dJ}{dA^{[2]}} = 
\begin{bmatrix}
da_{1}^{[2](1)}\ & da_{1}^{[2](2)} & ... & da_{1}^{[2](m)}\\ 
&  &  & \\ 
da_{2}^{[2](1)}\ & da_{2}^{[2](2)} & ... & da_{2}^{[2](m)}\\
 &  &  & \\ 
da_{3}^{[2](1)}\ & da_{3}^{[2](2)} & ... & da_{3}^{[2](m)}\\
\end{bmatrix}
=
\begin{bmatrix}
\frac{dL(a_{1}^{[2](1)}, y_{1}^{(1)})}{da_{1}^{[2](1)}}\  & \frac{dL(a_{1}^{[2](2)}, y_{1}^{(2)})}{da_{1}^{[2](2)}} & ... & \frac{dL(a_{1}^{[2](m)}, y_{1}^{(m)})}{da_{1}^{[2](m)}}\\ 
&  &  & \\ 
\frac{dL(a_{2}^{[2](1)}, y_{2}^{(1)})}{da_{2}^{[2](1)}}\  & \frac{dL(a_{2}^{[2](2)}, y_{2}^{(2)})}{da_{2}^{[2](2)}} & ... & \frac{dL(a_{2}^{[2](m)}, y_{2}^{(m)})}{da_{2}^{[2](m)}}\\
 &  &  & \\ 
\frac{dL(a_{3}^{[2](1)}, y_{3}^{(1)})}{da_{3}^{[2](1)}}\  & \frac{dL(a_{3}^{[2](2)}, y_{3}^{(2)})}{da_{3}^{[2](2)}} & ... & \frac{dL(a_{3}^{[2](m)}, y_{3}^{(m)})}{da_{3}^{[2](m)}}\\
\end{bmatrix}

dZ[2] (size 10pts):
dZ^{[2]} 
=
dA^{[2]}\;*\;{g^{[2]}}'(Z^{[2]})
=
\begin{bmatrix}
da_{1}^{[2](1)}\cdot {g^{[2]}}'(z_{1}^{[2](1)}) & da_{1}^{[2](2)}\cdot {g^{[2]}}'(z_{1}^{[2](2)}) & ... & da_{1}^{[2](m)}\cdot {g^{[2]}}'(z_{1}^{[2](m)})\\ 
&  &  & \\ 
da_{2}^{[2](1)}\cdot {g^{[2]}}'(z_{2}^{[2](1)}) & da_{2}^{[2](2)}\cdot {g^{[2]}}'(z_{2}^{[2](2)}) & ... & da_{2}^{[2](m)}\cdot {g^{[2]}}'(z_{2}^{[2](m)}) \\ 
 &  &  & \\ 
da_{3}^{[2](1)}\cdot {g^{[2]}}'(z_{3}^{[2](1)}) & da_{3}^{[2](2)}\cdot {g^{[2]}}'(z_{3}^{[2](2)}) & ... & da_{3}^{[2](m)}\cdot {g^{[2]}}'(z_{3}^{[2](m)}) 
\end{bmatrix}

Size and format:
12pts, format svg
-->
