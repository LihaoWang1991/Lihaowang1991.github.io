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

As NN convention, the superscript of each variable represents its layer order and sample order and the subscript repsents its node order. For example, the second node in layer 1 (hidden layer) of sample 3 is noted as a<sub>2</sub><sup>\[1]\(3)</sup>.  Let's take *m* as the total number of samples. 


![](https://latex.codecogs.com/gif.latex?%5Cinline%20%5CLARGE%20%5Cbegin%7Bbmatrix%7D%20a_%7B1%7D%5E%7B%5B2%5D%281%29%7D%20%26%20a_%7B1%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20a_%7B1%7D%5E%7B%5B2%5D%28m%29%7D%5C%5C%20%26%20%26%20%26%20%5C%5C%20a_%7B2%7D%5E%7B%5B2%5D%281%29%7D%20%26%20a_%7B2%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20a_%7B2%7D%5E%7B%5B2%5D%28m%29%7D%20%5C%5C%20%26%20%26%20%26%20%5C%5C%20a_%7B3%7D%5E%7B%5B2%5D%281%29%7D%20%26%20a_%7B3%7D%5E%7B%5B2%5D%282%29%7D%20%26%20...%20%26%20a_%7B3%7D%5E%7B%5B2%5D%28m%29%7D%20%5Cend%7Bbmatrix%7D)


The loss funciton is defined as: 
