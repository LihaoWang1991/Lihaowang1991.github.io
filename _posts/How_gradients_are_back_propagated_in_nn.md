---
layout:     post
title:      How gradients are back propagated in NN
date:       2020-07-30
author:     Lihao Wang
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Machine Learning
---



In modern DL frameworks like Tensorflow and Pytorch, the backpropagation is computed automatically once the forward propagation is build. Nevertheless, it's always a good thing to understand how the gradients are "propagated" layer by layer during backpropagation.

I have taken a 2-layer NN as a simple example to show how the formula is derived.

<img src="https://i.postimg.cc/Xv10F1Tb/Deepin-Capture-cran-zone-de-s-lection-20200730120631.png" style="width:400px;">

![](https://latex.codecogs.com/gif.latex?\\\begin{bmatrix}&space;a_{1}^{[2](1)}&space;&&space;a_{1}^{[2](2)}&space;&&space;...&space;&&space;a_{1}^{[2](m)}\\&space;a_{2}^{[2](1)}&space;&&space;a_{2}^{[2](2)}&space;&&space;...&space;&&space;a_{2}^{[2](m)}&space;\\&space;a_{3}^{[2](1)}&space;&&space;a_{3}^{[2](2)}&space;&&space;...&space;&&space;a_{3}^{[2](m)}&space;\end{bmatrix})


![](https://latex.codecogs.com/gif.latex?\\\[\begin{bmatrix} a_{1}^{[2](1)} & a_{1}^{[2](2)} & ... & a_{1}^{[2](m)}\\ a_{2}^{[2](1)} & a_{2}^{[2](2)} & ... & a_{2}^{[2](m)} \\ a_{3}^{[2](1)} & a_{3}^{[2](2)} & ... & a_{3}^{[2](m)} \end{bmatrix}\])


![](http://latex.codecogs.com/gif.latex?\\sigma=\sqrt{\frac{1}{n}{\sum_{k=1}^n(x_i-\bar{x})^2}})
