---
layout:     post
title:      How gradient is back propagated
subtitle:   Train a model
date:       2020-07-30
author:     Lihao Wang
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Machine Learning
---



In modern DL frameworks like Tensorflow and Pytorch, the backpropagation is computed automatically once the forward propagation is build. Nevertheless, it's always a good thing to understand how the gradients are "propagated" layer by layer during backpropagation.

I have taken a 2-layer NN as a simple example to show how the formula is derived.

<img src="https://i.postimg.cc/BZ8Tf8pZ/Deepin-Capture-cran-zone-de-s-lection-20200730114902.png" style="width:600px;">
