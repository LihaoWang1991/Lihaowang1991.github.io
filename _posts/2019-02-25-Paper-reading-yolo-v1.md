---
layout:     post
title:      Simple Understanding of YOLO V1
date:       2019-02-25
author:     Lihao Wang
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Machine Learning
    - Computer Vision
---

In general, modern NN-based object detection algorithms can be classified into 2 types: one-stage methods like YOLO and two stage methods like Fast R-CNN. 

**Two-stage methods:** They propose at first some candidate regions (stage 1), and then output classfication and bounding box information on each proposed region (stage 2). This kind of method usually has better accuracy compared to one-stage methods however they are relatively slower.

**One-stage methods:** This group of methods predict diretly all the information such as classification and bounding box position as well as their size. Sometimes they are also called single-shot detectors because they take only one shot to output the detection.

YOLO is considered as a milestone one-stage method because it attained the comparable performance with Faster R-CNN, the state-of-the-art two-stage method at that time but YOLO was much faster. In fact, up to today YOLO has evolved 4 generations and here is a timeline:

<img src="https://i.postimg.cc/90s4780S/image.jpg" style="width:600px;">

I will show the key ideas of YOLO V1 in this blog. For a detailed comparison of YOLO V1 to V4, please refer to this [blog](https://lihaowang1991.github.io/2019/02/27/Paper-reading-yolo-v1-to-v4/).

YOLO divides the input image into an S × S grid (S is 7 in the experiment of the paper). If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object. Each grid cell predicts B bounding boxes (B is 2 in the experiment of the paper) and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts. One predictor is assigned to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall <sup>\[1]</sup>.

This is a global model of YOLO <sup>\[1]</sup>:

<img src="https://i.postimg.cc/zfsyyRgX/image.jpg" style="width:600px;">

#### Input

The input of YOLO is RGB images of resolution 448 × 448.

#### Network Architecture

The author has provided 2 versions of YOLO: a basic verison and a fast version.

The basic version has 24 convolutional layers followed by 2 fully connected layers as following:

<img src="https://i.postimg.cc/Qtk70sNq/yolo.jpg" style="width:600px;">

The Fast YOLO is designed to push the boundaries of fast object detection. Fast YOLO uses 9 convolutional layers instead of 24 in the basic version. All the other training and testing parameters are the same between YOLO and Fast YOLO.

#### Output

The output of the netwrok is a 7 × 7 × 30 tensor. That is, for each cell of the 7 × 7 grid, a vector of dimension 30 is predicted. The 30 elements in each vector is composed of:

*  B (number of possible bounding boxes, B = 2 in the paper) × 5 predictions of each bounding boxes: x, y, w, h, and confidence,  The (x, y) coordinates represent the center
of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. Finally the confidence prediction represents the IOU between the predicted box and any ground truth box <sup>\[1]</sup>.

*  C (C = 20 in the paper) conditional class probabilities, Pr(Class<sub>*i*</sub>|Object). These probabilities are conditioned on the grid cell containing an object.


#### Loss Function

For me this is the most important part in YOLO. The loss function is composed of 5 parts as following:

<img src="https://i.postimg.cc/C5TBZn9s/image.jpg" style="width:600px;">

where 1<sub>*i*</sub><sup style="margin-left:-5px">obj</sup> denotes if object appears in cell *i*, 1<sub>*ij*</sub><sup style="margin-left:-5px">obj</sup> denotes that the *j*th bounding box predictor in cell *i* is “responsible” for that prediction, and 1<sub>*ij*</sub><sup style="margin-left:-5px">noobj</sup> denotes that the *j*th bounding box predictor in cell *i* is not “responsible” for that prediction. The weight of each part is different: λ<sub>coord</sub> = 5 and λ<sub>noobj</sub> = 0.5.

#### Training Parameters

The network is trained for about 135 epochs on the training and validation data sets from PASCAL VOC 2007 and 2012. A batch size of 64, a momentum of 0.9 and a decay of 0.0005 are used.

The learning rate is special during the training, the schedule is as follows: For the first epochs the learning rate is slowly raised from 10<sup>-3</sup>
to 10<sup>-2</sup>. This is because if starting at a high learning rate the model often diverges due to unstable gradients. The the training continues with 10<sup>-2</sup> for 75 epochs, then 10<sup>-3</sup> for 30 epochs, and finally 10<sup>-4</sup> for 30 epochs.

To avoid overfitting dropout and extensive data augmentation are also applied.

#### Performance

As the following image shows, Fast YOLO was the fastest extant object detector at that time. With 52.7% mAP, it is more than twice as accurate as prior work on real-time detection. YOLO's perfromance (mAP 63.4%) is not very far from that of Faster R-CNN (mAP 73.2%) but YOLO still maintain real-time performance.

<img src="https://i.postimg.cc/dtxVxyVQ/image.jpg" style="width:400px;">



**Reference:**

1]: [YOLO V1 paper](https://arxiv.org/pdf/1506.02640.pdf)

