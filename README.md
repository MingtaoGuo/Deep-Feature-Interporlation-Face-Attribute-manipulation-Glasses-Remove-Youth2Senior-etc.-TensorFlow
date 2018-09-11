# Deep-Feature-Interporlation-Glasses-Remove-TensorFlow
A simple implementation of the paper 'Deep Feature Interpolation for Image Content Changes'
# Introduction
This code is simple to read, which mainly implement the paper [Deep Feature Interpolation for Image Content Changes](https://arxiv.org/abs/1611.05507). This method mainly address the problem of face attribute manipulation, e.g. glasses remove, change age, mouth close2open, etc.
![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/method.jpg)
This method is simple, but very effective. it is very similar with style transfer.
# How to use
Step 1. Downloading the dataset of face, this address: [Labeled Faces in the Wild(LFW)](http://vis-www.cs.umass.edu/lfw/lfw.tgz). unzip it, and put it into the folder 'lfw'. Pretrained [VGG19](https://pan.baidu.com/s/1YFKdRoB2v9nxScoUG8WRpw) model is needed, download it and put it into the folder 'vgg_para'.

Step 2. Excute the file 'KNN.py', which you can set the target attribute. In the final, you will obtain 100 souce images and 100 target images.

Step 3. Change the input_img's file path in 'DFI.py' and excute it 
# Python packages
=============================

1. python 3.5
2. tensorflow 1.4.0
3. numpy
4. scipy
5. pillow

=============================
# results

|Person|Senior|Mustache|Mouth Open|Smiling|Eye Close|
|-|-|-|-|-|-|
|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/3.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/3_senior.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/3_mustache.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/3_mouthopen.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/3_smiling.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/3_eyeclose.jpg)|
|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/4.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/4_senior.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/4_mustache.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/4_mouthopen.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/4_smiling.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/4_eyeclose.jpg)|
|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/5.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/5_senior.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/5_mustache.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/5_mouthopen.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/5_smiling.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/5_eyeclose.jpg)|
|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/6.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/6_senior.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/6_mustache.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/6_mouthopen.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/6_smiling.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/6_eyeclose.jpg)|
|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/2_0.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/2_senior.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/2_mustache.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/2_mouthopen.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/2_smiling.jpg)|![](https://github.com/MingtaoGuo/Deep-Feature-Interporlation-Glasses-Remove-TensorFlow/blob/master/IMAGES/2_eyeclose.jpg)|
