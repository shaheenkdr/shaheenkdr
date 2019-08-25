---
layout: post
title: "Creating an Indian English ASR"
tags: SpeechRecognition DeepLearning Kaldi DNN
comments: true
permalink: indian-english
---
<br>
<br>

##### Introduction

This project was completed as part of [Google Summer of Code 2019](https://summerofcode.withgoogle.com/projects/#6492594211651584) under the mentorship of RedHen Lab. I'd like to thank Prof Dr TM
Thasleema, Central University Kerala and Karan Singla, PhD graduate student at USC Viterbi school of Engineering for their extended and comprehensive support without which this project wouldn't have been completed. Also absolutely indebted to Prof Mark Turner, Case Western Reserve University, and Prof Francis Steen UCLA, for exceptional support from the community perspective. 

The scope of this project is to build a generic high accuracy Speech Recognition engine using Deep Learning to understand English news and specifically Indian English news. 

##### Datasets used

As Deep learning based Speech Recognition engines require high amount of data to converge, this model was trained with a large volume of data, which includes : 

* NPTEL : Indian Video lecture series, 1000 hours of which was processed and transcribed and was used for training.

* Fisher : Noisy conversational Speech dataset of 1750 hours

* LibriSpeech : 1000 hours read audio book dataset.

* AMI : close mic 100 hour meeting dataset.

* Switchboard : 250 hours noisy conversational dataset. 

* Tedlium : 500 hours of Ted Speaker dataset. 

* 
