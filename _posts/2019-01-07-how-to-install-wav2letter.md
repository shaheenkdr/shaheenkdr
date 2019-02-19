---
layout: post
title: "How to install Wav2letter"
tags: SpeechRecognition DeepLearning Wav2letter CNN
comments: true
permalink: how-to-install-wav2letter
---
<br>
<br>
The year 2018 was fabulous altogether. It got an even exhilarating finish, as [FAIR](https://research.fb.com/category/facebook-ai-research/) (Facebook AI Research Team) by surprise released their end-to-end deep learning toolkit for Speech recognition.

According to FAIR, Wav2letter follows a completely convolutional approach, and uses Convolutional Neural Networks(CNN) for acoustic modelling as well as language modelling. What’s even more interesting here is, beginning from the Waveform to the end word transcription, the architecture of Wav2letter is composed only of convolutional layers. In a domain where recurrent architectures are more prevalent, this is quite exciting to see CNNs producing results that are competitive with recurrent architectures.

Also, to help everyone get a concise & precise overview of how Wav2letter works, the FAIR team has also made the architecture of Wav2letter public:
<br>
<br>
![Wav2letter](https://raw.githubusercontent.com/shaheenkdr/shaheenkdr.github.io/master/images/arch.png
 "Architecture of Wav2letter")
 <br>
 <br>
 What follows in this article is an installation guide for newbies. If you are coming from a Pythonic background or as a newbie to deep learning with sequential data, setting up Wav2letter can be a bit intimidating task. So the scope of this article is to simplify that. Let’s get started !
 
 ##### Before getting started :

In order to build `Wav2letter`, we need to make sure we have a good `C++` compiler installed with `C++11` support(preferably `g++ > 4.8` ). Also we require `CMake`(version 3.5.1 or later), a toolkit used to control the software compilation process using simple platform and compiler independent configuration files. If you do not have that already installed, install it by :

```> sudo apt-get install cmake g++
```


