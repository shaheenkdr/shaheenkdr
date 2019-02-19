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
 
##### Before getting started 

* Make sure you have a stable Linux system, preferably a stable and latest version. This guide assumes that you are an Ubuntu user, as this was tested on Ubuntu 16.04 LTS as well as Ubuntu 18.04

* If you want to build by leveraging the performance of GPU, make sure you have the latest version of CUDA (preferably 9.2) ,Nvidia NCCL library and CUDNN (preferably 7.2.1) properly installed and configured.

* Its recommended to setup Wav2letter inside a virtual environment (Anaconda) inside of a container (docker), so even if you mess up, the depth of havoc and destruction can be minimised.

##### So Let’s get started !



In order to build `Wav2letter`, we need to make sure we have a good `C++` compiler installed with `C++11` support(preferably `g++ > 4.8` ). Also we require `CMake`(version 3.5.1 or later), a toolkit used to control the software compilation process using simple platform and compiler independent configuration files. If you do not have that already installed, install it by :

```
> sudo apt-get install cmake g++
```

##### Flashlight 

Up next, we need to build `flashlight` as `wav2letter` uses it as a dependency, `flashlight` is a fast, flexible machine learning library written entirely in `C++` from the Facebook AI Research Speech team and the creators of Torch and Deep Speech.

In order to build `flashlight`, up first we’ll need to satisfy few of its dependencies. First, we’ll need to install `Arrayfire`.


##### ArrayFire

`ArrayFire` is a general-purpose library that simplifies the process of developing software that targets parallel and massively-parallel architectures including CPUs, GPUs, and other hardware acceleration devices.

To install `Arrayfire`, we’ll get its binary, `flashlight` has been tested with the `3.6.1` version of `Arrayfire`, with no-gl support, ie the lightweight version without graphic support. To setup:

```
> wget https://arrayfire.s3.amazonaws.com/3.6.1/ArrayFire-no-gl-v3.6.1_Linux_x86_64.sh
```

Once the binaries are downloaded, add permission to execute:

```
>> chmod u+x ArrayFire-no-gl-v3.6.1_Linux_x86_64.sh
```

Then, execute the installer :

```
> bash ArrayFire-no-gl-v3.6.1_Linux_x86_64.sh --include-subdir --prefix=/opt
```

Once its done, set PATH as :

```
> echo /opt/arrayfire-no-gl/lib > /etc/ld.so.conf.d/arrayfire.conf
> ldconfig
```




