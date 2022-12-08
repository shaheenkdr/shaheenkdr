---
layout: post
title: "How to install Kaldi"
tags: SpeechRecognition DeepLearning Kaldi DNN
comments: true
permalink: how-to-install-kaldi
---
<br>
If you are a huge fan of Deep Learning and was thinking how that can be applied to help machines understand human Speech, ie Automated Speech Recognition (ASR), you are in the right place. This article explains how you can kickstart your voyage to the world of Speech understanding using Deep Learning.

In this post, we'll see how to set [Kaldi](https://kaldi-asr.org) up and running.  

If you do not know what Kaldi is, it is primarily a high level versatile and flexible toolkit for speech recognition written in C++, Python & Bash and is licensed under the Apache License v2.0. The official documentation says Kaldi is primarily intended to be used by speech recognition researchers.

##### Lil History
According to mythology, the name `Kaldi` originates from the Ethiopian goatherder who discovered the coffee plant.

A major part of Kaldi has been developed by the iconic Prof Daniel Povey from Johns Hopkins University. Kaldi came to existence in 2009 with the goal of *Low Development Cost, High Quality Speech Recognition for New Languages and Domains* .
Though most of the code base was initially developed between 2009 - 2011, it was mainly focused towards the HMM - GMM approaches. However, fast forward to this day, where neural networks have completely disrupted the domain of Automated Speech Recognition, Kaldi houses state of the art tools to train DNNs in various architectures with high efficacy & efficiency. 

##### Installation
Alright, its time for some action. However before proceeding any further I'd like to provide some **warnings** for the readers:

* I tried this mainly on an Ubuntu 18.04 LTS system, though other variants of Linux and even Windows is supported by Kaldi, for a smooth run-time experience I'd recommend using an Ubuntu OS which has a long term support. 

* WARNING : If you plan to install it on a normal PC or a laptop, chances are high it will be an overkill for normal systems and can heat and destroy your machines if you train larger projects with it. 

* Hence, this article assumes you'd be installing Kaldi on a server system, or over the cloud (AWS, Azure or GCP) with a server grade processor and server grade GPUs. 

* Also ensure you have enough hard disk space available, a typical project I work with usually takes up 500GB of space around, this is for 2000 hours of data or more. So ensure you have a good hdd with good read and write speed with enough space in it. 

Alright, so here we go :

The first step to installation is to get the source code of the Kaldi project, which can be done by : 

```
git clone https://github.com/kaldi-asr/kaldi.git
```

Once you've cloned the repository `cd` to that directory. You must now see a structure similar to this : 

```
.
├── COPYING
├── docker
├── egs
├── INSTALL
├── misc
├── README.md
├── scripts
├── src
├── tools
└── windows

```

First we'll need to setup the tools required by Kaldi before building the core engine itself, for which navigate to the `tools` directory : `cd tools/`

Which should look like this : 

```

.
├── ATLAS_headers
├── CLAPACK
├── config
├── extras
├── INSTALL
├── install_pfile_utils.sh -> extras/install_pfile_utils.sh
├── install_portaudio.sh -> extras/install_portaudio.sh
├── install_speex.sh -> extras/install_speex.sh
├── install_srilm.sh -> extras/install_srilm.sh
└── Makefile

```

As Kaldi is dependent on many other libraries, we'd need to install them in case if you haven't already, this includes libraries such as `gcc`, `cmake`, `sox`, `ffmpeg`, `python` etc ...

Thankfully Kaldi has an in-built script which checks for which all libraries are missing and provides the user the command to install the missing libraries. To deploy this in action, from the `tools/` directory, run : 

```
./extras/check_dependencies.sh
```

Now if you have all the tools up and running already, the script should give you a green light to proceed, or it would ask you to install the missing items. Once you've installed them, we are good to build the tools from this directory.

In order to do so, run: 

```
make -j 20
```
from the same directory, this should initiate cmake and start the building process. Note the **20** followed by the -j switch. This indicates the system to complete this task by using 20 parallel jobs. Now if your system does not have that many number of cores, you'd want to lower the number. The higher the CPU and cores, you can assign the -j to the max no of jobs so that you can finish it quick. 

Now, if everything was in order, it should've successfully completed the build and should ve given a success message as well. Now we'd move onto the core section. 

Navigate back to the root Kaldi directory and then to the `src/` directory, this installs the core toolkits of Kaldi, including the nnetbin and other critical components. Once you are in the `src/` directory, it should look like this : 

```

.
├── base
├── bin
├── chain
├── chainbin
├── configure
├── cudadecoder
├── cudadecoderbin
├── cudafeat
├── cudafeatbin
├── cudamatrix
├── decoder
├── doc
├── Doxyfile
├── feat
├── featbin
├── fgmmbin
├── fstbin
├── fstext
├── gmm
├── gmmbin
├── gst-plugin
├── hmm
├── INSTALL
├── itf
├── ivector
├── ivectorbin
├── kws
├── kwsbin
├── lat
├── latbin
├── lm
├── lmbin
├── Makefile
├── makefiles
├── matrix
├── nnet
├── nnet2
├── nnet2bin
├── nnet3
├── nnet3bin
├── nnetbin
├── NOTES
├── online
├── online2
├── online2bin
├── onlinebin
├── probe
├── rnnlm
├── rnnlmbin
├── sgmm2
├── sgmm2bin
├── tfrnnlm
├── tfrnnlmbin
├── TODO
├── transform
├── tree
└── util

```

To initiate the installation process, from the `src` directory we'd run :

```
./configure \
--shared \
--use-cuda=yes \
--cudatk-dir=/usr/local/cuda-9.0/ \
--mathlib=MKL \
--mkl-root=/opt/intel/mkl

```

Now pay attention the switches here, the `shared` switch indicates the type of build we want to do, then `use-cuda` tells the system we'd want to use the Nvidia CUDA libraries as we intend to train with GPUs, if you do not have a GPU and CUDA setup properly, this would fail. Followed by which we pass the cuda path using `cudatk-dir`. The next important switch is `mathlib`. Here we set it as MKL, which is a fast math kernel library, that helps to compute FFTs and linear algebra way faster, however this is optimized for intel processors. Because I use a latest intel processor, this would speed up the process a whole lot. Followed by which I pass the `mkl-root` which is the path where the intel MKL is installed. 

Now that the config part is done, we'll build Kaldi. To do so, run :

```
make depend -j 20
```

This should generate the dependencies of the the C/CPP source files. Followed by which run :

```
make -j 20
```

This should build Kaldi, It takes some time to complete depending upon the specs of your system. Once its done if you get a success message, Congrats on your completion of the first mission. You've successfully built Kaldi. In next post, we'll see the structure of Kaldi and an overview to its working. Followed by which we'll build some real engines using Kaldi. 

Till then Happy Hacking ◕‿◕
