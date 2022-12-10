---
layout: post
title: "Get Started with the Wav2letter Toolkit: A Beginner's Guide to Installing and Configuring Wav2letter"
excerpt: Ready to start using the Wav2letter toolkit for speech recognition? This beginner-friendly guide will help you install and configure Wav2letter quickly and easily, with step-by-step instructions and helpful tips for optimizing your setup. Get started now and take your speech recognition skills to the next level with Wav2letter!
tags: SpeechRecognition DeepLearning Wav2letter CNN
comments: true
permalink: how-to-install-wav2letter
---
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
> chmod u+x ArrayFire-no-gl-v3.6.1_Linux_x86_64.sh
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

##### googletest

Now that `ArrayFire` is installed, we’ll now need to install `googletest` . This is a testing framework developed by the Testing & Technology team at `Google`.

In order to install `googletest` follow the guide [here](https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/).

##### OpenMPI

Next up, we’ll need to install `openmpi`, which is a toolkit for High Performance and parallel computing . To install :


```
> sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

```

Now presuming all dependencies are successfully installed, we can now finally build `flashlight`. To do so :


Clone the repository of `flashlight` :

```
> git clone --recursive https://github.com/facebookresearch/flashlight.git
> cd flashlight
> mkdir -p build && cd build
> cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CUDA -DArrayFire_DIR=/opt/arrayfire-no-gl/share/ArrayFire/cmake/
> make -j4  # (or any number of threads)
> make install
```

> For more info on configuring flashlight with different installation flags, see the [DOCS](https://fl.readthedocs.io/en/latest/installation.html)

##### libsndfile

Now that we have `flashlight` configured, we need to install another dependency called `libsndfile`. This is required for loading audio. This library helps in reading and writing files containing sampled audio data. To Install :


```
> apt install autoconf autogen automake build-essential libasound2-dev libflac-dev libogg-dev libtool libvorbis-dev pkg-config python

```

##### Intel MKL

`Wav2letter` uses Intel’s Math Kernel library for featurization purpose. In order to install `mkl` goto this [LINK](https://software.intel.com/en-us/mkl/choose-download/linux) , register and download the binaries. And all you need to do is execute the shell script by adding permission once its downloaded.

Once its installed , export the PATH as :

```
export MKL_INCLUDE_DIR=/opt/intel/mkl/include 
```

You may want to change this if `mkl` is not installed in `/opt` dir by default.

##### FFTW

`fftw` is a `C` subroutine library for computing the discrete Fourier transform (DFT) in one or more dimensions, of arbitrary input size, and of both real and complex data (as well as of even/odd data, i.e. the discrete cosine/sine transforms or DCT/DST). Since `Wav2letter` depends this library for featurization, we’ll need to build that as well. To do so :

```
> sudo apt-get install libfftw3-dev
```

##### KenLM

`kenlm` is a an efficient library that implements two data structures for efficient language model queries, reducing both time and memory costs. The `PROBING` data structure uses linear probing hash tables and is designed for speed. Compared with the widely used `SRILM`, the `PROBING` model in `kenlm` is 2.4 times as fast while using 57% of the memory. The `TRIE` data structure is a trie with bit-level packing, sorted records, interpolation search, and optional quantization aimed at lower memory consumption. `TRIE` simultaneously uses less memory and less CPU. To install `kenlm`:


```
> apt-get install zlibc zlib1g zlib1g-dev libeigen3-dev bzip2 liblzma-dev libboost-all-dev
> wget http://kheafield.com/code/kenlm.tar.gz
> tar -xvzf kenlm
> cd kenlm
> mkdir -p build && cd build
> cmake ..
> make -j 4
```

After installation, do not forget to export the PATH, as :

```
> export KENLM_ROOT_DIR=/home/kenlm2
```

Change the directory values depending up on where you choose to extract and install the library.

##### gflags

`gflags` library implements command-line flags processing. It includes built-in support for standard types such as string and the ability to define flags in the source file in which they are used. `Wav2letter` requires this library, and to install the library :

```
> sudo apt-get install libgflags2 libgflags-dev

```

##### glogs

`glogs` is a `C++` implementation of Google’s logging library, and for logging purposes, `Wav2letter` depends this library. In order to install :

```
> sudo apt install libgoogle-glog-dev
```

##### gtest & gmock

Finally, if you plan to build tests on this, you can install `gtest`. Its OPTIONAL and if you prefer to install, you can install from [HERE](https://github.com/google/googletest).


And finally, if you’ve managed to successfully install all the required dependencies without errors, then we are good to build `Wav2letter`. In order to build:

```
> git clone --recursive https://github.com/facebookresearch/wav2letter.git
> cd wav2letter
> mkdir -p build && cd build
> cmake .. -DCMAKE_BUILD_TYPE=Release -DCRITERION_BACKEND=CUDA -DArrayFire_DIR=/opt/arrayfire-no-gl/share/ArrayFire/cmake/ -DMKL_INCLUDE_DIR=/opt/intel/mkl/include -DBUILD_TESTS=OFF
```

If everything passes successfully, congrats and you are good to proceed with training neural networks using `Wav2letter`. Happy hacking & good luck :)

<div class="reachout">
 
PS : Thanks for reading this blogpost! If you have any questions, problem statement, or requirements
Please dont hesitate to reach out to me. I offer free AI consultations to help you explore how AI and
Machine learning can be used to solve your business challenges.
I am available at : shaheenkdr [@] gmail [dot] com
</div>
