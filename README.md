[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Project : Deep Learning Follow Me

## Writeup by Muthanna A. Attyah
## Mar 2018

In this project we will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

<p align="center"> <img src="./misc/simulator.png"> </p>



## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points


Specifications are met if a reader would be able to replicate what you have done based on what was submitted in the report. 

This means all network architecture should be explained, parameters should be explicitly stated with factual justifications, and plots / graphs are used where possible to further enhance understanding.


The student clearly explains each layer of the network architecture and the role that it plays in the overall network. The student can demonstrate the benefits and/or drawbacks of different network architectures pertaining to this project and can justify the current network with factual data. Any choice of configurable parameters should also be explained in the network architecture.

The student shall also provide a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.

# Software & Hardware used for training:

In order to get the best learning from the lab I have decided to use my own GPU enabled hardware to do the learning instead of using AWS Udacity ready made image.

I have used the following hardware:

* Lenovo Yoga 520 laptop (i7-8550U CPU @ 1.8GHz 15GB Memory)
* NVIDIA GEFORCE 940MX (2GB Memory 384 CUDA cores)

And the following OS/Drivers:

* Ubuntu 16.04 LTS
* NVIDIA Drivers 390.30
* NVIDIA CUDA 8.0
* NVIDIA cuDNN 5.1

And the following frameworks and packages:

* Python 3.x
* Tensorflow GPU 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

It took me very long time to figure out the right version of CUDA/cuDNN that will match the selected version of TensorFlow-GPU then later I found this link which clearly states the required CUDA/cuDNN versions:

https://www.tensorflow.org/install/install_sources

When I used batch size of 64 and 32 I was getting "ResourceExhaustedError : OOM when allocating tensor with shape.." error. I was able to resolve it by reducing the batch size to 16. Below screen shot showing the utilization details of the GPU:
<p align="center"> <img src="./misc/nvidia-smi.png"> </p>

# Neural Network Hyper Parameters

| **Parameter** | **Value** |
|:--|:--:|
| learning_rate | 0.01 |
| batch_size | 30 |
| num_epochs | 15 |
| steps_per_epoch | 50 |
| validation_steps | 50 |
| workers | 2 |



The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

Epoch
Learning Rate
Batch Size
Etc.

All configurable parameters should be explicitly stated and justified.

The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.

The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.

The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.

# Trained TensorFlow model

The file is in the correct format (.h5) and runs without errors.

# Neural Network Accuracy

The neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric.

# Future Enhancements

A discussion on potential improvements to the project submission should also be included for future enhancements to the network / parameters that could be used to increase accuracy, efficiency, etc. It is not required to make such enhancements, but these enhancements should be explicitly stated in its own section titled "Future Enhancements".
