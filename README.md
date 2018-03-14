[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Project : Deep Learning Follow Me

## Writeup by Muthanna A. Attyah
## Mar 2018

In this project we will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

<p align="center"> <img src="./docs/misc/simulator.png"> </p>



## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points

This means all network architecture should be explained, parameters should be explicitly stated with factual justifications, and plots / graphs are used where possible to further enhance understanding.

The student clearly explains each layer of the network architecture and the role that it plays in the overall network. The student can demonstrate the benefits and/or drawbacks of different network architectures pertaining to this project and can justify the current network with factual data. Any choice of configurable parameters should also be explained in the network architecture.

The student shall also provide a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.

The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.
The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.

The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.


# Software & Hardware used for training:

In order to get the best learning from the lab I have decided to use my own GPU enabled hardware to do the learning instead of using AWS Udacity ready made image.

I have used the following hardware:

* Lenovo Yoga 520 laptop (i7-8550U CPU @ 1.8GHz 16GB Memory)
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

# Data Recording

I did not record any data from simulator, I was able to do all required steps using the provided Training, Validation, and Sample Evaluation Data.

# Fully Convolutional Network (FCN) Layers

## Separable Convolutions Layer


```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

## Regular Convolution Layer

```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

## Code:



# Neural Network Hyper Parameters

* **batch_size:** number of training samples/images that get propagated through the network in a single pass. When I used batch size of 64, 50, 40 and 32 I was getting "**ResourceExhaustedError : OOM when allocating tensor with shape..**" error. I was able to resolve it after reducing the batch size down to **20**. Below screen shot showing the utilization details of the GPU using this batch size:
<p align="center"> <img src="./docs/misc/nvidia-smi.png"> </p>

* **workers:** maximum number of processes to spin up. I used 8 workers to fully utilize the power of my Intel i7 processor cores. Below are the actual utilization graphs when no work is done and when training:

<p align="center"> <img src="./docs/misc/cpu_0_workers.png"> </p>
<p align="center"> <img src="./docs/misc/cpu_8_workers.png"> </p>

My plan to figureout the rest of parameters was mostly based on brute force; having my own GPU enabled tensorflow machine helped me alot in doing a good number of brute force runs as needed. Below are the captures of each attempt along with the related training curve:

* **Learning Rate:** 

* **num_epochs:** number of times the entire training dataset gets propagated through the network.

* **steps_per_epoch:** number of batches of training images that go through the network in 1 epoch. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.

* **validation_steps:** number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset.

  
## Paramter Set 1:
| **Parameter** | **Value** |
|:--|:--:|
| learning_rate | 0.01 |
| batch_size | 20 |
| num_epochs | 10 |
| steps_per_epoch | 100 |
| validation_steps | 50 |
| workers | 8 |
| final_score | **20%** |
<p align="center"> <img src="./docs/misc/train_curve_1.png"> </p>

## Paramter Set 2:
| **Parameter** | **Value** |
|:--|:--:|
| learning_rate | 0.01 |
| batch_size | 20 |
| num_epochs | 15 |
| steps_per_epoch | 100 |
| validation_steps | 50 |
| workers | 8 |
| final_score | **40%** |
<p align="center"> <img src="./docs/misc/train_curve_2.png"> </p>




# Trained TensorFlow model

The file is in the correct format (.h5) and runs without errors.

# Neural Network Accuracy

The neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric.

# Future Enhancements

* Recording a bigger dataset capturing more angles/distances of the target will help in further improving the network accuracy.

* Parameters
