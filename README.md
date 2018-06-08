# CS231N Project: Simultaneous Pose Estimation and Classification

Authors: Eric Cristofalo, Preston Culbertson and Zijian Wang

This reporsitory contains the source code of our project.

## Branch Description

We tried different network architectures. They are organized into different branches. Here are some details about them.

- **posenet**: Our main result, with both modified Faster R-CNN detection pipeline and a pose estimation network for doing pose regression. During training, the weights in the Faster R-CNN part of the network are fixed, and we only perform gradient descent on the pose estimation layers.

- **stereo-pose**: End-to-end joint training on both Faster R-CNN and pose regression network.

- **stereo**: Our modified proposed detection pipeline with stereo images, without the pose estimation network.

- **modeo9**: Based on **stereo** branch, with a more complicated model for the disparity convolution.

## Dependencies

- Python 2.7

- pytorch v3.1.0

- CUDA 8.0

## Disclaimer

Our project uses the open-source project https://github.com/longcw/faster_rcnn_pytorch as the starter code. Substantial changes have been made to implement our model described in our project report.
