# Stroke-based Neural Painting and Stylization with Dynamically Predicted Painting Region


## Abstract

We show the training and testing method of our model here.
The training process can be divided into three steps:

(0) Data prepare: download the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset or [ImageNet](https://image-net.org) dataset


(1) Train the stroke renderer;

(2) Train the painter network;

(3) Train the compositor network.

## Train the stroke renderer


To train the neural stroke renderer, you can run the following code
```
python3 compositor/train_renderer_FCN.py
```

## Train the painter network

After the neural renderer is trained, you can train the painter network by:
```
$ cd painter
$ python3 train.py
```

## Train the compositor network

With the trained stroke renderer and painter network, you can train the compositor network by:
```
$ cd compositor
$ python3 train.py --dataset=path_to_your_dataset
```

##Test

### Image to paint
After all the training steps are finished, you can paint an image by:
```
$ cd compositor
$ python3 test.py --img_path=path_to_your_test_img
```

### Test the DT loss

We provide the differentiable distance transform code in compositor/DT.py, you can have a test by:

```
$ cd compositor
$ python3 DT.py --img_path=path_to_your_test_img
```
