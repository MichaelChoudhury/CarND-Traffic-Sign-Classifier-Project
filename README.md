# Traffic Sign Classifier [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This is my solution of Project 2 of Udacity's Self Driving Car Nanodegree.  

### Goals & steps of the project
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Jupyter Notebook

* Source Code: [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb)

### Basic summary of the data set

* Download dataset: [traffic-signs-data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Exploratory visualization on the data set

The bar chart shows the data distribution of the training data. Each bar represents one class (traffic sign) and how many samples are in the class.

![histogram](./images/histogram.png "histogram")

The next image show some traffic sign images from the training data.
![original training images](./images/training_data_raw.jpg "original training images")

### Design and Test a Model Architecture

LeNet-5 implementation

#### Preprocessing

|original image|preprocessed image
|----|----|
|![original image](./images/original_image.png "original image")|![preprocessed  image](./images/preprocessed_image.png "preprocessed image")|
 
 My final model consisted of the following layers:

| Layer         		|     Description	        					| Input |Output| 
|:---------------------:|:---------------------------------------------:| :----:|:-----:|
| Convolution 5x5     	| 1x1 stride, valid padding, RELU activation 	|**32x32x1**|28x28x48|
| Max pooling			| 2x2 stride, 2x2 window						|28x28x48|14x14x48|
| Convolution 5x5 	    | 1x1 stride, valid padding, RELU activation 	|14x14x48|10x10x96|
| Max pooling			| 2x2 stride, 2x2 window	   					|10x10x96|5x5x96|
| Convolution 3x3 		| 1x1 stride, valid padding, RELU activation    |5x5x96|3x3x172|
| Max pooling			| 1x1 stride, 2x2 window        				|3x3x172|2x2x172|
| Flatten				| 3 dimensions -> 1 dimension					|2x2x172| 688|
| Fully Connected       | connect every neuron from layer above			|688|84|
| Fully Connected       | output = number of traffic signs in data set	|84|**43**|

EPOCHS = 35
BATCH_SIZE = 128
EPOCH 35 ... Validation Accuracy = 0.971
Test Accuracy = 0.954
Optimizer: AdamOptimizer

### Test on new images


![new images](./images/new_images.png "new images")
![priority road softmax k-top](./images/priority_road_k_top.png "priority road softmax k-top")
![speed limit softmax k-top](./images/speed_limit_k_top.png "speed limit softmax k-top")

| Image			        |     Prediction		| 
|:---------------------:|:---------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)  | 
| Priority road   		| Children crossing 	|
| Yield					| Yield					|
| Stop	      			| Stop					|
| No entry				| No entry    			|

### Resources
* Source code: [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb)
* Pickle files: [traffic-signs-data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* Original data set: [German Traffic Sign Data Set](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Project specification: [Udacity Rubrics](https://review.udacity.com/#!/rubrics/481/view)
* Udacity repository: [CarND-Traffic-Sign-Classifier-Project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)
* [Udacity Self Driving Nanodegree](http://www.udacity.com/drive)
