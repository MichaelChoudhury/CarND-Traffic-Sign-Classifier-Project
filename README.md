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

The bar chart shows the data distribution of the training data. Each bar represents one class (traffic sign) and how many samples are in the class. The mapping of traffic sign names to class id can be found here: [signnames.csv](./signnames.csv)

![histogram](./images/histogram.png "histogram")

Here are some traffic signs from the training data set. More can be found in the jupyter notebook.

![original training images](./images/training_data_raw.jpg "original training images")

### Design and Test a Model Architecture

#### Preprocessing

|original image|preprocessed image
|----|----|
|![original image](./images/original_image.png "original image")|![preprocessed  image](./images/preprocessed_image.png "preprocessed image")|

#### Model Architecture
 
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
| Fully Connected | connect every neuron from layer above			|688|84|
| Fully Connected | output = number of traffic signs in data set	|84|**43**|

#### Model Training

I trained the model on my local machine with a GPU (NVIDA GeForce GT 750 M). It is not a high-end gpu, but it has a compute capability of 3.0, which is the absolute minimum requirement of tensorflows gpu support. Compared to my cpu, the training was about 3.3 times faster.

**NOTE:** If you are on windows and get an CUDA_ERROR_ILLEGAL_ADDRESS in gpu mode, it is probably an issue with ```tf.one_hot()``` Have a look: https://github.com/tensorflow/tensorflow/issues/6509 That is the reason why I use the function ```one_hot_workaround``` in my code.

Here are my final training parameters:
* EPOCHS = 35
* BATCH_SIZE = 128
* SIGMA = 0.1
* OPIMIZER: AdamOptimizer (learning rate = 0.001)

My results after training the model:
* Validation Accuracy = **97.1%**
* Test Accuracy = **95.4%**

#### Solution Approach

My first implementation was LeNet-5 shown in the udacity classroom. I modified it to work with the input shape of 32x32x3. It was a good starting point and I get a validation accuracy of about 90%, but the test accuracy was much lower (about 81%). 


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
* LeNet-5: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
* Project specification: [Udacity Rubrics](https://review.udacity.com/#!/rubrics/481/view)
* Udacity repository: [CarND-Traffic-Sign-Classifier-Project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)
* [Udacity Self Driving Nanodegree](http://www.udacity.com/drive)
