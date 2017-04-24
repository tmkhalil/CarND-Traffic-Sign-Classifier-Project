# **Traffic Sign Recognition** 

## Talaat Khalil

[//]: # (Image References)

[image1]: ./examples/sample0.png "Dataset Sample 1"
[image2]: ./examples/sample1.png "Dataset Sample 2"
[image3]: ./examples/sample2.png "Dataset Sample 3"
[image4]: ./examples/train_stats.png "Training data histogram"
[image5]: ./examples/valid_stats.png "Validation data histogram"
[image6]: ./examples/valid_stats.png "Testing data histogram"
[image7]: ./examples/web0.png "Web Test Sign 1"
[image8]: ./examples/web1.png "Web Test Sign 2"
[image9]: ./examples/web2.png "Web Test Sign 3"
[image10]: ./examples/web3.png "Web Test Sign 4"
[image11]: ./examples/web4.png "Web Test Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

To get an idea about the dataset, I got the following statistics using simple invocations of the numpy arrays attributes as shown in the provided notebook:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 X 32 X 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

To get more insights abou the data set I visualized some of the samples and I printed their classes names. Here are some samples:

![alt text][image1]
![alt text][image2]
![alt text][image3]

Then, I visualsed the distribution of the data over the different classes. It seems that the data was splitted in a total random way that's why the ditributions are almost consitent over the three data splits as shown below.

![alt text][image4]
![alt text][image5]
![alt text][image6]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For pre-processing, I did the minimal possible preprocessing. First, I converted the images to grayscale by applying the following formula given the Red, Green, and Blue channels values:

Gray = 0.2989 x Red + 0.5870 x Green + 0.1140 x Blue

This was useful for reducing the number of filters (trainable parameters).

Then, I normalized the data to be between -1 and 1 as it's helpful for the optimization process.

In addition to the stated reasons, I noticed performance increase on the validation set after applying the pre-processing steps.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x26   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x26 				    |
| Flatten       		| Output = 650        							|
| Fully connected		| Input = 650, Output = 350        	     		|
| RELU					|												|
| Dropout               | 0.6 keep probability                          |
| Fully connected		| Input = 350, Output = 84        	     		|
| RELU					|												|
| Fully connected		| Input = 84, Output = 43        	     		|
| Softmax				|            									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I started with LeNet implementation as a baseline. Over my differnt experiments, increasing the number of training epochs always helped to get better performace. It seemed like 200 epochs was good enough for my final model given some time restrictions.

I tried bigger batches namely 256, 512, and 1024. It was better from the computational point of view (faster), however using batch size of 128 performed better on the validation set.

I tried increasing and decreasing the learning rate. I experimented with 0.0001, 0.0005, 0.0007, 0.0008, 0.001, 0.005 and 0.01. Using 0.0008 gave me the best validation peroformance.

I tried to introduce dropout at different places in the network, however the best performance was by introducing one dropout layer after the first fully connected layer with a dropout rate of 0.4.

finally, I suggested that I can get better improvements by increasing the Number of convolutional filters. I increased each of them by 10 and the performance got much better on both the validation (96%) and the test set (94.2%).

That was enough for getting the required performance, however if I had more time, I would have experimented with more archtichural changes including using convolutional filters of different sizes. I also would have considered adding some artificial examples by introcusing some rotation and distortion variations. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Already discussed in the previous point.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]

qualities and difficulities are discussed in the next section.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)              			| 
| Speed limit (50km/h)  | Speed limit (50km/h)							|
| Pedestrians			| Go straight or right							|
| Road work      		| Road work 					 				|
| Stop      			| Dangerous curve to the left   				|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This test set size is not by any means considered appropriate for evaluation. The "Stop" image orientation and small size seemed to trick the model. Also the quality of the Pedestrians image confused the model. This strongly suggests introducing some fake and distorted data during the training phase.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model seemed to be pretty sure regaring it's correct predictions and pretty uncertain regarding the misclassified ones as shown in the following table.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)							| 
| 1.0     				| Speed limit (50km/h)							|
| 0.0999				| Go straight or right							|
| 1.0	      			| Road work      				 				|
| 0.0999			    | Dangerous curve to the left					|


