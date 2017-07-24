#**Traffic Sign Recognition WriteUp** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image4]: ./TrafficImages/1.jpg "Traffic Sign 1"
[image5]: ./TrafficImages/2.jpg "Traffic Sign 2"
[image6]: ./TrafficImages/3.jpg "Traffic Sign 3"
[image7]: ./TrafficImages/4.JPG "Traffic Sign 4"
[image8]: ./TrafficImages/5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Link to my [project code](https://github.com/stevendae/CarND-TrafficSignClassifier-P2-StevenLee/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python and numpy functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32 pixels
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...
Two exploratory visualizations of the data set are in the iPython notebook or the HTML file. The first exploratory visualization is a visual display of one of every unique label or trafffic sign in the dataset. This was later useful when the new test images were loaded into memory and had to be classified so that we may evaluate the accuracy of the neural network. The second visualization was a histogram that showed the number of samples that were included for each label in the training set. By assessing the count of each label, we can determine how the network will form its biases based on the skeness of the training data.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to normalize the image data because I wanted to minimize the error in the loss function which can occur when computing very large numbers and to also put less emphasis on the optimizer doing work by centering the range of values on a zero mean with equal variance. The way that this was performed was simply using the normalize method in the OpenCV2 library. Here the alpha(min) and beta(max) values which are the boundaries of the normalize function are -1 and 1. 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3     	| 1x1 stride, valid padding, outputs 28x28x6, 6 Kernels |
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride, valid padding, outputs 14x14x64 |
| Convolution 5x5x6	    | 1x1 stride, valid padding, outputs 10x10x16, 16 Kernels |
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride, valid padding, outputs 5x5x16 |
|	Flatten              | Input 5x5x16 | Output 1x400
|Fully Connected       | Input 1x400, Output 1x120 |
| RELU					|												
|Fully Connected       | Input 1x120, Output 1x84 |
| RELU					|			        |
|Dropout|                Keep Probability = 0.5 |
|Fully Connected|        Input 1x84 , Output 1x43 |

Return Logits - 1 x 43

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer which uses Kingma and Ba's Adam algorithm that deploys stochastic gradient descent while storing momentum values in the direction of optimization. This was also the default optimizer in the LeNet Lab. The batch size that was used was 64 as according to a report from Quora about a validation accuracy that was peaked at a batch size of 64. (https://www.quora.com/Intuitively-how-does-batch-size-impact-a-convolutional-network-training) The number of epochs chosen were 30 although it appears that according to the latest run of the neural network the network surpasses 92% accuracy after 16 epochs but rises to 94.4%. The learning rate that was selected was 0.0005. The initial trials were ran at 0.001 but improvements occured from 91 to 92% from a 0.0005 point reduction in learning rate. The dropout rate was also reduced from 1.0 to 0.5 which further improved results from 92 to 94%. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I decided to use the LeNet architecture as it indicated that validation accuracy was already known to be 89% according to the README file. Hence if I directly applied it I would have a strong starting point and improve further from there. I was intially getting some technical issues with putting all the pieces together. Once I was able to achieve validation accuracies of above 85% I tweaked further. At this point my batch size was 128 and epochs were 10. When I encountered the Quora article of how someone achieved a peak accuracy at a batch size of 64 I changed my batch size and increased epochs to 20 and I was able to reach accuracies of 91% From there I lowered my learning rate from 0.001 to 0.0005 and achieved 92% accuracy. I also noticed by default that the dropout keep rate was 1.0 hence wasn't active. I lowered it to 0.5 and got my final validation accuracy at 94.4%

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.4%
* test set accuracy of 93.5%
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the aspect ratio of the image shows that of a more rectangular nature. Thereby when it is resized the sign will be squished and then the positions of activated points on the activation map maybe offset from the activation map that is finalized during training. 

The second image appears to be more square than the the others and is in high resolution 1578x1394 pixels. Resizing to a significantly smaller size will cause a reduction in information of the original image. 

The third image appears to have camera noise or grain that is filtered on top of the sign image which distorts the overall composition. 

The fourth image has the top of the sign cut off resulting in a loss of information as a full composite. There is also another sign below the traffic sign that may pick up its own activations. Also the trees in the back create alot of noise in the image.

The fifth image appears as if it has a yellow hue filter on the image that makes the sign change. Since the dataset was not converted into greyscale this will emphasize values in the yellow channel.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:--------------------------------------:|:---------------------------------------------:| 
| Right-of-way at next intersection    		| Right-of-way at next intersection           		|
| General Caution                     			| General Caution		                       						|
| Bumpy Road			                         	| Speed Limit (120km/h)				               						|
| Speed Limit (30km/h)	      	          	| Yield					                           		     		|
| Priority Road			                       | Roundabout Mandatory                    						|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares infavorably to the accuracy on the test set of 93.5%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located near the bottom of the Ipython notebook.

First Image

| Probability          	|     Prediction	        			               		| 
|:---------------------:|:---------------------------------------------:| 
| .55         			| Right-of-way at the next intersection   									| 
| .17     				|Dangerous curve to the left 										|
| .16					| Beware of ice/snow											|
| .11	      			| Slippery Road				 				|
| .09				    | Priority Road     							|


Second Image

| Probability          	|     Prediction	        				                  	| 
|:---------------------:|:---------------------------------------------:| 
| .48         			| General Caution   									| 
| .27     				|Right-of-way at the next intersection 										|
| .22					| Traffic Signals											|
| .18	      			| Pedestrians			 				|
| .14				    | Roundabout mandatory     							|

Third Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .15         			| Speed limit (120km/h)   									| 
| .12     				|Speed limit (60km/h)									|
| .11					| Wild animals crossing										|
| .08	      			| Bumpy Road 				|
| .05				    | Beware of ice/snow   							|

Fourth Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .20         			| Yield 									| 
| .13     				|Turn left ahead								|
| .13					| No passing											|
| .09	      			| Ahead only				 				|
| .05				    | Go straight or right    							|

Fifth Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .25         			| Roundabout mandatory   									| 
| .12     				| Keep right									|
| .07					| End of speed limit (80km/h)											|
| .04	      			| Go striaght or left				 				|
| .01				    | Right-of-way at the next intersection    							|

