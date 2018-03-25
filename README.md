# **Traffic Sign Recognition** 

## Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the German Traffic Sign Dataset. After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Training_data_bar_graph.png "Train-Data Visualization"
[image2]: ./images/Validation_data_bar_graph.png "Valdiation-Data Visualization"
[image3]: ./images/Test_data_bar_graph.png "Test-Data Visualization"
[image4]: ./images/30kph.jpg "30 Kph Tsign"
[image5]: ./images/Arterial.jpg "Priority Road"
[image6]: ./images/Menatwork.jpg "Road Work"
[image7]: ./images/Roundabout.jpg "Round-about"
[image8]: ./images/ChildrenCrossing.jpg "Children-Crossing"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sourav90in/Traffic-Sign-Classifer/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set by utilizing the shape method:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x2x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across classes
in Training, Valdiation and Test sets.From the Data visualization of the training data, it can be seen that some of the classes
are very poorly represented as compared to average number of examples per class which was 809.

![alt text][image1]
![alt text][image2]
![alt text][image3]

Also, I have plotted 20 examples from the training-set per class, which can be seen in the output of the section "Include an exploratory visualization of the dataset" in the Source code file. This helped me visualize the variation in Intensities/Scale/Perturbations of each Traffic-sign class across images belonging to the same class.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My initial approach was to convert to Grayscale and perform normalization to (-1,+1) range. In the first iteration of this appraoch,I had utilized the Luminosity approach to convert an image to Grayscale i.e using cvtColor from BGR2GRAY but eventually found that the averaging method of averaging intensities across R,G and B channels gave better results for the initially chosen Lenet Architecture.I got the idea to try out this approach for Grayscaling from the link: https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/.

As I moved from the basic Lenet architecture to Multi-scale Lenent architecture proposed in Le Cun's paper, I found that using RGB image channels instead of Grayscale gave better results for the Test-Data set and almost equivalent results for the Validation data-set, so I continued using RGB channels instead of preprocessing to Grayscale.

The second preprocessing step that I adopted was that of Data-Augmentation by augmenting the Training set with random rotation of angles between -15 to 15 degrees as was done in the Multi-scale Lenet architecture proposed by Le Cun. This helped achieve marginal increase in Validation accuracy.

The third preprocessing step, was augmenting the data with randomly Scaled between [0.9,1.1] and randomly Translated variations [-2,2] pixels of the Training data-set with the previously adopted data normalization method of scaling each channel to (+1,-1) range. This appraoch was also borrowed from the approaches for Data-Augmentation suggested in Le Cun's paper. Eventually, I also tried Min-Max Scaling as well as Standard Scaling instead of data normalization to (+1,-1) range and finally settled with Min-Max Scaling as it gave the best results.

Also, I tried another Data-Augmentation approach suggested in the Alex-Net paper where Principle Component analysis was done per pixel per channel on the entire-training set and generated perturbations for each of the R,G and B channels was added to all the images of the training data. However this didnot help in any improvement to the Valdiation accuracy, so I have not utilized it, in the final approach taken.

Also, another approach of Data-Augmentation was tried inspired from the fact that some of the classes were poorly represented in the Training Data-set. I had segregated the Training-Data into two categories: One category was for examples belonging to classes whose representation(i.e. number of examples) was greater than the mean, and another category was for examples belonging to classes whose represenation was lesser than the mean. Further, the Rotation., Translation and Scaling would only be done for Examples belonging to the poor Represetation classes and no Augmentation would be done for Examples belonging to the good Represenation classes. This approach yeilded around 93% Validation accuracy with very limited memory usage and can be used in a system with memory constraints but I disabled the usage of this approach since a better Validation accuracy can be achieved without it and since the memory constaints didn't exist using the AWS GPUs.


To summarize, my Data Augmentation and preprocessing steps are as follows for RGB images:
a) Input Training Data and Augment it with Rotated Samples of the images and shuffle them.
b) Input Training Data and Augment it with Translated Samples of the images and shuffle them.
c) Input Training Data and Augment it with Scaled samples of the images and shuffle them.
d) Combine all the generated image samples into an Extended Training Set and normalize them with Min-Max Scaling.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		 |     Description	        					| 
|:---------------------: |:--------------------------------------------:| 
| Input         		 | 32x32x3 RGB image   							| 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					 |												|
| Least Response normal  |	depth-radius of 6							|
| Max pooling	      	 | 2x2 stride,  outputs 14x14x6  				|
| Max pooling and flatten| 2x2 stride and flattened op is 294			|
| for conv layer1 Residue| 												|
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					 |												|
| Least Response normal  |	depth-radius of 7							|
| Max pooling	      	 | 2x2 stride,  outputs 5x5x16  				|
| Flatten and concat     |  ouput shape 1x694                           |
  with Residue           |                                              |
| Fully Connected layer  | 694x120     									|
| RELU					 |												|
| Fully Connected layer  | 120x84    									|
| RELU					 |												|
| Fully Connected layer  | 84x83    									|
| Softmax of Logits      |												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I first compared Adamoptimizer as compared to Stoachastic gradient descent for different learning rates without any Data-Augmentation and observed that AdamOptimizer always output Stoachastic Gradient Descent in terms of the achievable Validation accuracy.
Also I experiemtned with different batch-sizes for AdamOptimizer for the same learning-rate and observed that retaining the same batch-size of 128 as in the Lenet Lab implementation gives good results across Epochs.
I also kept increasing eh Epochs gradually from 10 to 20,30,40 and finally fixated upon fixing the number of Epochs as 60as the Valivation accuracy results were saturating towards the final stages of the Epochs.
I only tried out learning rates of 0.1, 0.01 and 0.001 and adapted the best suited batch-sizes for each of these learning rates and found that a learning rate of 0.001 was giving the best results for Adam Optimizer with batch-size of 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.9% 
* test set accuracy of 94.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture chosen was from the Lenent Lab and it was chosen to get a baseline on the maximum achievable Validation
accuracy.

* What were some problems with the initial architecture?
The max achievable Validationa ccuracy could not exceed 93.5% and as times yeilded accuracies lesser than 93%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I came across the Multiscale architecture of Lenent introduced in Le Cun's paper and tried out implementing the same as it was suggested that Multiscale architecture performed better than the basic Lenet Model and indeed helped in achieving higher validation accuracies as compared to Lenet.
The second architectural change that assisted in boosting the accuracy was using the Least Response Normalization which was introcuded in the AlexNet paper and was used on top of the RELU ouputs in the Convolution layer stages. This also intutitively made sense as it simualtes the inhibitory effect of neurons.

* Which parameters were tuned? How were they adjusted and why?
The only parameter that I adjusted was that of the number of Epochs after fixing the batch-size and learning Rate. I kept varying it from 10 to 20 to 30 to 40 to 50 and finally stopped at 60, as the variations in Validation Accuracies across Epochs towards the final stages seemed to saturate.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Using the Multiscale architecture of Lenent was an imporatant design choice becasue it made sense to carry over the raw residual features that would have been identified at the end of the first conv layer to assist the classifier in the fully connected stages since the input to the fully connected layers would then be a good mix of Finer features identified by the second conv layer and coarse features identified by the first conv layer.
I have not implemented a dropout layer since the Validation accuracies were not in the very high 90's and the current architecture gave a good balance between Valdiation and Test Accuracies.

If a well known architecture was chosen:
* What architecture was chosen?
I have chosen a mix of AlexNet and the Multiscale Lenet architecture.

* Why did you believe it would be relevant to the traffic sign application?
the Multiscale lenet architecture has shown to give good results in the paper "Traffic Sign Recognition with Multi-Scale Conv Networks by Yann LeCun" so I decided to try it out. Also using the Least Response Normalization from the AlexNet paper on the output of the ReLus suggested an improvement on the overall validation accuracy, so I attmepted to implement it and see its results.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The model's accuracy good accuracy rate on the Test-Set shows that it can recognize new data pretty well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

All these images are of different sizes but since the Model architecutre takes an input image of size 32x32x3, I have resized these images to the 32x32 size before feeding them to the Model for classification.

The loaded and resized images are displayed in the Ipython notebook Step 3 and it can be seen that the first image might be difficult to classify because upon resizing there looks to be a lot of noise. Also the image Road Work was probably difficult  to classify due to presence of other patches of Noise in the Red channel.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 kph         		| Turn Right ahead								| 
| Priority road			| Priority road									|
| Road-Work				| Right-of-way at the next intersection			|
| Round-about      		| Round-about mandatory			 				|
| Children-Crossing		| Children-Crossing    							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The classification of the image
showing 30 kph was particularly poor, since the training-set for the class 30 kph had the second higheest number of training examples.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Turn right ahead sign witha  very high probability of 0.99 whereas the image is actually that of a 30 kph speed limit sign. The top five soft max probabilities are as below, none of which belong to the Traffic Sign Class of 30 kph, indicating a very bad case of mis-classification:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99685049e-01		|  Turn right ahead								| 
| 3.14376637e-04		| Keep left								     	|
| 5.09809013e-07		| Go straight or left							|
| 1.60833210e-07		| Yield			 			                 	|
| 2.68427058e-08	    |  End of all speed and passing limits			|



For the second image, the prediction is Priority Road with almost 100% certainity and the image is indeed that of a Priority Road Traffic Sign and has very low probabilities(almost close to zero) for the other 4 top classes. This is expected as in the resized scaled-down image too, its quite clearly visible that the sign is that of a Priority Road.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		|  Priority road								| 
| 1.79868745e-26		| Stop								     	    |
| 2.25483209e-31		| Ahead only							        |
| 1.74094901e-33		| No passing for vehicles over 3.5 metric tons	|
| 1.12303819e-33	    |  Go straight or left			                |

For the third image, the prediction with the highest probability is that of Right-of-way at the next intersection, where-as the actual image is that of a Road-Work Traffic sign. This mis-classification was somewhat expected due to high presence of noise in the Red channel of the image.However the second top-most prediction is that of Road-work which is the actual correct prediction.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.91569996e-01		|  Right-of-way at the next intersection		| 
| 8.14233441e-03		| Road work								     	|
| 2.87614472e-04		| Pedestrians							        |
| 4.22264144e-08		| Children crossing	                            |
| 2.53426252e-10	    |  Speed limit (20km/h)			                |

For the fourth image, the top-most prediction of that ofRound-about mandatory is indeed correct and the prediction is that of almost certaintly(close to 1):

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		|  Roundabout mandatory		| 
| 3.71758830e-13		| Go straight or left			     	|
| 1.15623851e-15		| Keep right					        |
| 5.68447675e-16		| Speed limit (50km/h)	                |
| 5.99329147e-17	    |  Ahead only			                |

For the fifth image, the top-most prediction is that of Children Crossing with very high probability of 0.990 which is indeed correct as the image is that of a Chilren-Crossing Traffic Sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.90863383e-01		|  Children crossing		                    | 
| 9.07240435e-03		| Priority road								   	|
| 3.33693351e-05		| Pedestrians						            |
| 1.55608031e-05		| Right-of-way at the next intersection         |
| 1.25292754e-05	    |  Beware of ice/snow          	                |
