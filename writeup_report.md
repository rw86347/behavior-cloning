# **Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/recover01.png "Recovery Image"
[image2]: ./examples/recover02.png "Recovery Image"
[image3]: ./examples/recover03.png "Recovery Image"
[image4]: ./examples/normal01.png "Normal Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* finalModel.h5 containing a trained convolution neural network 
* writeup_report.md 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py finalModel.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

On line 76 you can see the model that was used.  I used the same model that Nvidia used for their
self-driving cars.

#### 2. Attempts to reduce overfitting in the model

This project was much nicer than the last project in that I could collect as much data as I wanted. 
and Trust me I collected a TON of data.  I had 64,000 images that I used for training.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 108).

#### 4. Appropriate training data

This is where my code probably differed from everyone elses!!!  I am not good at video games and actually
hate playing them.  I was running off the road all the time at first.  Then I realized this was only half
bad as I need recovery data.  So I created a new python program (deleteData.py) that allowed me to play 
back the video and delete data that was bad to train with and keep the good stuff.

This meant that when I ran off the road I would delete the data of actually running off the road but 
keep the data where I recovered.  :-)  I only showed the good stuff that I did.  

As time went on I got better at driving that I keeped that data too.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

I figured if Nvidia like their NN strategy this what not what I needed to spend my time on.  Rather
I spent time currating my data.

To combat the overfitting, I modified the model so that ...

I got a TON of data.  It was easy to get data so why not get lots and lots of currated perfect data.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76ff) consisted of a convolution neural network as designed by
Nvidia

#### 3. Creation of the Training Set & Training Process

This is where my code probably differed from everyone elses!!!  I am not good at video games and actually
hate playing them.  I was running off the road all the time at first.  Then I realized this was only half
bad as I needed recovery data.  So I created a new python program (deleteData.py) that allowed me to play 
back the video and delete data that was bad to train with and keep the good stuff.

This meant that when I ran off the road I would delete the data of actually running off the road but 
keep the data where I recovered.  :-)  I only showed the good stuff that I did.  

As time went on I got better at driving that I keeped that data too.

This data was also augmented by using the side camera's and flipping the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

This before creating the generator I was using fit().  This was working but not too well.  
Then after I created the fit_generator() my performance increased by 100x.  It was HUGE!  
I think this was due to the randomization.  Similar pictures were always sitting beside each
other in the training set.

Notice the arrow in the bottom left.  This is the angle of my steering wheel.  
The window is from my data curration tool.  The number in the top left is the 
frame number (CSV line number)

![Recovery01][image1]
![Recovery02][image2]
![Recovery03][image3]

Here is one where I am centered and driving straight ahead. 
![Recovery04][image4]