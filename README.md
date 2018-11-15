# Using neural nets to transcribe ASL letters

## Capstone Summary
  The goal of this project was to develop a machine learning model capable of reading images of hands making ASL(American Sign Language) letter signs. The goal of such a network would be to develop an application that could directly translate sign language from video into text or perhaps audio. This would serve to make communication easier for deaf or mute people, who otherwise might have to write what they want to say down so that a hearing person would understand them. This would take the onus of translation off of the deaf or mute person, hopefully making their day to day lives a bit easier. It also allows Deaf culture to exist without a need to make it understandable by hearing people.
  For the model itself, I started with a Convolutional Neural Network that I custom built, and fed it data from a kaggle dataset of ASL letters. I would then predict on images taken from video using OpenCV, and output those predictions to the console.
  This resulted in some major errors, so I pivoted to transfer learning and a custom dataset. This led to an incredibly deep rabbit hole, in which I learned quite a bit, but my model could not say the same.

### Data
  My original dataset was from Kaggle, consisting of 87,000 images across 29 classes, every letter of the alphabet plus 'space' and 'delete' characters as well as a set of empty backgrounds. This was a pre-arranged dataset, with every class in individual folders, but with a tiny test set. The test set consisted of 29 images, one for each class, and nothing else, making validation somewhat difficult. This led to me augmenting the training set with images I found in another kaggle dataset, as well as images I took myself.
  
  ![ADD IMAGE OF STATS](images/bad.png)
  
  This proved to be my downfall, as I discovered the data was incredibly specialized, and any outside images weren't recognized. My accuracy plummeted to 5% from where it was, and a new plan was needed.
### Transfer Learning and Data Augmentation
  I then decided to move to transfer learning and create a new, smaller dataset to predict for actual use cases. This led to an in-depth exploration of OpenCV's video reading and image classification methods,something I had not planned to explore so soon. I developed a script to read a video, take individual frames from said video, and then resize and crop the images before saving them. This allowed me to build an initial dataset of ~1200 images that I planned to use with the XCeption model from keras.
  [ADD IMAGE OF AUG PROG]
  The transfer learning began to run into isssues showing my model was not properly optimized for the images I was using. This led me back to using my old model, and creating my own dataset of approximately ~6000 images to start. I wanted a small baseline to begin with, and this seemed best. I wrote another program that piggybacked off of my earlier program to automatically name and assign images from video, and used my computer webcam to extend my dataset.
### Model
  [ADD TENSORBOARD IMAGE]
  I ended up using a 
### Results
  
### For the Future
  I would want to put this into a Flask app so it could be accessed from the Internet and from any browser. Beyond that It could even be ported to phones, making translation from deaf to hearing people easier. I also intend to pour more time into coming up with a better model and a larger dataset.
### References

http://cs231n.stanford.edu/reports/2016/pdfs/214_Report.pdf

https://arxiv.org/pdf/1806.02682.pdf

https://www.kaggle.com/grassknoted/asl-alphabet
