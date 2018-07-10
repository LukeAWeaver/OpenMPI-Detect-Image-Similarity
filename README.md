# OpenMPI-Detect-Image-Similarity
OpenMPI Image Similarity Detector is a C++ program that compares images and outputs a percentage indicating how similiar the selected images are to eachother.

# How does it work?
OpenMPI Image Similarity Detector utilizes a high performance message passing Library to send RGB histograms to other processes for their evaluation! The program takes in N images and fires off N processes, each to process an image. Each process computes a histogram by evaluating the RGB values of their native image. Once all processes are done creating their histogram they broadcast their data to each process in an All-to-All fashion.  Bird Watcher uses a convolutional neural network trained using the Keras machine learning library on the [Caltech-UCSD Birds-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200.html). The model was converted for the CoreML Swift library and integrated with the iPhone camera so that pictures taken are saved and then run through the model resulting in a prediction.

# Why use OpenMPI as Opposed to threads?
OpenMPI allows for data transfer messaging between processes (even processes across multiple machines); sending large data between processes can have significant overhead but for this project (depending on the image size) the overhead is neglectable. Threads are great but because the data sent relatively small, there would be larger overhead accessing shared memory.

# Where can I get it?
Right here!
