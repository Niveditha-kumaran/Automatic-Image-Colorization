



## INTRODUCTION
The traditional techniques of colouring black and white images are regarded as dilatory, due to the requirement of human intervention. Some of the techniques include:

    -Convex optimisation using colour seeds (manually prediction)
    -Colour transfer technique which uses parts of similar coloured images
    -Semi-Neural Networks in which image features are extracted and fed into a vanilla neural network
    -Standard Photoshop in which colours are selected manually and parts of the grayscale image is coloured
    
 
These are not efficient techniques because it takes a lot of time for each image to be coloured and manual inspection has to be done. The techniques also do not learn the colour patterns.

As a boon in regard to such a problem, CNNs have recently emerged as the highest standard for solving image classification problems, achieving error rates lower than 4 percent in the ImageNet challenge.

CNNs owe much of their success to their ability to learn and discern colours, patterns, and shapes within images and associate them with object classes. Hence, these characteristics naturally lend themselves well to colorizing images since object classes, patterns, and shapes generally correlate with colour choice.
The main objective is to take a grayscale image as input and provide a coloured image as output which is as close as that of the real image colour.

 
## IMPLEMENTATION
 We use Convolutional Neural Nets because colour of a pixel is strongly dependent on the features of its neighbours. For good output results, the VGG model is used. Images are of RGB format and are converted to YUV format to separate them as input and output channels (Y-Intensity, U,V - Chrominance). So, Y channel of the image is taken as input and U and V channels are the output of the model. The model consists of three parts:
 
- Extract features from the image: VGG-CNN model is used for this purpose. The different neurons will capture different features of the     image such as edges and at deeper layers, more focused features such as tire of a car can be captured. 

- Hypercolumn: Taking all the layers obtained from above step and creating a hypercolumn of the grayscale image. A hypercolumn for a      pixel in the input image is a vector of all the activations above that pixel which is obtained by forwarding an image in the VGG        network and extracting a few layers, upscaling them and concatenating them all together.

- Shallow net: A shallow net which takes the concatenated layers as input and outputs the U,V channels. The shallow net will also be a    small sequential 2 or 3 layer CNN. 2 loss functions will be taken, Mean Squared Error and Absolute Error, and results will be            observed and compared with both.
    
For training, we will be using an available data-set from the Internet which will contain natural scenes like mountains, beaches, fruits, animals and human portraits. In training, features are extracted using the VGG model and shallow net's parameters are trained to colour those features. After training, a new grayscale image can be fed into the network to obtain a coloured image.
## ARCHITECTURE
  1. The image is feed-forwarded into the VGG-model
  2. The higher abstract layers of the VGG model captures the important features present in the image
  3. Some of the layers are selected and are stacked as an object
  4. This is called hypercolumn. It holds a lot of information about the image and its features and is treated as the input to the shallow net for training for abstract colours. Hypercolumn dimension: 224x224x963
  5. The hypercolumns are used as input to the shallow net and the chrominance channels (UV) are the output of the CNN network.
  6. The weights and biases of the CNN shallow net are trained to give the corresponding pixel values of the chrominance channels.
  7. The UV channels are combined with the input Y channel to output the plausible colour image of the black and white input image.
  
 ## RESULT
#### Observations

The Convolutional Network Networks trained well, producing outputs of plausible images. As the network is trained mainly on images of coasts, forests, open country, street images, it can colour those category images well. It produces a possible image even in case of other category images but not all coloured images are good. This is because the model never learnt the certain features local to that particular category images. For example, it learns the horizon feature, separating river from sky, but does not learn how to colour graffiti properly as training images did contain images of beaches but not graffiti walls.

The network also cannot predict the correct colour of a general object such as colour of clothes or colour of a car. This can be explained by understanding that general objects like clothes and cars are present as different colours in the training images. The network tries to learn an interpretable colour when it detects those same features in different colours. Also, it is hard even for humans to predict a general object colour when it is show in the grayscale format. Although the model cannot also predict the correct colour, it does produce a possible colour for that general object.

The model also seems to favour Sepia tone for some group portrait images. This can be resolved by training more on group portraits for a longer time. 

Although, the model does not meet the required standards of colourisation of category of images not present in the training dataset like grafitti and portraits, it performs well and outputs good coloured images on test images which are scenery based - coasts, open country, forests, street category images. 

#### Training
The model was trained on an online GPU - FloydHub. It used a Tesla K80 GPU with 12GB RAM. LabelMe dataset of MiT is used to train a thousand images of coasts and beach, open country, forest and street images. The model is set to train for 20 hours.

## MODEL ANALYSIS
#### Comparison Of Images

The gray scale image is taken and passed through the model to obtain it's coloured version.

The forward pass of the gray scale image takes 40 seconds to process.
Hence, it just takes 40 seconds to colour an image which in comparison to traditional methods of colouring, is a drastic reduction in time.

The comparison of sample input gray scale images are contrasted with the output coloured images below. 

![Original Image](https://github.com/Niveditha-kumaran/Automatic-Image-Colorization/blob/master/Sample%20Outputs/original_25.jpg)
![Colored Image](https://github.com/Niveditha-kumaran/Automatic-Image-Colorization/blob/master/Sample%20Outputs/coloured_25.jpg)



![Original Image](https://github.com/Niveditha-kumaran/Automatic-Image-Colorization/blob/master/Sample%20Outputs/original_20.jpg)
![Colored Image](https://github.com/Niveditha-kumaran/Automatic-Image-Colorization/blob/master/Sample%20Outputs/coloured_20.jpg)



    
 
