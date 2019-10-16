# FaceDetectionAndRecognitionOnIndianFaces

The repository contains implementation of Facenet from scratch using OpenCV and Keras Framework on Indian Movie Face Dataset.
The following steps are followed for the same.

## PREPROCESSING

Face Detection was done using the following steps:
* Walking through the training images for pre-processing.
* Resizing the image to 160 x 160.
* Using the dlib frontal detector with 68-point facial landmarks.
* Getting the eyes, ears and nose x-y coordinates as a template for face detection and
using Min-Max Scaling on the same so that the coordinate are uniformly
distributed in the range (0,1).
* Using affine transformation of OpenCV library for aligning the image and resizing it to
96 x 96.
* The above pre-processed images are saved to the output directory with the same
tree structure as that of the original dataset.

## MODEL ARCHITECTURE

* Inception v3 Model was used with to build the same.
* It comprised of convolution layers with activation as ReLU and batch
normalization layer at the end of it.
* Weight Decay was used to prevent overfitting.
* The inception concept was used that is branches were used comprising of
convolution and max pooling layers which were later concatenated.
* The output of the concatenated layers is flattened out and is passed to a Dense
Layer with 128 units.
* The output from the Dense Layer is passed to the L2 normalizer function of keras
using Lambda Functionality.

## MODEL LOSS LAYER AND CUSTOMIZATION

* A customizable Loss Layer can be added to the model.
* Here Triplet Loss Layer is added with triplet loss defined in it.
* The margin for triplet loss is set to 0.2
* The above calculated loss is added to the layer using add_loss functionality.
* Three input tensors i.e. anchor, positive and negative with shape (96,96,3) are
passed into the model to get embedding tensors.
* The previously defined model architecture can be extended to add Triplet Loss
Layer as output of the model.
* The customized model is compiled with loss=None and optimizer used here is
AdaBound with initial and final learning rates as hyperparameters.

## GETTING THE TRAINING AND TEST DATA

* The image class folders were randomly shuffled.
* The train rate was set as 0.9.
* The corresponding preprocessed images are moved to the train and test folders.
* The train and test folders are the image classes.
* A constraint was there that the images in the original training sample for each
label should be greater than 1 so that the train and the test folders get atleast 1
image.

## OFFLINE TRIPLET GENERATION

* Using naive approach to calculate anchor, positive and negative triplets.
* The method is loosely based on random indices initialization and getting anchor
and positive triplets closer while the negative triplets further away from the
anchor.
* The anchor, positive and negative images are normalized by dividing it by 255.
* The batch size for anchor, positive and negative triplets is set to 32.

## TRAINING ON TRIPLET LOSS

* Computing train generator from the offline triplet method approach.
* Fitting the train generator in keras model using fit_generator method.
* Using AdaBound optimizer with initial learning rate 0.001 and final learning rate
0.1
* The steps_per_epoch used here is 500 and number of epochs 100.

## TESTING ON TRIPLET LOSS

* Getting the Training and Test embeddings from model post training.
* Calculation of Euclidean Distance between test embeddings based on threshold
value for class labels.
* Computing accuracy scores based on the above.

## VISUALIZING THE TRAIN AND TEST EMBEDDINGS

* The above train and test embeddings can be used for visualization.
* Tensorboard comes as an inbuilt notebook extension in colab environment.
* The projector configuration of tensorboard can be setup accordingly by giving the
corresponding feature vector.
* The embedding vectors can be visualized dynamically and t-SNE can be used to
reduce the dimension of them to project them into 2D space and forming clusters
based on perplex and learning rate which can be set by the user.

## TRAINING AND TESTING ON SOFTMAX LOSS

* As an alternative the model can be trained on SoftMax loss.
* The model architecture is same except the L2 normalization layer is not used and
an additional Dense layer with number of units equal to the number of output
labels is added.
* The class labels are encoded categorically as they are string values, they are
converted to number using Label Encoder and on its categorical encoding is
applied.
* The batch size is 32 and number of epochs is 100

## TRAINING AND TESTING EMBEDDINGS ON SVM CLASSIFIER
* The train and test embeddings obtained from the above are scaled using
StandardScaler preprocessing transform for faster convergence.
* Grid Search CV is used to tune the hyperparameters i.e. C, kernel and gamma.
* The model trained from the above is used to predict on unseen test set.
* Based on classification report, precision, recall and f1-score can be calculated for
each of the class label.
* Based on confusion matrix, accuracy scores for each of the class label can be
calculated.
