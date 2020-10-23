#COVID 19 Face Mask Detection using Pytorch Tutorial
###Author: Rutuja Kawade


Few Months back, in the lockdown phase, due to covid 19, we all were instructed to wear face masks to prevent community spread in the pandemic. It was a huge burden on the police force of India, to check whether people are following the guidelines issued or not. To help them out, I made a face mask detection project in pytorch. Pytorch is an awesome framework by Facebook AI Research. In this tutorial I will explain my approach to the problem mentioned above. So let’s begin!

1. Dataset: To train a deep learning model to classify whether a person is wearing a mask or not, we need to find a good dataset with a fair amount of images for both classes: Wearing Mask and Not Wearing Mask. Real World Masked Face Dataset (RMFD) provides just what we need! This dataset was created for facial recognition purposes. However, we’re going to use it for face mask detection.

2. Steps: The rest of this post is organized in the following way:
2.1. Data extraction
2.2. Building the Dataset class
2.3. Building our face mask detector model
2.4. Training our model
2.5. Testing our model on real data
2.6. Results
2.1. Data extraction
The RMFD provides 2 datasets:
Real-world masked face recognition dataset: it contains 5,000 masked faces of 525 people and 90,000 normal faces.
Simulated masked face recognition datasets.
In this experiment, we are going to use the first dataset. After downloading and unzipping the dataset, its structure looks as follows:

We create our pandas DataFrame by iterating over the images and assigning to each image a label of 0 if the face is not masked, and 1 if the face is masked. The images of this dataset are already cropped around the face, so we won’t need to extract the face from each image.
The following code illustrates the data extraction process:
2.2. Building the Dataset class
Now that we have our pandas DataFrame ready, it is time to build the Dataset class, which will be used for querying samples by batches in a way interpretable by PyTorch. Our model is going to take 100x100 images as input, so we transform each sample image when querying it, by resizing it to 100x100 and then convert it to a Tensor , which is the base data type that PyTorch can manipulate:
dataset module
2.3. Building our face mask detector mode
We’re going to be using PyTorch Lightning, which is a thin wrapper around PyTorch. PyTorch Lightning structures your code efficiently in a single class containing everything we need to define and train a model, and you can overwrite any method provided to your needs, making it easy to scale up while avoiding spaghetti code.
PyTorch Lightning exposes many methods for the training/validation loop. However, we are going to be using some of them for our needs. The following are the methods we’re going to override, and are going to be called in the following order internally:
1. Setup:
__init__()
prepare_data()
configure_optimizer()
train_dataloader()
val_dataloader()
2. Training loop:
training_step()
3. Validation loop:
validation_step()
validation_epoch_end()
2.3.1. Defining the model and the forward pass
To define our model, we subclass the LightningModule of PyTorch Lightning, and define our model architecture along with the forward pass. We’re going to keep it simple and use 4 convolution layers followed by 2 linear layers. We’re going to use ReLU as activation function, and the MaxPool2das the pooling layer. We then initialize the weights of these layers with xavier_uniform as this will make the network train better:
CNN model definition
2.3.2. Preparing the data for the model
Our dataset is imbalanced (5,000 masked faces VS 90,000 non-masked faces). Therefore, when splitting the dataset into train/validation, we need to keep the same proportions of the samples in train/validation as the whole dataset. We do that by using the train_test_split function of sklearn and we pass the dataset’s labels to its stratisfy parameter, and it will do the rest for us. We’re going to use 70% of the dataset for training and 30% for validation:
prepare_data() method
When dealing with unbalanced data, we need to pass this information to the loss function to avoid unproportioned step sizes of the optimizer. We do this by assigning a weight to each class, according to its representability in the dataset.
We assign more weight to classes with a small number of samples so that the network will be penalized more if it makes mistakes predicting the label of these classes. While classes with large numbers of samples, we assign to them a smaller weight. This makes our network training agnostic to the proportion of classes. A good rule of thumb for choosing the weight for each class is by using this formula:
class weight for imbalanced data
This translates to the following code:
CrossEntropyLoss with adapted classes weights
2.3.3. Data loaders
We’re going to define our data loaders that are going to be used for training and validation. We train our model using a batch of size 32. We shuffle our training batch samples each time so our model can train better by receiving data in a non-repetitive fashion. To reduce the time of loading the batch samples, which can be a bottleneck in the training loop, we set the number of workers to 4, this will perform multi-process data loading:
Train/validation data loaders
2.3.4. Configuring the optimizer
We define our optimizer by overriding the configure_optimizers() method and returning the desired optimizer. We are going to use Adam for the purpose of this post, and we fix the learning rate to 0.00001 :
configure_optimizers() method
2.3.5. Training step
In the training step, we receive a batch of samples, pass them through our model via the forward pass, and compute the loss of that batch. We can also log the loss, and PyTorch Lightning takes care of creating the log files for TensorBoard automatically for us:
training_step() method
2.3.6. Validation step
At the end of each training epoch, the validation_step() is called on each batch of the validation data, we compute the accuracy and the loss, and return them in a dictionary. These returned values will be used in the next section:
validation_step() method
2.3.7. Validation epoch end
In the validation_epoch_end() we receive all the data returned from validation_step() (from the previous section). We calculate the average accuracy and loss and log them so we can visualize them in TensorBoard later on:
validation_epoch_end() method
2.4. Training our model
To train our model we simply initialize our MaskDetector object and pass it to the fit() method of the Trainer class provided by PyTorch Lightning. We also define a model checkpointing callback, we want to save the model with the best accuracy and the lowest loss. We are going to train our model for 10 epochs:
Training model
We can see that the validation loss is decreasing across epochs:
And the validation accuracy of our model reaches its highest peak on epoch 8, yielding an accuracy of 99%.

After epoch 8 (where the red arrow is pointing) our model starts overfitting. Thus, the validation accuracy starts to degrade. So we’re gonna take the saved model of epoch 8 and use it for testing on real data!
2.5. Testing our model on real videos
To test our model on real data, we need to use a face detection model that is robust against occlusions of the face. Fortunately, OpenCV has a deep learning face detection model that we can use. This deep learning model is a more accurate alternative to the Haar-Cascade model, and its detection frame is a rectangle and not a square. Therefore, the face frame can fit the entirety of the face without capturing parts of the background, which can interfere with our face mask model predictions.
A good tutorial on how to use OpenCV’s deep learning face detection is the following:
Face detection with OpenCV and deep learning — PyImageSearch
To run inferences on a video, we’re going to use our saved model from the previous section, and process each frame:
Extract the faces
Pass them to our face mask detector model
Draw a bounding box around the detected faces, along with the predictions computed by our model.
The following is an extract of the processing video code:

2.6. Results: I asked a couple of friends to film themselves, put a mask on, and then take it off. These are the results! It looks like our model is working great even with a custom made mask! Our model’s weights file size is around 8 Mb, and the inferences on a CPU are near-real time.


