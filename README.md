# AI-nm-project
Generate Realistic Human Face using GAN

The project “GAN model for face generation” employs Generative Adversarial Networks (GANs) to create lifelike images of human faces. GANs consist of two neural networks—the generator and the discriminator—that work in tandem. The generator generates images resembling human faces, while the discriminator evaluates these images, aiming to distinguish real faces from generated ones. Through continuous competition and improvement, the generator learns to produce increasingly authentic facial images, while the discriminator gets better at identifying fakes. This project involves training the GAN model on a dataset of human faces to generate new and realistic face images, contributing to advancements in artificial intelligence and image generation technology.

Why Use GANS ? It has been noticed most of the mainstream neural nets can be easily fooled into misclassifying things by adding only a small amount of noise into the original data. Surprisingly, the model after adding noise has higher confidence in the wrong prediction than when it predicted correctly. The reason for such an adversary is that most machine learning models learn from a limited amount of data, which is a huge drawback, as it is prone to overfitting. Also, the mapping between the input and the output is almost linear. Although it may seem that the boundaries of separation between the various classes are linear, in reality, they are composed of linearities and even a small change in a point in the feature space might lead to misclassification of data.

STEPS TAKEN IN GAN:

The generator takes in random numbers and returns an image.

This generated image is fed into the discriminator alongside a stream of images taken from the actual, ground-truth dataset.

The discriminator takes in both real and fake images and returns probabilities, a number between 0 and 1, with 1 representing a prediction of authenticity and 0 representing fake.

The discriminator is in a feedback loop with the ground truth of the images, which we know.

The generator is in a feedback loop with the discriminator.

Dataset: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

Architecture Selection:

Generator Architecture: Input layer: latent vector of shape (LATENT_DIM,) Dense layer: 1281616 units, LeakyReLU activation, reshaping to (16,16,128) Convolutional Transpose Layers: Conv2DTranspose Final Convolutional layer: channels(RGB) using Conv2D, tanh activation function

Discriminator Architecture: Input layer: shape(Height,width,channels) Convolutional layers: LeakyReLU activation Flatten layer and Dropout Layer Dense layer: sigmoid activation

TRAINING THE GAN MODEL:

Training is the hardest part and since a GAN contains two separately trained networks, its training algorithm must address two complications:

GANs must juggle two different kinds of training (generator and discriminator).
GAN convergence is hard to identify. As the generator improves with training, the discriminator performance gets worse because the discriminator can’t easily tell the difference between real and fake. If the generator succeeds perfectly, then the discriminator has a 50% accuracy. In effect, the discriminator flips a coin to make its prediction. This progression poses a problem for convergence of the GAN as a whole: the discriminator feedback gets less meaningful over time. If the GAN continues training past the point when the discriminator is giving completely random feedback, then the generator starts to train on junk feedback, and its quality may collapse.
LOSS OPTIMIZATION:

1.DISCRIMINATOR LOSS: Real Images: The discriminator aims to predict a label close to 1 (real) for actual images. Generated Images: It aims to predict a label close to 0 (fake) for generated images.

ADVERSARIAL LOSS (GAN loss): The generator aims to create images that deceive the discriminator into predicting them as real (label close to 1). To achieve this, during training, the generator's goal is to minimize the loss associated with generated images being labeled as fake (0), encouraging the discriminator to output high values for these generated images.

3.OPTIMIZATION: Lr=0.0001, the RMSprop uses squared gradients to adjust the learning rate Clipvalue= 1.0, helps control the magnitude of the gradients during training Decay rate= 1e-8

HYPERPARAMETER TUNING:

1.LATENT_DIM: Dimensionality of the latent space. It defines the size of the input noise vector fed to the generator. In the code, LATENT_DIM is set to 32. 2.CHANNELS: Number of channels in the images (e.g., RGB images have 3 channels). This parameter defines the number of channels in the generator's output and discriminator's input layers. In the code, CHANNELS is set to 3, indicating RGB color images. 3.Learning Rate: The rate at which the model's parameters are updated during training. In the provided code, the learning rate for the optimizer RMSprop is set to 0.0001.

Systematic Search: Implement a grid search or random search by defining a range of values for the hyperparameters (LATENT_DIM, CHANNELS, learning rate, etc.). Systematically loop through different combinations, train the GAN, and evaluate the generated images using appropriate evaluation metrics like Inception Score (IS), Frechet Inception Distance (FID), or visual inspection to identify the best-performing configuration.

Output:

Screenshot 2023-12-03 215657

ref: https://www.kaggle.com/code/nagessingh/generate-realistic-human-face-using-gan/notebook
