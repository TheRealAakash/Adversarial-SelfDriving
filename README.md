# Adversarial-SelfDriving
[![Watch the video](https://www.youtube.com/watch?v=cdovxn3BRM4)]

With the rise of self-driving cars, Artificial Intelligence is becoming increasingly prevalent. In particular, self-driving cars rely on a type of Artificial Intelligence called convolutional neural networks, which allow an AI to see. Due to the increase in the prevalence of Convolutional Neural Networks, it is essential to ensure these critical AI applications are resilient to making errors. However, current Convolutional Neural Networks require tremendous amounts of data to classify images successfully. In practice, acquiring such large amounts of data is often impractical. As a result, most current Computer Vision algorithms are vulnerable to errors due to minor environmental variations. These errors can have disastrous consequences with critical applications such as self-driving cars.

The Adversarial Self-Driving framework seeks to improve the reliability of these AI applications that are becoming increasingly important to the world. The objective of this framework is to increase the reliability of Convolutional Neural Networks without requiring additional data over current methods. The framework achieves this objective using three components: a classifier, a generator, and a discriminator.

The classifier is the Convolutional Neural Network responsible for making decisions on data. Labeled images from a base dataset initially train the classifier. Next, the classifier is iteratively trained on minor variations of these base images as modified by the generator.

The generator makes modifications with two objectives: reducing the accuracy of the classifier and ensuring that new images reflect real-world variations. The generator achieves these objectives by training using a custom loss function, which determines loss based on the change in accuracy of the classifier and the probability of the modification occurring in the real world.

This probability is determined by the discriminator, which is trained on images from the original dataset and images modified by the generator to determine the probability of a new image being modified, enabling the discriminator to ensure that the images modified by the generator are realistic.

In the adversarial self-driving framework, an iterative process occurs. The generator identifies new situations that cause the classifier to make errors and generates additional examples of these situations. These examples are used to retrain the classifier until it can accurately classify all realistic images modified by the generator.

The adversarial self-driving framework was evaluated in a classification, simulation, and real-world setting, in comparison to baseline models in each setting. These settings were chosen to represent the most common deployments of Convolutional Neural Networks.

In the classification setting, the dataset used contained 50 thousand images of 43 types of traffic signs in varying visibility conditions. In this setting, the Adversarial Self-Driving framework achieved a substantial increase in reliability over the baseline model, achieving an accuracy of 99.7% compared to the baseline’s 96% accuracy, as well as less noise in its confusion matrix.

In the simulation setting, the model was evaluated in the Carla self-driving environment, an environment designed to assess the performance of various self-driving algorithms. In this setting, the model's objective was to drive in a simulated city under varied visibility conditions while avoiding collisions. Four hundred hours of driving data, with image data mapped to actions, was used to train the model. In this setting, the adversarial self-driving framework achieved higher reliability in poor visibility conditions, for instance, during rain or fog, having no collisions in 30 minutes of driving compared to the baseline model’s six collisions.

In the real-world setting, the objective was to drive around a track and follow directions on display. In this setting, the baseline model was unable to generalize to the environment due to image noise introduced by the camera and limited training data. In contrast, the adversarial self-driving framework was resilient to camera noise and successfully followed directions on the signs, substantially increasing performance over the baseline. This increase in performance can primarily be attributed to the generator identifying the variations in images that caused the classifier to make errors, and by generating additional examples, the generator enabled the classifier to become resilient to these errors.

Ultimately, the Adversarial Self-driving framework is able to improve the reliability of the numerous critical applications of AI that are becoming increasingly important to the world.

