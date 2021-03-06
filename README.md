# CEAL_MNIST
Cost Effective Active Learning on Mnist dataset with LeNet

MNIST:http://yann.lecun.com/exdb/mnist/

LeNet:http://yann.lecun.com/exdb/lenet/

# Environment
Pytorch 1.3.1

# Experiment Setting
I randomly selected 600 samples from each class (6000 total) to initialize the weights. With the same initial model, CEAL experiments are trained with parameters: delta_0 = 3e-7, dr = 1e-7, epochs = 20, t = 1 and k = 3000. 

# Result
![Alt text](ceal1.png)

![Alt text](ceal2.png)

AL_RAND: Randomly select samples with manual annotation. 

CEAL_EN: CEAL with entropy criteria

CEAL_MS: CEAL with marginal sampling criteria

CEAL_LC: CEAL with least confidence criteria

Uncertain_Only: Common active learning w/o high confidence sample pseudo-labeling

Comparison Results and Analysis:
  
  Starting from the same initialization, CEAL improves better than common active learning (CAL) and random samples(AL_RAND) at first (an average of 0.2% better than CAL and 0.4% better than AL_RAND). But they fluctuate a little comparing to CAL after first several iterations. At the end, CEAL_MS and CEAL_LC converge to 98.2% accuracy which is around 0.7% less than othere models.
