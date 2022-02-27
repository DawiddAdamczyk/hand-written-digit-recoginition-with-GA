# hand-written-digit-recoginition-with-GA
Evolutionary algorithm for recognition of hand written digits with individuals based on self implemented neural network. Tested and trained against the MNIST dataset, which contains 60,000 training data examples and a test dataset of 10,000 elements. The images are stored in grayscale and have dimensions of 28x28.

Built using Numba, Pickle and Pillow libraries.

Written as a part of "Artificial Intelligence" course taken at Gdańsk University of Technology, 2020.
# Status of project
This project has been completed. 
# Setup
In order to work requires [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to be put inside main directory.
```
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
```
Also requires cache.bin created from [NN recogintion repo](https://github.com/DawiddAdamczyk/hand-written-digit-recognition-with-NN).
# Genetic algorithm
## Description of used method
The algorithm used, which belongs to the group of evolutionary algorithms, is a kind of heuristics that search the space of alternative solutions to a problem in order to find the set of best solutions. The algorithm learns to classify into 10 disjoint classes - it has 10 resultant neurons. The algorithm for teaching a 1 layer network (no hidden layers) is 2 phase and consists of the steps shown below:

0. The initial population is drawn.
### Phase 1: Teaching of single class
1. 1000 consecutive learning images are selected from a random initial set element
2. Subsequent images are fed to the network with information about the selected expected class
3. The algorithm calculates the network responses for this image using NeuronSoft type activation function:
Sigmoid response of selected neuron / sum of all activation functions (Sigmoid) response for all classes. (This type of activation function is recommended for last layer neurons for multiple classifiers)
4. The algorithm calculates Fitness for the entire population of genes. This fitness is highly nonlinear which allows it to compute a distribution well preferring the highest probability scores as parents of the next population. Fitness additionally can return information about the class that has a better response of its neuron than that of the expected neuron.
```
self.fit=(1/err)
err=abs(1-score+max)
```
5. Parents drawn from the fitness distribution reproduce to form a progeny population in which only the genes for the neuron of the expected class and of the class that had a score higher than the expected neuron (if any) are changed.
6. Offspring go through a process of mutation, random changes in the genotype of an individual.
7. A new generation is created from the offspring population and parents optimizing the network response for one class
8. Once per N circuits Phase 2 is called
9. Return to para. 2.
### Phase 2: Multi-class learning based on results from multiple classes
1. The population of the network is given 200 consecutive learning images from a random initial element of the set.
2. The mean fitnesses of all classifications are calculated. FitnessSoft is used = e^(C * expected network response/ sum of all output neuron responses) - 1 :
```
fit=np.e**((result*2)/sum)-0.999
```
3. A population of parents is selected based on the averaged fitness of all classes. In this case, direct selection of the best candidates without using the distribution is used (due to the fact that it is too uniform after averaging).
4. Parents create new individuals, which are then subjected to mutation.

## Results
On a single-layer network, the best result obtained was 82% on the test set (the criterion for the end of the algorithm) after 19 iterations of Phase 1 (about 19,000 images (19 subsets with a random starting element and size of 1000 elements) from the learning set given to Phase 1. For Phase 2 this means giving 1900 random subsets of 100 consecutive images with a random starting element).

The graph shows the increment of classification correctness on the test set, depending on the number of images learned.
<img width="499" alt="image" src="https://user-images.githubusercontent.com/100523391/155898761-e5a3766b-f24b-485b-810b-27f0653a1c90.png">
