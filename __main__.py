from typing import List
import time
import numpy as np
import math
import random
from numba import vectorize, cuda, float32, float64, guvectorize, jit
import pickle
from os import path

MAXN = 64
INPUTSIZE = 784
OUTPUTSIZE = 10
OUTPUTS_TO_HIDDEN1 = 10
FLOAT = np.float32


class Mat:
    """
  static and useful methods
  """

    @staticmethod
    def drawWeight():
        return (np.random.random_sample() - 0.5) * 2

    @staticmethod
    def drawFloat():
        return np.random.random_sample()

    @staticmethod
    def drawInt(max=1):
        return np.random.randint(0, int(max))

    @vectorize(['float32(float32, float32)'], target='cpu')
    def add(a, b):
        a = a + b
        return a

    @vectorize(['float32(float32, float32)'], target='cpu')
    def sub(a, b):
        a = a - b
        return a

    @staticmethod
    def desiredIndex(desiredOutput):
        for i in range(0, len(desiredOutput)):
            if desiredOutput[i] == 1:
                return i
        return 0

    @staticmethod
    def checkResult(lastLayer, desiredIndex):
        return lastLayer[0][desiredIndex]

    @staticmethod
    def findElite(scores):  # find best individuals [returns positions of leaders]
        m = max(scores)
        return [i for i, j in enumerate(scores) if j == m]

    @staticmethod
    def draw():
        randLiczba = (random.random() - 0.5) * 2  # liczba <0,1)
        return randLiczba

    @staticmethod
    @guvectorize([(float32[:], float32[:, :], float32[:])], '(l),(l,n)->(n)', target='cpu')
    def neuronSoft(A, B, out):
        sum = 0.00001
        for j in range(len(out)):
            tmp = 0.
            for k in range(len(A)):
                tmp += A[k] * B[k, j]
            out[j] = 1 / (1 + math.exp(-12 * ((tmp / 13) - 0.5)))  # activation function
            sum += out[j]
        for j in range(len(out)):
            out[j] = out[j] / sum

    # counting values of layer:
    @guvectorize([(float32[:, :], float32[:, :], float32[:, :])], '(m,l),(l,n)->(m,n)', target='cpu')
    def neuron(A, B, out):
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                tmp = 0.
                for k in range(A.shape[1]):
                    tmp += A[i, k] * B[k, j]
                out[i, j] = tmp / 100  # activation function


class Network:
    # defines body of individual
    dtype = np.float32
    Input = np.empty(INPUTSIZE, dtype=FLOAT)
    DesiredOutput = np.empty(OUTPUTSIZE, dtype=FLOAT)
    desiredIndex = -1

    def __init__(self, numoflayers, _layers, child=False):
        self.Connections = []  # weights matrix
        self.Output = []
        self.FitResults = np.zeros(shape=(OUTPUTSIZE), dtype=FLOAT)
        self.fit = 0.0  # network fitness
        self.fitErr = 0.0  # fitness error
        self.ans = 0  # answer showing if it is the expected digit
        self.ansErr = 0  # answer error (answer complement)
        self.ansErrSqRoot = 0  # answer square root error ( calculated from formula)
        self.errorIndex = -1
        self.errorFitness = 0
        self.result = 0
        self.phase = 0
        if not (child):
            for i in range(0, numoflayers - 1):
                connectionMatrix = np.empty(shape=(_layers[i + 1], _layers[i]), dtype=self.dtype)
                # randomRange = .2  # 1.
                seedt = int(time.time()) % 100
                np.random.seed(seed=seedt)
                for m in range(0, _layers[i + 1]):
                    for n in range(0, _layers[i]):
                        connectionMatrix[m][n] = Mat.draw()
                self.Connections.append(connectionMatrix)
        else:
            self.Connections = []

    def fitness(self, Layers, desiredIndex):
        max = 0
        maxIndex = -1
        errorIndex = -1
        for i in range(0, len(Layers[0][0])):
            if i != desiredIndex:
                if max < Layers[0][0][i]:
                    max = Layers[0][0][i]
                    maxIndex = i
        score = Layers[0][0][desiredIndex]
        if ((max + 0.05) >= score):
            errorIndex = maxIndex
            if (score > max):
                score = 1.5 * score
                max -= 0.2 * max
                if (score > 1): score = 1
        else:
            score = 1.5 * score
            max -= 0.2 * max
            if (score > 1): score = 1
        err = abs(1 - score + max)
        if err < 0.005:
            err = 0.005
        self.fit = (1 / err)
        self.errorIndex = errorIndex
        return (self.fit, errorIndex)

    def fitnessSoft(self, Layers, desiredIndex):
        max = 0
        maxIndex = -1
        sum = 0.00001
        for i in range(0, len(Layers[0][0])):
            sum += Layers[0][0][i]
            if i != desiredIndex:
                if max < Layers[0][0][i]:
                    max = Layers[0][0][i]
                    maxIndex = i
        result = Layers[0][0][desiredIndex]
        if result > max:
            result = result * 2
            maxIndex = -1
        fit = np.e ** ((result * 2) / sum) - 0.999
        self.FitResults[desiredIndex] = fit
        return (fit, maxIndex)

    def flow(self, Layers):  # calculates network score
        for i in range(len(self.Connections) - 1, -1, -1):
            if i == 0:
                Mat.neuronSoft(Layers[i + 1], self.Connections[i], Layers[i])
            else:
                Mat.neuron(Layers[i + 1], self.Connections[i], Layers[i])
        self.Output = Layers[0]

    def cacheData(self):
        with open('cache.bin', 'wb') as _file:
            content = [
                self.exampleInputs,
                self.exampleOutputs,
                self.numberOfExamples,
                self.exampleInputst,
                self.exampleOutputst,
                self.numberOfExamplest]
            pickle.dump(content, _file)


class AI:  # class for model training and testing
    population = []  # creating population of N individuals
    maxn = 0
    # One indvidual at the start has:
    # array of connections
    e = np.float32(np.e)
    Layers = []
    index = 0
    dtype = np.float32
    target = 'cpu'
    saveName = 'Weights.bin'
    numberOfExamples = 0
    exampleInputs = []
    exampleOutputs = []
    numberOfExamplest = 0
    exampleInputst = []
    exampleOutputst = []
    accuracy = 0

    def learnAfterTest(self):  # training on whole iteration
        good = 0
        all = 0
        classCounts = np.zeros(shape=(10, len(self.population)), dtype=np.int)
        classValues = np.zeros(shape=(10, len(self.population)), dtype=np.float32)
        populationFitness = np.zeros(len(self.population), dtype=np.float32)
        start = random.randint(0, int(self.numberOfExamples - 100))
        for z in range(0, len(self.population)):
            for i in range(start, start + 100):
                desiredIndex = 0
                for desiredIndex in range(10):
                    if self.exampleOutputs[i][0, desiredIndex] == 1: break
                self.setInputOutput(i)

                self.population[z].flow(self.Layers)
                maxValue = self.Layers[0][0, 0]
                maxInd = 0
                for k in range(1, 10):
                    if self.Layers[0][0, k] > maxValue:
                        maxValue = self.Layers[0][0, k]
                        maxInd = k
                (result, error) = self.population[z].fitnessSoft(self.Layers, desiredIndex)
                classValues[desiredIndex][z] += result
                classCounts[desiredIndex][z] += 1

        scores = []

        for z in range(0, len(self.population)):
            counts = 0
            for i in range(10):
                populationFitness[z] += classValues[i][z]
                counts += classCounts[i][z]
            value = populationFitness[z] / counts
            scores.append(value)
        parentArr = []
        newPopulation = []
        for i in range(int(self.maxn / 8)):
            parentArr.append(self.select_parents_for_crossover(scores, self.population))
        howMany = int(self.maxn / 2)
        for i in range(0, howMany):
            chosen = self.select_one(scores, 20)
            parent1 = parentArr[i % int(self.maxn / 8)][0]
            parent2 = parentArr[i % int(self.maxn / 8)][1]
            for c in range(len(chosen[0].Connections)):
                for k in range(0, len(chosen[0].Connections[c][:])):
                    if Mat.drawWeight() > 0:
                        chosen[0].Connections[c][k, :] = parent1.Connections[c][k, :]
                    else:
                        chosen[0].Connections[c][k, :] = parent2.Connections[c][k, :]
            self.mutateAll(chosen[0], scores[chosen[1]])
            newPopulation.append(chosen[0])
            del self.population[chosen[1]]
            del scores[chosen[1]]
        for parent in parentArr:
            self.population.append(parent[0])
            self.population.append(parent[1])
        for child in newPopulation:
            self.population.append(child)

    def __init__(self, _layers):
        self.maxn = MAXN
        self.layers = _layers
        if path.isfile('cache.bin'):
            with open('cache.bin', 'rb') as _file:
                content = pickle.load(_file)
                self.exampleInputs = content[0]
                self.exampleOutputs = content[1]
                self.numberOfExamples = content[2]
                self.exampleInputst = content[3]
                self.exampleOutputst = content[4]
                self.numberOfExamplest = content[5]
        # common part:
        for i in range(0, len(_layers)):
            newVec = np.empty(shape=(1, _layers[i]), dtype=self.dtype)
            self.Layers.append(newVec)
        # intialization of individuals:
        for individual in range(0, self.maxn):
            self.population.append(Network(len(self.Layers), _layers))

    def teach(self):  # main training function
        # setup:
        plik = open('result.csv', 'w')
        desiredOutput = np.zeros(self.layers[0], dtype=self.dtype)
        j = 0
        bestPerformance = 0
        np.set_printoptions(precision=4, suppress=True)
        repeats = 0
        while True:
            repeats += 1
            iter_no = 0
            error = 0
            errorAvg = 0
            thisErr = 0
            errorIT = []
            scores = []
            classErr = np.zeros(self.layers[0], dtype=self.dtype)
            classIters = np.zeros(self.layers[0], dtype=int)
            indexes = np.random.randint(0, self.numberOfExamples - 2000)
            for _o in range(indexes, indexes + 2000):  # after every 'o' from training set an epoch
                iter_no += 1
                desiredOutput = self.setInputOutput(
                    _o)  # sets desired score and sets input to AI object
                currentIndex = Mat.desiredIndex(desiredOutput)
                if (currentIndex == 5):
                    toRange = random.randint(1, 4)
                elif (currentIndex == 8):
                    toRange = random.randint(1, 3)
                elif (currentIndex == 3) or (currentIndex == 9) or (currentIndex == 2):
                    toRange = random.randint(1, 2)
                else:
                    toRange = 1
                for obj in range(0, toRange):
                    scores = []
                    errorIT = []
                    errSum = 0
                    for individual in range(0, self.maxn):
                        self.population[individual].flow(self.Layers)  # calc score of this network
                        (thisFitness, errorIndex) = self.population[individual].fitness(self.Layers, currentIndex)
                        scores.append(thisFitness)
                        errorIT.append(errorIndex)
                        isTrue = Mat.checkResult(self.Layers[0], currentIndex)
                        thisErr = (isTrue - 1) ** 2  # error function
                        errSum += thisErr
                        classErr[currentIndex] += thisErr
                        classIters[currentIndex] += 1
                    elite = Mat.findElite(scores)
                    elitePop = self.population[elite[0]]
                    afitness = scores[elite[0]]
                    # Scores:
                    self.population[elite[0]].flow(self.Layers)
                    isTrue = Mat.checkResult(self.Layers[0], currentIndex)
                    thisErr = (isTrue - 1) ** 2  # error function
                    errorAvg = errSum / sum(classIters)
                    if iter_no % 100 == 0:  # and obj == 1:
                        print("Repetition=", repeats - 1, ",Iteration=", iter_no, ",Accuracy=", self.accuracy, end="",
                              file=plik)
                        print("\n\nRepetition=", repeats - 1, ",Iteration=", iter_no)
                        print("Fitnes=%0.4f" % (scores[elite[0]]))
                        print("Class=", currentIndex, ",ClassError=", elitePop.errorIndex)
                        print(self.population[elite[0]].Output)
                        print("Neuron=%0.4f" % (isTrue))
                        print("Avg Err=%0.4f" % (errorAvg))
                        classSum = 0
                        sumIt = 0
                        print("[", end="")
                        for it in range(0, self.layers[0]):
                            if classIters[it] != 0:
                                thisClass = classErr[it] / classIters[it]
                                print("%0.4f" % (thisClass), end=" ")
                                classSum += thisClass
                                sumIt += 1
                            else:
                                print("  ", end="")
                        print("]")
                        print("All Class Avg Error= %0.6f" % (classSum / sumIt))
                        print(",ClassErr=", (classSum / sumIt), file=plik)
                    # breeding:
                    # leftovers of population
                    newPopulation: List[Network] = []
                    newScores = []
                    newErrorIt = []
                    howMany = int(self.maxn / 2)
                    maxn = int(self.maxn)
                    for i in range(0, howMany):
                        chosen = self.select_one(scores, 20)
                        newIndex = len(newPopulation)
                        newPopulation.append(chosen[0])
                        # newPopulation.append(chosen[0])
                        newScores.append(scores[chosen[1]])
                        newErrorIt.append(errorIT[chosen[1]])
                        del scores[chosen[1]]
                        errorIndex = errorIT[chosen[1]]
                        del errorIT[chosen[1]]
                        del self.population[chosen[1]]
                        # Crossover:
                        parentArr = self.select_for_crossover(scores, self.population)
                        parent1 = parentArr[0]
                        parent2 = parentArr[1]
                        myFitness = 0
                        if (scores[parent1[1]] < scores[parent1[1]]):
                            myFitness = scores[parent1[1]]
                        else:
                            myFitness = scores[parent2[1]]

                        # when 2 visible layers:
                        if len(parent1[0].Connections) == 1:
                            for k in range(0, len(chosen[0].Connections[0][:])):
                                if (k <= int(len(chosen[0].Connections[0][:]) * 0.25)):
                                    chosen[0].Connections[0][k, currentIndex] = parent1[0].Connections[0][
                                        k, currentIndex]
                                    if (errorIndex != -1) and (i % 3) == 1:
                                        chosen[0].Connections[0][k, errorIndex] = parent1[0].Connections[0][
                                            k, errorIndex]
                                elif (k <= int(len(chosen[0].Connections[0][:]) * 0.5)):
                                    chosen[0].Connections[0][k, currentIndex] = parent2[0].Connections[0][
                                        k, currentIndex]
                                    if (errorIndex != -1) and (i % 3) == 2:
                                        chosen[0].Connections[0][k, errorIndex] = parent2[0].Connections[0][
                                            k, errorIndex]
                                elif (k <= int(len(chosen[0].Connections[0][:]) * 0.75)):
                                    chosen[0].Connections[0][k, currentIndex] = parent1[0].Connections[0][
                                        k, currentIndex]
                                    if (errorIndex != -1) and (i % 3) == 1:
                                        chosen[0].Connections[0][k, errorIndex] = parent1[0].Connections[0][
                                            k, errorIndex]
                                else:
                                    chosen[0].Connections[0][k, currentIndex] = parent2[0].Connections[0][
                                        k, currentIndex]
                                    if (errorIndex != -1) and (i % 3) == 2:
                                        chosen[0].Connections[0][k, errorIndex] = parent2[0].Connections[0][
                                            k, errorIndex]
                        # when there is more layers
                        elif len(parent1[0].Connections) == 2:
                            for k in range(0, len(chosen[0].Connections[0][:])):
                                if (k <= int(len(chosen[0].Connections[0][:]) * 0.25)):
                                    chosen[0].Connections[0][k, currentIndex] = parent1[0].Connections[0][
                                        k, currentIndex]
                                    if (errorIndex != -1) and (i % 3) == 1:
                                        chosen[0].Connections[0][k, errorIndex] = parent1[0].Connections[0][
                                            k, errorIndex]
                                elif (k <= int(len(chosen[0].Connections[0][:]) * 0.5)):
                                    chosen[0].Connections[0][k, currentIndex] = parent2[0].Connections[0][
                                        k, currentIndex]
                                    if (errorIndex != -1) and (i % 3) == 2:
                                        chosen[0].Connections[0][k, errorIndex] = parent2[0].Connections[0][
                                            k, errorIndex]
                                elif (k <= int(len(chosen[0].Connections[0][:]) * 0.75)):
                                    chosen[0].Connections[0][k, currentIndex] = parent1[0].Connections[0][
                                        k, currentIndex]
                                    if (errorIndex != -1) and (i % 3) == 1:
                                        chosen[0].Connections[0][k, errorIndex] = parent1[0].Connections[0][
                                            k, errorIndex]
                                else:
                                    chosen[0].Connections[0][k, currentIndex] = parent2[0].Connections[0][
                                        k, currentIndex]
                                    if (errorIndex != -1) and (i % 3) == 2:
                                        chosen[0].Connections[0][k, errorIndex] = parent2[0].Connections[0][
                                            k, errorIndex]

                            for k in range(0, len(chosen[0].Connections[1][:])):
                                if Mat.drawWeight() > 0:
                                    chosen[0].Connections[1][k, :] = parent1[0].Connections[1][k, :]
                                else:
                                    chosen[0].Connections[1][k, :] = parent2[0].Connections[1][k, :]
                        else:
                            for j in range(0, len(parent1[0].Connections)):
                                connectionMatrix = np.empty(
                                    shape=(parent1[0].Connections[j].shape[0], parent1[0].Connections[j].shape[1]),
                                    dtype=self.dtype)
                                for x in range(0, len(parent1[0].Connections[j])):
                                    if x < len(parent1[0].Connections[j]) / 2:
                                        for y in range(0, int(len(parent1[0].Connections[j][x]) / 2)):
                                            connectionMatrix[x][y] = parent1[0].Connections[j][x][y]
                                        for y in range(int(len(parent1[0].Connections[j][x]) / 2),
                                                       len(parent1[0].Connections[j][x])):
                                            connectionMatrix[x][y] = parent2[0].Connections[j][x][y]
                                    else:
                                        for y in range(0, int(len(parent1[0].Connections[j][x]) / 2)):
                                            connectionMatrix[x][y] = parent2[0].Connections[j][x][y]
                                        for y in range(int(len(parent1[0].Connections[j][x]) / 2),
                                                       len(parent1[0].Connections[j][x])):
                                            connectionMatrix[x][y] = parent1[0].Connections[j][x][y]
                                    chosen[0].Connections[j] = connectionMatrix

                    for i in range(howMany, maxn):
                        newPopulation.append(self.population[i - howMany])
                        newScores.append(scores[i - howMany])
                        newErrorIt.append(errorIT[i - howMany])
                    self.population = newPopulation
                    scores = newScores
                    errorIT = newErrorIt
                    # Mutating:
                    fErr = 1 / afitness
                    if len(chosen[0].Connections) > 1:
                        self.mutateMany(fErr, errorIT[newIndex], currentIndex)
                    else:
                        self.mutate(fErr, errorIT[newIndex], currentIndex)
                    ## copying best individual
                    if (((iter_no + 1) % 1000) == 0) and obj == 0:
                        # Testing:
                        performance = self.test(elitePop)  # on test set
                        # print("Test Score=",performance)
                        if performance > 0.82:
                            print("Success :)")
                            # saving weights of one layer - > score:
                            f = open(self.saveName, "wb")
                            pickle.dump(elitePop.Connections, f)
                            f.close()
                            plik.close()
                            return
                    if (((iter_no + 1) % 10) == 0) and obj == 0:
                        self.learnAfterTest()

    def select_one(self, wyniki, MAXFITNESS):
        """"Select individuals for crossover based on fitness."""
        # sort fitness tuples
        antiScore = []
        for i in range(0, len(wyniki)):
            antiScore.append(1 / wyniki[i])
        population_fitness = antiScore
        fitness_sum = float(sum(population_fitness))
        if fitness_sum == 0:
            returned = self.population[0]
            return (returned, 0)  ##error
        relative_fitness = [f / fitness_sum for f in population_fitness]
        probabilities = []
        sum = 0
        for i in range(0, len(relative_fitness)):  # distribution
            sum += relative_fitness[i]
            probabilities.append(sum)
        probabilities[len(relative_fitness) - 1] = 1  # distribution correction if num error
        while True:
            r = random.random()
            if (r > probabilities[len(self.population) - 1]): r = probabilities[len(self.population) - 1]
            for (i, individual) in enumerate(self.population):
                if r <= probabilities[i]:
                    returned = individual
                    return (returned, i)

    def select_parents_for_crossover(self, scores, population):
        parentIndex1 = np.where(scores == np.max(scores))[0][0]
        buf = scores[parentIndex1]
        scores[parentIndex1] = -1.0
        parentIndex2 = np.where(scores == np.max(scores))[0][0]
        scores[parentIndex1] = buf
        parent1 = population[parentIndex1]
        parent2 = population[parentIndex2]
        chosen = []  # parents to return
        chosen.append((parent1))
        chosen.append((parent2))
        if parentIndex1 < parentIndex2:
            del population[parentIndex2]
            del population[parentIndex1]
            del scores[parentIndex2]
            del scores[parentIndex1]
        else:
            del population[parentIndex1]
            del population[parentIndex2]
            del scores[parentIndex1]
            del scores[parentIndex2]
        return chosen

    def select_for_crossover(self, scores, population):
        """"Select individuals for crossover based on fitness."""
        # sort fitness tuples
        population_fitness = scores
        fitness_sum = float(sum(population_fitness))
        relative_fitness = [f / fitness_sum for f in population_fitness]
        probabilities = []
        suma = 0
        for i in range(0, len(relative_fitness)):  # distribution
            suma += relative_fitness[i]
            probabilities.append(suma)
        probabilities[len(relative_fitness) - 1] = 1  # distribution correction if num error

        chosen = []  # parents to return
        number = 2  # number of individuals to be chosen
        for n in range(number):
            r = random.random()
            if (r > probabilities[len(population) - 1]): r = probabilities[len(population) - 1]
            for i in range(0, len(population)):
                individual = population[i]
                if r <= probabilities[i]:
                    chosen.append((individual, i))
                    break
        if (len(chosen) < 2):
            print("Error in corssover")
            chosen.append(population[0], 0)
        return chosen

    def mutateAll(self, luckyOne: Network, score):
        blad = (2.2 - self.accuracy) / (score + 0.0001)
        if Mat.drawWeight() > self.accuracy * 0.9:
            for c in range(0, len(luckyOne.Connections)):
                # mutate
                mutationTimes = np.random.randint(luckyOne.Connections[c].shape[0],
                                                  luckyOne.Connections[c].shape[0] + luckyOne.Connections[c].shape[1])
                mutationTimes = int(mutationTimes * blad)
                for i in range(0, mutationTimes):
                    randIndexX = np.random.randint(0, luckyOne.Connections[c].shape[0])
                    randIndexY = np.random.randint(0, luckyOne.Connections[c].shape[1])
                    luckyOne.Connections[c][randIndexX][randIndexY] = Mat.draw()

    def mutate(self, thisErr, errorIndex, currentIndex):
        # select for mutation:
        for luckyOneindex in (0, int(self.maxn / 2)):
            if Mat.drawWeight() < self.accuracy * 0.9: continue
            mutationTimes = int((abs(thisErr - self.accuracy / 2) * 1.8 + 0.3) / 0.01)
            for c in range(0, len(self.population[luckyOneindex].Connections)):
                # mutate
                for i in range(0, mutationTimes):
                    randIndexX = np.random.randint(0, self.population[luckyOneindex].Connections[c].shape[0])
                    randIndexY = currentIndex  # choosing column taking part in image detection
                    self.population[luckyOneindex].Connections[c][randIndexX][randIndexY] = Mat.draw()
                mutationTimes = int(mutationTimes * 0.8)
                # mutate
                if errorIndex != -1:
                    for i in range(0, mutationTimes):
                        randIndexX = np.random.randint(0, self.population[luckyOneindex].Connections[c].shape[0])
                        randIndexY = errorIndex  # choosing error column taking part in image detection
                        self.population[luckyOneindex].Connections[c][randIndexX][errorIndex] = Mat.draw()

    def mutateMany(self, thisErr, errorIndex, currentIndex):
        for (i, luckyOne) in enumerate(self.population):
            if i > int(self.maxn / 2): return
            if Mat.drawWeight() > 0.2: continue
            for c in range(0, len(luckyOne.Connections)):
                # mutate
                if (c != 0):
                    mutationTimes = np.random.randint(int(luckyOne.Connections[c].shape[0] / 4),
                                                      luckyOne.Connections[c].shape[0] * 2)
                    # * int(luckyOne.Connections[c].shape[1] / 2))
                    mutationTimes = int((thisErr * 1.5 + 0.3) * 0.5 * mutationTimes)
                else:
                    mutationTimes = int((thisErr * 1.5 + 0.3) * 0.2 * luckyOne.Connections[c].shape[0])

                for i in range(0, mutationTimes):
                    randIndexX = np.random.randint(0, luckyOne.Connections[c].shape[0])
                    if (c == 0):
                        randIndexY = currentIndex  # choosing column taking part in image detection
                    else:
                        randIndexY = np.random.randint(0, luckyOne.Connections[c].shape[1])
                    luckyOne.Connections[c][randIndexX][randIndexY] = Mat.draw()
                if (c == 0):
                    mutationTimes = int(mutationTimes * 0.8)
                    # mutate
                    if errorIndex != -1:
                        for i in range(0, mutationTimes):
                            randIndexX = np.random.randint(0, luckyOne.Connections[c].shape[0])
                            randIndexY = errorIndex  # choosing error column taking part in image detection
                            luckyOne.Connections[c][randIndexX][errorIndex] = Mat.draw()

    def test(self, network):  # testing on test set
        good = 0
        all = 0
        for i in range(self.numberOfExamplest):
            output = 0
            for output in range(10):
                if self.exampleOutputst[i][0, output] >= 0.9:
                    break
            self.setInputOutputt(i)
            network.flow(self.Layers)
            maxValue = self.Layers[0][0, 0]
            maxInd = 0
            for k in range(1, 10):
                if self.Layers[0][0, k] > maxValue:
                    maxValue = self.Layers[0][0, k]
                    maxInd = k
            all += 1
            if maxInd == output:
                good += 1
        print("\n\nAccuracy " + str(good / all))
        self.accuracy = good / all
        return good / all

    def setInputOutput(self, number):  # loads input for network and gets "perfect" (expected) score
        lastLayerIndex = len(self.Layers) - 1
        self.Layers[lastLayerIndex] = self.exampleInputs[
            number]  # put input data on first layer vertices
        return self.exampleOutputs[number][0]  # returns perfect

    def setInputOutputt(self, number): # same as above but for test set
        lastLayerIndex = len(self.Layers) - 1
        self.Layers[lastLayerIndex] = self.exampleInputst[
            number]
        return self.exampleOutputst[number][0]

    def cacheData(self):
        with open('cache.bin', 'wb') as _file:
            content = [
                self.exampleInputs,
                self.exampleOutputs,
                self.numberOfExamples,
                self.exampleInputst,
                self.exampleOutputst,
                self.numberOfExamplest]
            pickle.dump(content, _file)

    def readExamplesFile(self, images='train-images.idx3-ubyte', labels='train-labels.idx1-ubyte'):
        if self.numberOfExamples > 0:
            return
        with open(images, 'rb') as examples_file, open(labels, 'rb') as labels_file:
            order = 'big'
            size = 28
            int.from_bytes(examples_file.read(4), order)
            self.numberOfExamples = int.from_bytes(examples_file.read(4), order)
            print("Number of examples: " + str(self.numberOfExamples))
            print("size: " + str(int.from_bytes(examples_file.read(4), order)) + "x" + str(
                int.from_bytes(examples_file.read(4), order)))
            print("reading examples...")
            for i in range(self.numberOfExamples):
                self.exampleInputs.append(np.empty(shape=(1, size * size), dtype=self.dtype))
                for j in range(size * size):
                    self.exampleInputs[i][0, j] = int.from_bytes(examples_file.read(1), order) / 255
            st = ''
            for j in range(size * size):
                if j % size == 0:
                    st += '\n'
                if self.exampleInputs[0][0, j] == 0.:
                    st += '_'
                else:
                    st += 'O'
            print(st)
            int.from_bytes(labels_file.read(8), order)
            print("reading labels...")
            for i in range(self.numberOfExamples):
                self.exampleOutputs.append(np.zeros(shape=(1, 10), dtype=self.dtype))
                self.exampleOutputs[i][0, int.from_bytes(labels_file.read(1), order)] = 1.
            print("finished reading labels")

    def treadExamplesFile(self, images='train-images.idx3-ubyte', labels='train-labels.idx1-ubyte'):
        if self.numberOfExamplest > 0:
            return
        with open(images, 'rb') as examples_file, open(labels, 'rb') as labels_file:
            order = 'big'
            size = 28
            int.from_bytes(examples_file.read(4), order)
            self.numberOfExamplest = int.from_bytes(examples_file.read(4), order)
            print("Number of examples: " + str(self.numberOfExamplest))
            print("size: " + str(int.from_bytes(examples_file.read(4), order)) + "x" + str(
                int.from_bytes(examples_file.read(4), order)))
            print("reading examples...")
            for i in range(self.numberOfExamplest):
                self.exampleInputst.append(np.empty(shape=(1, size * size), dtype=self.dtype))
                for j in range(size * size):
                    self.exampleInputst[i][0, j] = int.from_bytes(examples_file.read(1), order) / 255
            st = ''
            for j in range(size * size):
                if j % size == 0:
                    st += '\n'
                if self.exampleInputst[0][0, j] == 0.:
                    st += '_'
                else:
                    st += 'O'
            print(st)
            int.from_bytes(labels_file.read(8), order)
            print("reading labels...")
            for i in range(self.numberOfExamplest):
                self.exampleOutputst.append(np.zeros(shape=(1, 10), dtype=self.dtype))
                self.exampleOutputst[i][0, int.from_bytes(labels_file.read(1), order)] = 1.
            print("finished reading labels")

    # ?
    def save(self):
        with open(self.saveName, 'wb') as save_file:
            pickle.dump(self.Connections, save_file)

    def getConnectionsFromFile(self):
        with open(self.saveName, 'rb') as read_file:
            Connections = pickle.load(read_file)
            self.elitePop = Network(len(self.Layers), self.layers)
            self.elitePop.Connections = Connections
            self.Layers = []
            i = 0
            for i in range(len(self.elitePop.Connections)):
                self.Layers.append(np.zeros((1, self.elitePop.Connections[i].shape[1]), dtype=self.dtype))
                self.Layers.append(np.zeros((1, self.elitePop.Connections[i].shape[0]), dtype=self.dtype))
            for i in self.population:
                i = self.elitePop


if __name__ == '__main__':
    # model intialization
    AI = AI([10, 784])  # 784=28px*28px
    # AI = AI([10, 240, 784])  # 784=28px*28px

    # load data from file if needed
    AI.treadExamplesFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    AI.readExamplesFile()
    # creating cache of data for speedup
    AI.cacheData()

    # teaching:
    # if weights have been previously set we can start with them ( instead of drawing new):
    # AI.getConnectionsFromFile()
    # print("Training in progress...")
    # AI.teach()
    # print("Training finished")

    # testing trained model:
    # loading weights:
    print("Testing trained network:")
    AI.getConnectionsFromFile()
    # testing on test set:
    AI.test(AI.elitePop)  # test set
