from scipy.spatial import distance
import random
import numpy as np


def readInput():
    with open('input.txt', 'r') as f:
        lines = f.readlines()
        numCities = int(lines[0].strip())
        cities = [tuple(map(int, line.strip().split())) for line in lines[1:numCities+1]]
    return cities, numCities


def writeOutput(bestSolution):
    totalDistance = bestSolution[2]
    tour = bestSolution[0]
    
    with open('output.txt', 'w') as f:
        f.write(f"{totalDistance:.3f}\n")
        for city in tour:
            f.write(f"{city[0]} {city[1]} {city[2]}\n")
        
        f.write(f"{tour[0][0]} {tour[0][1]} {tour[0][2]}\n")


def calculateDistance(array):
    totalDist = 0
    for i in range(len(array)):
        point1 = array[i]
        point2 = array[(i + 1) % len(array)]
        dist = distance.euclidean(point1, point2)
        totalDist += dist
    return round(totalDist, 3)


def createInitialPopulation(arr, populationSize):
    permutations = []
    for _ in range(populationSize):
        currentIndexes = random.sample(range(len(arr)), len(arr))
        currentArray = [arr[i] for i in currentIndexes]
        dist = calculateDistance(currentArray)
        fitness = (1 / dist)
        permutations.append([currentArray, fitness, dist])
    permutations.sort(key=lambda x: x[1], reverse=True)
    return permutations


def createMatingPool(initialPopulation):
    fitnessValues = [obj[1] for obj in initialPopulation]
    totalFitness = sum(fitnessValues)
    selectionProbability = [fitness / totalFitness for fitness in fitnessValues]
    selectedParents = []
    for _ in range(len(initialPopulation)):
        selectedIndex = np.random.choice(len(initialPopulation), p=selectionProbability)
        selectedParents.append(initialPopulation[selectedIndex])
    return selectedParents


def orderCrossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]
    ptr = (end + 1) % size
    for city in parent2:
        if city not in child:
            child[ptr] = city
            ptr = (ptr + 1) % size

    return child

def cycleCrossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    cycleStart = 0
    cycle = 0

    while None in child:
        if child[cycleStart] is None:
            indices = [cycleStart]
            val = parent1[cycleStart]
            nextIndex = parent2.index(val)
            while nextIndex != cycleStart:
                indices.append(nextIndex)
                val = parent1[nextIndex]
                nextIndex = parent2.index(val)
            cycle += 1
            for i in indices:
                if cycle % 2 == 1:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
        else:
            cycleStart = (cycleStart + 1) % size

    return child


def crossover(parent1, parent2):
    if random.random() < 0.5:
        child1 = orderCrossover(parent1, parent2)
        child2 = orderCrossover(parent2, parent1)
    else:
        child1 = cycleCrossover(parent1, parent2)
        child2 = cycleCrossover(parent2, parent1)
    return child1, child2

def swapMutation(child):
    idx1, idx2 = random.sample(range(len(child)), 2)
    child[idx1], child[idx2] = child[idx2], child[idx1]
    return child


def scrambleMutation(child):
    start, end = sorted(random.sample(range(len(child)), 2))
    sublist = child[start:end]
    random.shuffle(sublist)
    child[start:end] = sublist
    return child


def inversionMutation(child):
    start, end = sorted(random.sample(range(len(child)), 2))
    child[start:end] = reversed(child[start:end])
    return child


def mutate(child, stagnationCounter, mutationRate=0.1):
    if stagnationCounter > 5:
        mutationRate += min(0.1 * stagnationCounter, 0.5)
    else:
        mutationRate = 0.1
        
    if random.random() < mutationRate:
        mutationChoice = random.choice([swapMutation, scrambleMutation, inversionMutation])
        return mutationChoice(child)
    return child


def hasConverged(population, stagnationCounter, bestSolution, prevSolution):
    if bestSolution is not None and prevSolution == bestSolution[2]:
        stagnationCounter += 1
    else:
        stagnationCounter = 0
        
    return stagnationCounter, bestSolution




def createNewGeneration(population, eliteSize, stagnationCounter):
    population.sort(key=lambda x: x[1], reverse=True)
    newGeneration = population[:eliteSize]

    topPerformers = population[:100]

    
    parents = createMatingPool(topPerformers)
    for i in range(0, len(parents)-1):
        child1, child2 = crossover(parents[i][0], parents[i][0])

        child1 = mutate(child1, stagnationCounter)
        child2 = mutate(child2, stagnationCounter)

        dist1 = calculateDistance(child1)
        fitness1 = 1 / dist1
        dist2 = calculateDistance(child2)
        fitness2 = 1 / dist2

        newGeneration.append([child1, fitness1, dist1])
        newGeneration.append([child2, fitness2, dist2])
    
    return newGeneration[:len(population)]


def geneticAlgorithm(arr, numCities, populationSize, eliteSize):
    population = createInitialPopulation(arr, populationSize)
    if numCities <= 90:
        generations = 200
    elif numCities <= 150:
        generations = 150
    elif numCities <= 350:
        generations = 125
    else:
        generations = 80
    stagnationCounter = 0
    #print(len(population))
    bestSolution = None
    prevSolution = None
    for gen in range(generations):
        population = createNewGeneration(population, eliteSize, stagnationCounter)
        print(len(population))
        bestSolution = population[0]

        print(f"Generation {gen}: Best distance = {bestSolution[2]}")

        stagnationCounter, bestSolution = hasConverged(
            population, 
            stagnationCounter, 
            bestSolution,
            prevSolution
        )
        prevSolution = bestSolution[2]
        

    return bestSolution


if __name__ == "__main__":
    cities, numCities = readInput()
    bestSolution = geneticAlgorithm(cities,numCities, populationSize=500, eliteSize=15)
    writeOutput(bestSolution)
