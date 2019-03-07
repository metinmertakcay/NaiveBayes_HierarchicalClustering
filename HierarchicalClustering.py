# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:04:32 2018
@author: Metin Mert Akçay
"""
import sys

def createAttributeValueAndLabels(line):
    attributeValues = line.split(',')
    label = attributeValues[-1].rsplit('\n')[0]
    del attributeValues[-1]
    return attributeValues, label

# Read data in txt file and prepare it for train and test
def readSamplesAndLabels():
    samples = []
    labels  = []
    with open("missing_outlier_somefeatures_removed.txt", "r") as file:
        for line in file:
            sample, label = createAttributeValueAndLabels(line)
            labels.append(label)
            samples.append(sample)
    return samples, labels

# add samples related cluster
def combineCentroids(centroids, row, column, size):
    i = 0
    while i < size:
        if row in centroids[i]:
            if column in centroids[i]:
                return False
            else:
                j = 0
                while j < size:
                    if column in centroids[j]:
                        for k in range(0, len(centroids[j])):
                            centroids[i].append(centroids[j][k])
                        centroids[i].sort()
                        del centroids[j]
                        return True
                    j += 1
        i += 1

# check two sample in the same cluster
def isNotSameCentroid(centroids, row, column):
    for i in range(0, len(centroids)):
        if row in centroids[i] and column in centroids[i]:
            return False
    return True

# find min distance in distance matrix.
def findMinDistanceInDistanceMatrix(distanceMatrix, size, centroids):
    minValue, row, column = distanceMatrix[0][0], 0, 0 
    for i in range(0, size):
        for j in range(0, size):
            if i < j:
                if minValue > distanceMatrix[i][j]:
                    if isNotSameCentroid(centroids, i, j):
                        minValue = distanceMatrix[i][j]
                        row = i
                        column = j
                    else:
                        distanceMatrix[i][j] = sys.maxsize
    return row, column

def createAllCentroids(size):
    centroids = []
    for i in range(0, size):
        centroids.append([i])
    return centroids 

# assign samples to clusters
def applySingleLinkageAgglomerative(distanceMatrix, size):
    centroids = createAllCentroids(size)
    
    while len(centroids) != 2:
        row, column = findMinDistanceInDistanceMatrix(distanceMatrix, size, centroids)
        combineCentroids(centroids, row, column, len(centroids))
        distanceMatrix[row][column] = sys.maxsize
    return centroids

# Haming distance used for nominal attribute, Euclidean distance used for numeric attribute and the distance of numeric attribute normalized.
def measureDistance(sampleI, sampleII, dictionary):
    distance = 0
    for i in range(0, len(sampleI)):
        try:
            float(sampleI[i])
            distance += abs(float(sampleI[i]) - float(sampleII[i])) / (dictionary[i]['max'] - dictionary[i]['min'])
        except ValueError:
            if sampleI[i] == sampleII[i]:
                distance += 0
            else:
                distance += 1
    return distance

# find min and max value for numeric attribute.
def findMinAndMaxValues(samples):
    attributeInfos = []
    for i in range(0, len(samples[0])):
        attributeInfos.append([])

    for smpl in samples:
        for i in range(0, len(smpl)):
            attributeInfos[i].append(smpl[i])
    
    dictionary = {}
    for i in range(0, len(attributeInfos)):
        try:
            float(attributeInfos[i][0])
            dictionary.update({i : {'min' : min(list(map(float, attributeInfos[i]))), 'max' : max(list(map(float, attributeInfos[i])))}})
        except ValueError:
            pass
    
    return dictionary 

# There are two labels. '+' and '-'. Find predicted label for all samples.
def determineCentroidsLabel(centroids, labels):
    centroids.sort(key=len, reverse=True)
    
    p_count = 0
    n_count = 0
    predictedLabel = []
    for i in range(0, len(centroids[0])):
        if labels[centroids[0][i]] == '+':
            p_count += 1
        else:
            n_count += 1
    
    if p_count < n_count:
        predictedLabel.append('-')
        predictedLabel.append('+')
    else:
        predictedLabel.append('+')
        predictedLabel.append('-')
        
    return predictedLabel

# Predicted and real label compared. Count is increased, if the predicted and real label is equal
def findAccuracy(centroids, predictedLabel, labels):
    accuracy = 0
    for i in range(0, len(centroids)):
        for j in range(0, len(centroids[i])):
            if predictedLabel[i] == labels[centroids[i][j]]:
                accuracy += 1
    return accuracy * 1.0 / len(labels)

def printCluster(centroids):
    print("Cluster0 <--" + str(len(centroids[0])))
    print("Cluster1 <--" + str(len(centroids[1])) + "\n")

def hierarchicalClustering(samples, labels):
    size = len(samples)
    dictionary = findMinAndMaxValues(samples)
    
    distanceMatrix = []
    for i in range(0, size):
        row = []
        for j in range(0, size):
            if i < j:
                distance = measureDistance(samples[i], samples[j], dictionary)
                row.append(distance)
            else:
				# max number add for prevent to select it
                row.append(sys.maxsize)
        distanceMatrix.append(row)
    centroids = applySingleLinkageAgglomerative(distanceMatrix, size);
    printCluster(centroids)
    predictedLabel = determineCentroidsLabel(centroids, labels)
    accuracy = findAccuracy(centroids, predictedLabel, labels)
    return accuracy    

if __name__ == '__main__':
    samples, labels = readSamplesAndLabels()
    acc = hierarchicalClustering(samples, labels)
    print("Incorrectly Clustered Instances Ratio: " + str(100 - round(acc * 100, 4)) + " % ")