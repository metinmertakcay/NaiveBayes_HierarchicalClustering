"""
Created on Mon Dec 10 20:04:32 2018
@author: Metin Mert Akçay
"""
from statistics import stdev
import math

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

def calculateMean(attributeList): 
    return sum(attributeList) / len(attributeList)

def calculateStandardDeviation(attributeList):
    return stdev(attributeList)

def calculateGaussDistribution(x, mean, stder):
    exponent = math.exp(-(math.pow(x - mean, 2) * 1.0 / (2 * math.pow(stder,2))))
    return (1 / ((math.sqrt(2 * math.pi)) * stder)) * exponent
    
def findClassProbablity(labels):
    positive = labels.count('+') / len(labels)
    negative = labels.count('-') / len(labels)
    return positive, negative

# Find unique nominal values.
def uniqueElement(attributeValues): 
    uniqueList = [] 
    for x in attributeValues: 
        if x not in uniqueList: 
            uniqueList.append(x) 
    return uniqueList

# Mean and standard deviation for numeric values ​​were calculated. Number of occurrences for nominal values ​​was calculated.
def prepareAttributeForTesting(index, train):
    attributesInfos = []
    for i in range(0, len(train[0])):
        attributesInfos.append([])

    for idx in index:
        for i in range(0, len(train[0])):
            attributesInfos[i].append(train[idx][i])

    for i in range(0, len(attributesInfos)):
		# Check attribute is numeric or nominal
        try:
            float(attributesInfos[i][0])
            dictionary = {'mean' : calculateMean(list(map(float, attributesInfos[i]))),
                          'stder': calculateStandardDeviation(list(map(float, attributesInfos[i])))}
            attributesInfos[i] = dictionary
        except ValueError:
            uniqueList = uniqueElement(attributesInfos[i])
            dictionary = {}
            for j in range(0, len(uniqueList)):
                dictionary.update({uniqueList[j] : attributesInfos[i].count(uniqueList[j])})
            dictionary.update({'total' : len(attributesInfos[i])})
            attributesInfos[i] = dictionary
    return attributesInfos    

def findPredictedLabel(positive, positiveClassProbability, negative, negativeClassProbability, test):
    predicted = []
    
    for smpl in test:
        positiveProbability = positiveClassProbability
        negativeProbability = negativeClassProbability
		# Calculate positive and negative class label for test data. And determine a label for data.
        for i in range(0, len(smpl)):
            try:
                float(smpl[i])
                positiveProbability *= calculateGaussDistribution(float(smpl[i]), positive[i]['mean'], positive[i]['stder'])
            except ValueError:
                try:
                    positiveProbability *= positive[i][smpl[i]] / positive[i]['total']
                except KeyError:
                    positiveProbability *= 0               
        for i in range(0, len(smpl)):
            try:
                float(smpl[i])
                negativeProbability *= calculateGaussDistribution(float(smpl[i]), negative[i]['mean'], negative[i]['stder'])
            except ValueError:
                try:
                    negativeProbability *= negative[i][smpl[i]] / negative[i]['total']
                except KeyError:
                    negativeProbability *= 0
        
		# Compare positive and negative labels result.
        if(positiveProbability >= negativeProbability):
            predicted.append('+')
        else:
            predicted.append('-')      
    return predicted

# Predicted and real label compared. Count is increased, if the predicted and real label is equal
def findAccuracy(predicted, testLabel):
    count = 0
    for i in range(0, len(predicted)):
        if predicted[i] == testLabel[i]:
            count +=1
    return count

def trainingAndTesting(samples, labels, k_folk=10):
	# Shows the number of elements in a folk.
    partSize = int(round(len(samples) / k_folk, 0))
    accuracy = 0
    i = 0
    
	# Data divided 10 folks
    while i < k_folk:
        j = 0;
		# Data divided train and test
        """TRAINING DATA"""
        train = []
        trainLabels = []
        while j < len(samples):
            if ((j < partSize * i) or (partSize * (i + 1) - 1) < j):
                train.append(samples[j])
                trainLabels.append(labels[j])
            j += 1
        
		# Required calculation made for Naive Bayes
        positiveClassProbability, negativeClassProbability = findClassProbablity(trainLabels)
        positiveIndex = [x for x in range(0, len(trainLabels)) if trainLabels[x] == '+']
        negativeIndex = [x for x in range(0, len(trainLabels)) if trainLabels[x] == '-']
        positive = prepareAttributeForTesting(positiveIndex, train)
        negative = prepareAttributeForTesting(negativeIndex, train)           
        
        size = partSize
        j = partSize * i
		
        """TEST DATA"""
        test = []
        testLabel = []
        # All folk are not the same size. Therefore, the IndexError was checked.
        while (size != 0):
            try:
                test.append(samples[j])
                testLabel.append(labels[j])
                j += 1
                size = size - 1
            except IndexError:
                size = 0
        
		# Predicted value saved for calculate accuracy
        predicted = findPredictedLabel(positive, positiveClassProbability, negative, negativeClassProbability, test)
        accuracy += findAccuracy(predicted, testLabel)
        i+=1
    return accuracy / len(samples)

if __name__ == "__main__":
    samples, labels = readSamplesAndLabels()
    acc = trainingAndTesting(samples, labels)
    print("Correctly Classified Instances Ratio: " + str(round(acc * 100, 4)) + " % ")
    