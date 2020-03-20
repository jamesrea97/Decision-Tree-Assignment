import numpy as np
import matplotlib.pyplot as plt

"""

A file to read in given data files into a 2D numpy array of form:
[[Attribute11, Attribute12, ..., Attribute1N, Classification1],
[Attribute21, Attribute22, ..., Attribute2N, Classification2],
.                                           .
.                                           .       
.                                           .
AttributeM1, AttributeM2, ..., AttributeMN, ClassificationM]]

Also includes a number of functions to then analyse this parsed data and present visually using matplotlib

"""


def parseFile(fname):
    """Reads in file line by line and appends to array
        returns a multi-dim array of dimensions numRows x numFeatures+1
        with the final column being the classification """
    attributes = []
    characteristics = []
    with open(fname) as file:
        for line in file:
            # New line character needs to be removed
            line = line.strip()
            attributes.append([int(attribute) for attribute in line[:-2].split(",")])
            characteristics.append(line[-1])
    return np.array(attributes), characteristics


def mergeAttributesAndCharacteristics(attributes, characteristics):
    "Appends array of 1D chars cast to ASCII numbers to the final column of a 2D array of attributes"
    return np.c_[attributes, [ord(i) for i in characteristics]]


def attributesToScatterData(data):
    """Returns the attributes, attribute values and their frequency for all attributes in a given dataset"""
    attributeNumber = 0
    attribute = []
    attributeValue = []
    frequency = []

    # For each column in the data append the frequency value for each attributeValue that is present
    for column in data[:, :-1].transpose():  # Final column is classification so not included
        setAttributeValueFreqFromAttributeColumn(attribute, attributeValue, frequency, column, attributeNumber)
        attributeNumber += 1
    return attribute, attributeValue, frequency


def getClassFreq(dataSet):
    """Get the classification and percentage occurence of that class for a data set"""
    classification = []
    normFrac = []
    # Get characteristic data from final column of the datasets
    setAttributeValueFreqFromAttributeColumn([], classification, normFrac, [chr(i) for i in dataSet[:, -1]], "")
    return classification, np.array([i * 100 for i in normFrac])


def setAttributeValueFreqFromAttributeColumn(attributeList, attributeValueList, frequencyList, dataColumn,
                                             attributeName):
    """Adds attribute names, values and normalised frequency to relevant input lists"""
    normFactor = 1 / len(dataColumn)
    attributeValueFreq = {}

    # Create a dictionary with attribute value as the key and frequency as the value
    for attributeValue in dataColumn:
        if attributeValue in attributeValueFreq:
            attributeValueFreq[attributeValue] += 1
        else:
            attributeValueFreq[attributeValue] = 1

    # Use dictionary to fill attributeList etc....
    for attributeValue, k in sorted(attributeValueFreq.items()):
        attributeList.append(attributeName)
        attributeValueList.append(attributeValue)
        frequencyList.append(attributeValueFreq[attributeValue] * normFactor)


def plotScatterResults(scatterResults, fname):
    """Plots given data as a scatter graph saving the resulting graph to fname"""
    plt.figure(figsize=(6, 4), dpi=140)
    plt.scatter(scatterResults[0], scatterResults[1], s=[i * 1000 for i in scatterResults[2]], c='b', alpha=0.7)

    plt.axis([-0.5, max(fullDataScatter[0]) * 1.1, -0.5, max(fullDataScatter[1]) * 1.1])
    plt.ylabel("Attribute Value")
    plt.xlabel("Attribute Number")
    plt.xticks(range(fullData.shape[1] - 1))
    plt.savefig(fname)
    plt.show()


def plotSideBySideBarChart(xlabels, dataSet1, dataSet1Name, dataSet2, dataSet2Name, fname):
    """Plots dataSet1 and dataSet2 side by side along a common x axis and saves to fname"""
    plt.figure(figsize=(6, 4), dpi=140)
    barWidth = 0.4
    xpos = np.arange(len(xlabels))

    plt.bar(xpos + (barWidth / 2), dataSet1, width=barWidth, label=dataSet1Name)
    plt.bar(xpos - (barWidth / 2), dataSet2, width=barWidth, label=dataSet2Name)

    plt.ylim(0, np.amax([dataSet2, dataSet1]) * 1.1)
    plt.ylabel("Percentage Occurrence")
    plt.xlabel("Character")
    plt.xticks(xpos, xlabels)
    plt.legend()
    plt.savefig(fname)
    plt.show()


def percentageChange(dataSet1, dataSet2):
    """Prints the % of data rows have changed between dataSet1 and dataSet2"""
    numUnchanged = 0
    for data in dataSet2:
        if any((dataSet1[:] == data).all(1)):
            numUnchanged += 1
    print(dataSet1.shape[0] - numUnchanged)
    print("Percentage change: ")
    print((dataSet1.shape[0] - numUnchanged) * 100 / dataSet1.shape[0])


if __name__ == "__main__":
    """Run analysis of given data files"""
    # Read in data
    fullData = parseFile("data/train_full.txt")
    fullData = mergeAttributesAndCharacteristics(fullData[0], fullData[1])
    fullDataScatter = attributesToScatterData(fullData)
    fullDataClassPercent = getClassFreq(fullData)

    subData = parseFile("data/train_sub.txt")
    subData = mergeAttributesAndCharacteristics(subData[0], subData[1])
    subDataClassPercent = getClassFreq(subData)

    noisyData = parseFile("data/train_noisy.txt")
    noisyData = mergeAttributesAndCharacteristics(noisyData[0], noisyData[1])
    noisyDataClassPercent = getClassFreq(noisyData)

    testData = parseFile("data/validation.txt")
    testData = mergeAttributesAndCharacteristics(testData[0], testData[1])
    testDataClassPercent = getClassFreq(testData)

    # Plot relevant graphs
    plotScatterResults(fullDataScatter, "q1_1.png")
    plotSideBySideBarChart(fullDataClassPercent[0], fullDataClassPercent[1], "train_full.txt", subDataClassPercent[1],
                           "train_sub.txt", "q1_2.png")
    plotSideBySideBarChart(fullDataClassPercent[0], fullDataClassPercent[1], "train_full.txt", noisyDataClassPercent[1],
                           "train_sub.txt", "q1_3.png")

    plotSideBySideBarChart(fullDataClassPercent[0], fullDataClassPercent[1], "train_full.txt", testDataClassPercent[1],
                           "test.txt", "q4_1.png")

    # Ouput raw data to terminal
    print("Class percentages")
    print(fullDataClassPercent[0])
    print("Full :")
    print(fullDataClassPercent[1])
    print("Sub :")
    print(subDataClassPercent[1])
    print("Noisy :")
    print(noisyDataClassPercent[1])
    percentageChange(fullData, noisyData)
