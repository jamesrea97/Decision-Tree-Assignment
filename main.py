import numpy as np

from classification import DecisionTreeClassifier
from eval import Evaluator
import dataReader


def printMetric(confusion):
    print("Confusion matrix:")
    print(confusion)

    accuracy = evaluator.accuracy(confusion)
    print()
    print("Accuracy: {}".format(accuracy))

    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)

    print()

    print()
    print("Macro-averaged Precision: {:.2f}".format(macro_p))
    print("Macro-averaged Recall: {:.2f}".format(macro_r))
    print("Macro-averaged F1: {:.2f}".format(macro_f))


if __name__ == "__main__":
    print("Loading the datasets...");
    trainingData = dataReader.parseFile("data/train_full.txt")
    validationData = dataReader.parseFile("data/validation.txt")
    testData = dataReader.parseFile("data/test.txt")

    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(trainingData[0], trainingData[1])

    predictions = classifier.predict(testData[0])
    print("Pre prunning predictions: {}".format(predictions))

    print("Evaluating test predictions...")
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, testData[1])
    printMetric(confusion)

    print("Pruning the decision tree...")
    classifier.prune(validationData)

    predictions = classifier.predict(testData[0])
    print("Post prunning predictions: {}".format(predictions))

    print("Evaluating test predictions...")
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, testData[1])
    printMetric(confusion)
    classifier.plot_tree()
