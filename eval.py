##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks:
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np
import dataReader as dr
import classification as cp


class Evaluator(object):
    """ Class to perform evaluation
    """

    @staticmethod
    def getAccuracyOfDecisionTree(decisionTree, attributes, groundTruths):
        predictions = decisionTree.predict(attributes)
        confusionMatrix = Evaluator.confusion_matrix(predictions, groundTruths)
        return Evaluator.accuracy(confusionMatrix)

    @staticmethod
    def confusion_matrix(prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(annotation)

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################
        # iterate through rows and columns of the confusion matrix
        row = 0
        col = 0
        # storing count of when true letter is equal to some predicted letter
        for trueLetter in class_labels:
            for predictedLetter in class_labels:
                counter = 0
                for index in range(np.size(prediction)):
                    if trueLetter == annotation[index] and predictedLetter == prediction[index]:
                        counter += 1
                    confusion[row][col] = counter
                col += 1
                col %= len(class_labels)
            row += 1
            row %= len(class_labels)

        return confusion

    @staticmethod
    def accuracy(confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################

        # accuracy is given by instanceWhen(TRUTH == PREDICTED) / ALL EVENTS
        truePostive = np.trace(confusion)
        allEvents = np.sum(confusion)
        # divide by 0 check
        if truePostive == 0 or allEvents == 0:
            return 0
        else:
            return truePostive / allEvents

    @staticmethod
    def precision(confusion):
        """ Computes the precision score per class given a confusion matrix.

        Also returns the macro-averaged precision across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.
        """

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################
        # precision (per characteristic) == TRUTH / TOTAL PREDICTION THAT LETTER
        index = 0
        for letterIndex in range(np.size(confusion[:, -1])):
            if np.sum(confusion[:, letterIndex]) == 0:
                p[index] = 0
            else:
                p[index] = confusion[letterIndex][letterIndex] / np.sum(confusion[:, letterIndex])
            index += 1
        # finding average of the precision score for global
        macro_p = np.average(p)

        return p, macro_p

    @staticmethod
    def recall(confusion):
        """ Computes the recall score per class given a confusion matrix.

        Also returns the macro-averaged recall across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################
        # recall (per characteristic) == TRUTH / TOTAL TIMES THAT WAS THE TRUE LETTER
        index = 0
        for letterIndex in range(np.size(confusion[:, -1])):
            if np.sum(confusion[letterIndex]) == 0:
                r[index] = 0
            else:
                r[index] = confusion[letterIndex][letterIndex] / np.sum(confusion[letterIndex])
            index += 1

        # finding average of the recall score for global
        macro_r = np.average(r)

        return r, macro_r

    @staticmethod
    def f1_score(confusion):
        """ Computes the f1 score per class given a confusion matrix.

        Also returns the macro-averaged f1-score across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion),))

        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################
        precision, macro_p = Evaluator.precision(confusion)
        recall, macro_r = Evaluator.recall(confusion)

        index = 0
        for letterIndex in range(np.size(confusion[:, -1])):
            f[index] = 2 * (precision[index] * recall[index]) / (recall[index] + precision[index])
            index += 1

        # finding average of the f1 for global
        macro_f = np.average(f)

        return f, macro_f

    @staticmethod
    def print_eval(train_data, test_data):
        data = dr.parseFile(train_data)
        x = data[0]
        y = data[1]

        tree = cp.DecisionTreeClassifier()
        tree.train(x, y)
        test = dr.parseFile(test_data)
        xtruth = test[0]
        ytruth = test[1]
        test = dr.mergeAttributesAndCharacteristics(xtruth, ytruth)
        predictions = tree.predict(test)
        e = Evaluator()
        a = e.confusion_matrix(ytruth, predictions)
        print("Confusion" + "\n" + str(a))
        print("Accuracy: " + str(e.accuracy(a)))
        print("Recall: " + str(e.recall(a)))
        print("Precision: " + str(e.precision(a)))
        print("F1score: " + str(e.f1_score(a)))


if __name__ == "__main__":
    print("RESULTS FOR TRAIN_FULL.TXT:")
    Evaluator.print_eval("data/train_full.txt", "data/test.txt")
    print("RESULTS FOR TRAIN_NOISY.TXT:")
    Evaluator.print_eval("data/train_noisy.txt", "data/test.txt")
    print("RESULTS FOR TRAIN_SUB.TXT:")
    Evaluator.print_eval("data/train_sub.txt", "data/test.txt")

