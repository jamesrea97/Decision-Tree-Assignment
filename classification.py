##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the
# DecisionTreeClassifier
##############################################################################

import numpy as np
import dataReader as dr
import entropy as ent
import matplotlib.pyplot as plt
from scipy import stats
import eval


class LeafNode(object):
    """
    A leafNode

    Attributes
    ----------
    A letter which contains the majority label
    Number of data points used to classify the node
    Entropy of the data points used to classify the node
    """

    def __init__(self, letterList):
        self.letter = stats.mode(letterList)[0][0]
        self.leaf_total = len(letterList)
        self.letterList = letterList
        self.entropy = ent.calcEntropy(letterList)

    def __str__(self):
        """To string method for visual tree representation"""
        return chr(self.letter) + "\n" + "Tot:" + str(self.leaf_total) + "\n" + "S:" + str(round(self.entropy, 2))

    def NodeHeight(self):
        """All leaf nodes have depth 0 by default"""
        return 0


class Node(object):
    """
    A Node

    Attributes
    ----------
    split_col: int
        The column index referencing the attribute on which the data set will be split

    threshold: int
        The threshold value on which the data will be split for the attribute given

    left_node: Node
        The Node to the left, either leads to another Node or a LeafNode(label)

    right_node: Node
        The Node to the right, either leads to another Node or a LeafNode(label)

    letters: Node
        The letters present in the dataset used to create the node

    entropy: Node
        The entropy of the dataset used to create the node

    node_total: Node
        Total number of data points in the data set used to create the node
    """

    def __init__(self, split_col, threshold, leftData, rightData, letters, entropy, node_total):
        self.split_col = split_col
        self.threshold = threshold
        self.left_node = Node.induceDecisionTree(leftData)
        self.right_node = Node.induceDecisionTree(rightData)
        self.letters = letters
        self.entropy = entropy
        self.node_total = node_total

    def __str__(self):
        """To string method for visual representation of node in a tree"""
        return "x" + str(self.split_col) + "<" + str(self.threshold) + "\n" + "Tot:" + str(
            self.node_total) + "\n" + "E:" + str(round(self.entropy, 2)) + "\n" + str(self.letters) + "\n"

    def NodeHeight(self):
        """Gives max distance of node from a leaf node (for the root node= max depth of tree)"""
        return 1 + max(self.left_node.NodeHeight(), self.right_node.NodeHeight())

    @staticmethod
    def induceDecisionTree(dataSet):
        """"Recursive function to create nodes of tree based upon a given data set"""
        # Check that their are unique attributes and classifications
        attributeRepeats = len(np.unique(dataSet[:, :-1], axis=0))
        classificationRepeats = len(np.unique(dataSet[:, -1]))
        node_total = len(dataSet)
        entropy = ent.calcEntropy(dataSet[:, -1])
        # If not the data set cannot be split further so create a leafNode. Also true if length of dataSet is 1
        if (len(dataSet) == 1) or (attributeRepeats == 1) or (classificationRepeats == 1):
            return LeafNode(dataSet[:, -1])

        # Get dataSet info for __str__ method
        node_total = len(dataSet)
        (unique, counts) = np.unique(dataSet[:, -1], return_counts=True)
        characters = list()
        for i in range(len(unique)):
            characters.append(chr(unique[i]))
        np.asarray(characters)
        letters = np.asarray((characters, counts)).T

        # Get optimal splitting point
        split_col, threshold, leftChildData, rightChildData = ent.findBestNode(dataSet)

        # Create new node with children given by the split defined above
        return Node(split_col, threshold, leftChildData, rightChildData, letters, entropy, node_total)

    def reducedErrorPrune(self, decTree, accuracy, validationData):
        """An implementation of Reduced accuracy pruning"""
        leftCompacted = False
        rightCompacted = False
        # If both children are leaf nodes compact them
        if isinstance(self.left_node, LeafNode) and isinstance(self.right_node, LeafNode) and (
                self != decTree.rootNode):
            return self.compact(), accuracy, True
        if isinstance(self.left_node, Node):
            # Save state
            savedNode = self.left_node
            savedAccuracy = accuracy
            # Attempt to compact via a recursive call
            self.left_node, accuracy, leftCompacted = self.left_node.reducedErrorPrune(decTree, accuracy,
                                                                                       validationData)
            if leftCompacted:
                # If accuracy reduced return to saved state
                accuracy = eval.Evaluator.getAccuracyOfDecisionTree(decTree, validationData[0], validationData[1])
                if accuracy < savedAccuracy:
                    self.left_node = savedNode
                    accuracy = savedAccuracy
                    leftCompacted = False
        if isinstance(self.right_node, Node):
            # Save state
            savedNode = self.right_node
            savedAccuracy = accuracy
            # Attempt to compact via a recursive call
            self.right_node, accuracy, rightCompacted = self.right_node.reducedErrorPrune(decTree, accuracy,
                                                                                          validationData)
            if rightCompacted:
                # If accuracy reduced return to saved state
                accuracy = eval.Evaluator.getAccuracyOfDecisionTree(decTree, validationData[0], validationData[1])
                if accuracy < savedAccuracy:
                    self.right_node = savedNode
                    accuracy = savedAccuracy
                    rightCompacted = False
        return self, accuracy, (leftCompacted | rightCompacted)

    def compact(self):
        """Compact 2 child leaf nodes by combining attributes to return a single lead node"""
        letterList = np.append(self.left_node.letterList, self.right_node.letterList)
        return LeafNode(letterList)


class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X

    """

    def __init__(self):
        self.is_trained = False
        self.rootNode = None

    def train(self, x, y):
        """ Constructs a decision tree classifier from data

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################
        self.rootNode = Node.induceDecisionTree(dr.mergeAttributesAndCharacteristics(x, y))

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def predict(self, attributeInstances):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty list (will convert to numpy array)
        predictions = list()

        # predictions = np.zeros((attributeInstances.shape[0],), dtype=np.object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # remember to change this if you rename the variable

        for attributeList in attributeInstances:
            predictions.append((DecisionTreeClassifier.predictInstance(self.rootNode, attributeList)))

        return np.asarray(predictions)

    @staticmethod
    def predictInstance(node, attributeList):
        """Recursive function to return a prediction from the decision tree given a list of attributes"""
        if isinstance(node, LeafNode):
            return chr(node.letter)
        else:
            if int(attributeList[node.split_col]) <= int(node.threshold):
                return DecisionTreeClassifier.predictInstance(node.left_node, attributeList)
            else:
                return DecisionTreeClassifier.predictInstance(node.right_node, attributeList)

    def prune(self, validationData):
        """Continually calls reduced error prunning tree with a given set of validation data
            until no further pruning can occur"""
        pruneOccurred = True
        accuracy = eval.Evaluator.getAccuracyOfDecisionTree(self, validationData[0], validationData[1])
        while pruneOccurred:
            node, accuracy, pruneOccurred = self.rootNode.reducedErrorPrune(self, accuracy, validationData)

    def plot_tree(self):
        """Plots a visual representation of the current tree to a depth of 4 nodes"""
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        # set window size
        x1 = 0
        x2 = 1000
        y = 100
        middleWidth = 0.5*(x1 + x2)
        plt.figure(figsize=(30, 20))
        plt.axis('off')
        # plot root node as a rectangle
        plt.text(middleWidth, y, str(self.rootNode), size=7, color='green',
                 horizontalalignment="center", verticalalignment="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='green'))
        # call helper functions on left and right node to plot the children
        steps = 0
        DecisionTreeClassifier.plot_tree_helper(self.rootNode.left_node, x1, middleWidth, y - 5, steps, middleWidth)
        DecisionTreeClassifier.plot_tree_helper(self.rootNode.right_node, middleWidth, x2, y - 5, steps, middleWidth)
        plt.savefig("tree.png")
        #plt.show(aspect="auto")

    @staticmethod
    def plot_tree_helper(node, x1, x2, y, steps, parent_xval):
        """Helper function to plot a visual rep of the decision tree"""
        # calculate mid point of the sub window
        middleWidth = 0.5*(x1 + x2)
        # if node is a leaf, plot as a filled in box, else plot with a white background
        if isinstance(node, LeafNode):
            plt.text(middleWidth, y, str(node), size=7, color='white', horizontalalignment="center",
                     verticalalignment="center",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='green', edgecolor='white'))
            plt.plot([parent_xval, middleWidth], [y + 5, y], 'brown', linestyle=':', marker='')
            return
        else:
            plt.text(middleWidth, y, str(node), size=7, color='green', horizontalalignment="center",
                     verticalalignment="center",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='green'))
            # Line to parent, adjusting the number '5' with line length required
            plt.plot([parent_xval, middleWidth], [y + 5, y], 'brown', linestyle=':', marker='')
            # change depth of tree shown
            # if steps == 3:
            #  return
            left_height = node.left_node.NodeHeight()
            right_height = node.right_node.NodeHeight()
            # update the weight value to allocate a larger space for child with largest height
            weightedx = x1 + ((left_height + 1) / (left_height + right_height + 2)) * (x2 - x1)
            DecisionTreeClassifier.plot_tree_helper(node.left_node,
                                                    x1, weightedx, y - 5, steps + 1, middleWidth)
            DecisionTreeClassifier.plot_tree_helper(node.right_node,
                                                    weightedx, x2, y - 5, steps + 1, middleWidth)

if __name__ == "__main__":
    """Mock training programme"""
    trainingData = dr.parseFile("data/train_full.txt")
    validationData = dr.parseFile("data/validation.txt")
    testData = dr.parseFile("data/test.txt")

    tree = DecisionTreeClassifier()
    tree.train(trainingData[0], trainingData[1])
    # tree.predict(data)
    tree.plot_tree()
    print(eval.Evaluator.getAccuracyOfDecisionTree(tree, testData[0], testData[1]))

    print("----------------PRUNE------------------------")
    tree.prune(validationData)
    tree.plot_tree()
    print("----------------Test------------------------")
    print(eval.Evaluator.getAccuracyOfDecisionTree(tree, testData[0], testData[1]))
