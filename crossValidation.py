import numpy as np
import dataReader as dr
import classification as cls
import eval as ev
from scipy import stats


def k_fold_cross_validation(data_set, k, pruning=False):
    accuracy = np.zeros(k)
    tree = cls.DecisionTreeClassifier()
    best_tree = cls.DecisionTreeClassifier()
    max_accuracy = 0
    trees = []
    prePruneConfMatrix = []
    postPruneConfMatrix = []

    for i in range(1, k + 1):
        # Split Data into training and testing data
        split = split_set(data_set, k, i, pruning)
        testing = split[0]
        training = split[1]
        training_x = training[:, :-1]
        training_y = [chr(i) for i in training.T[-1]]

        # Train tree
        testing_y = [chr(i) for i in testing.T[-1]]

        trees.append(cls.DecisionTreeClassifier())
        trees[i-1].train(training_x, training_y)
        tree = trees[i-1]

        if pruning:
            predictions = tree.predict(testing)
            confusion = ev.Evaluator.confusion_matrix(predictions, testing_y)
            prePruneConfMatrix.append(confusion)
            validation = split[2]
            tree.prune((validation[:, :-1], [chr(i) for i in validation[:, -1]]))

        predictions = tree.predict(testing)

        # Evaluation metrics
        eval = ev.Evaluator()
        testing_y = [chr(i) for i in testing.T[-1]]
        confusion = eval.confusion_matrix(predictions, testing_y)
        accuracy[i - 1] = eval.accuracy(confusion)

        # Save tree with best accuracy
        confusion = ev.Evaluator.confusion_matrix(predictions, testing_y)
        postPruneConfMatrix.append(confusion)
        accuracy[i - 1] = ev.Evaluator.accuracy(confusion)

        if accuracy[i - 1] > max_accuracy:
            best_tree = trees[i-1]
            max_accuracy = accuracy[i-1]

    if pruning:
        print("Pre pruning metrics")
        analyseListOfConfMatrix(prePruneConfMatrix)
        print("Post pruning results")
        analyseListOfConfMatrix(postPruneConfMatrix)

    return accuracy, best_tree, trees


def analyseListOfConfMatrix(confMatrixList):
    metrics = []
    for confMatrix in confMatrixList:
        foldMetrics = [ev.Evaluator.accuracy(confMatrix), ev.Evaluator.precision(confMatrix)[1],
                       ev.Evaluator.recall(confMatrix)[1], ev.Evaluator.f1_score(confMatrix)[1]]
        metrics.append(foldMetrics)

    metrics = np.array(metrics)
    print("Accuracy: " + str(np.mean(metrics[:, 0])) + " " + str(np.std(metrics[:, 0])))
    print("Precision: " + str(np.mean(metrics[:, 1])) + " " + str(np.std(metrics[:, 1])))
    print("Recall: " + str(np.mean(metrics[:, 2])) + " " + str(np.std(metrics[:, 2])))
    print("F1: " + str(np.mean(metrics[:, 3])) + " " + str(np.std(metrics[:, 3])))


def k_decision_trees(testing, k, k_trees):

    predictions = list()

    # Get predictions for each tree
    for i in range(1, k + 1):
        predictions.append(k_trees[i - 1].predict(testing))

    # Set up arrays to return
    prediction = np.array(predictions)
    prediction.astype(str)
    best_predictions = np.zeros(len(testing))
    best_predictions.astype(str)

    # Calculate mode for each label
    best_predictions = stats.mode(prediction, axis=0)[0]

    return np.array(best_predictions[0, :])


def split_set(data_set, k, fold, createValidationSet=False):
    if fold > k or fold < 1:
        print("Incorrect usage: fold value greater than k")
        return
    elif k > len(data_set) or k < 2:  # Check for error in k input and return error if so
        print("Incorrect usage: Split value, k greater than sample size or less than 2")
        return
    else:
        width = len(data_set[0])
        data_splits = np.split(data_set, k)
        training_set = np.empty(shape=[0, width], dtype=int)
        validation_set = []

        for i in range(len(data_splits)):
            if i == fold % k:
                testing_set = np.array(data_splits[i])
            elif createValidationSet and i == (fold + 1) % k:
                validation_set = np.array(data_splits[i])
            else:
                training_set = np.concatenate((training_set, data_splits[i]), axis=0)

    training_set = np.asarray(training_set)

    return testing_set, training_set, validation_set


def print_results(predictions, labels, name):
    eval = ev.Evaluator()

    confusion = eval.confusion_matrix(predictions, labels)
    accuracy = eval.accuracy(confusion)
    precision = eval.precision(confusion)
    recall = eval.recall(confusion)
    f1_score = eval.f1_score(confusion)

    print(" ")
    print(" ")
    print("Summary evaluation for " + str(name))
    print("____________________________________")
    print("Confusion Matrix: ")
    print(str(confusion))
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(f1_score))
    print("____________________________________")


if __name__ == "__main__":
    # Data Imports
    full_data = dr.parseFile("data/train_full.txt")
    test_data = dr.parseFile("data/test.txt")
    full_data = dr.mergeAttributesAndCharacteristics(full_data[0], full_data[1])
    test_data = dr.mergeAttributesAndCharacteristics(test_data[0], test_data[1])

    # Random shuffle data once
    np.random.shuffle(full_data)

    #                  Question 3.3
    k = 10
    accuracy, best_tree, k_trees = k_fold_cross_validation(full_data, k)

    # Print Accuracies and Standard Deviations for Question 3.3
    print("Accuracy: " + str(round(accuracy.mean(), 4)))
    print("Standard Deviation: " + str(round(accuracy.std(), 4)))

    # Question 3.4
    x = full_data[:, :-1]
    y = [chr(i) for i in full_data.T[-1]]
    testing_y = [chr(i) for i in test_data.T[-1]]

    # Train tree on train_full.txt
    full_trained = cls.DecisionTreeClassifier()
    full_trained.train(x, y)

    # Generate predictions
    full_predict = full_trained.predict(test_data)
    cross_predict = best_tree.predict(test_data)

    # Print results
    print_results(full_predict, testing_y, "Fully Trained")
    print_results(cross_predict, testing_y, "K-Fold Trained")

    #                    Question 3.5

    # Get predictions for each tree trained in 3.3, k_trees
    k_predict = k_decision_trees(test_data, k, k_trees)
    print_results(k_predict, testing_y, "K-Fold Mode Predict")
