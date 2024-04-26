import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ID1: 205734049
# ID2: 208522094

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # gini = 1 - sum(probabilties^2)
    # more information on slide 19, recitation 2

    # get the last column of the data: slice[all rows, last column]
    labels = data[:, -1]
    # get the unique values and their counts
    unique, counts = np.unique(labels, return_counts=True)
    # calculate the probabilities
    probs = counts / len(labels)
    # calculate the gini impurity
    gini = 1 - np.sum(probs ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # entropy = -sum[(probabilities)*log(probabilities)]
    # more information on slide 19, recitation 2

    # get the labels from the last column slice[all rows, last column]
    labels = data[:, -1]
    # get the unique values and their counts
    unique, counts = np.unique(labels, return_counts=True)
    # calculate the probabilities
    probs = counts / len(labels)
    # calculate entropy
    entropy = -np.sum(probs * np.log2(probs))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []  # holds the value of the feature associated with the children (list).
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # reminder: the prediction of a node is the most common label in the node data
        # (if you would have to predict the label of a group member, you would guess the most common one)

        # get the labels from the last column: slice[all rows, last column]
        labels = self.data[:, -1]
        # get the unique values and their counts
        unique, counts = np.unique(labels, return_counts=True)
        # get the most common label:
        # use np.argmax(counts) to find the index of the most common label,
        # Then use it to find the label itself from unique[]
        pred = unique[np.argmax(counts)]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # add the child node to the children list
        self.children.append(node)
        # add the value of the child node to the children_values list
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # FI = prob(node) * (node impurity) - sumAllChildren:[prob(child) * (child impurity)]

        # calculate the probability of the node
        prob_node = len(self.data) / n_total_sample
        # calculate the impurity of the node
        impurity_node = self.impurity_func(self.data)
        # calculate the sum of the impurities of the children
        # can we do it without iterating all children?
        sum__weighted_children_impurities = 0
        for child in self.children:
            sum__weighted_children_impurities += len(child.data) / n_total_sample * child.impurity_func(child.data)

        # calculate the feature importance
        self.feature_importance = prob_node * impurity_node - sum__weighted_children_impurities
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Split the data according to the feature
        feature_values = np.unique(self.data[:, feature])
        for value in feature_values:
            groups[value] = self.data[self.data[:, feature] == value]

        # Forcing the impurity_func to be entropy if gain_ratio == True
        current_impurity_func = self.impurity_func if not self.gain_ratio else calc_entropy

        # Calculate the Info_gain or Gini_gain appropriately
        weighted_impurity_children = 0
        for sub_data in groups.values():
            weighted_impurity_children += len(sub_data) / len(self.data) * current_impurity_func(sub_data)
        # Calculate the goodness of split
        info_or_gini_gain = current_impurity_func(self.data) - weighted_impurity_children

        # If gain_ratio == False or goodness == 0 return only the goodness of split
        # (The Goodness == 0 condition is so the split info will not be 0, what will cause division by 0)
        if not self.gain_ratio or info_or_gini_gain == 0:
            goodness = info_or_gini_gain

        # If gain_ratio == true, calculate the gain ratio with split info
        else:
            split_info = 0
            for sub_data in groups.values():
                sub_data_weight = len(sub_data) / len(self.data)
                split_info -= sub_data_weight * np.log2(sub_data_weight)
            # Calculate the gain ratio
            goodness = info_or_gini_gain / split_info
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        # if the node is a leaf, return
        if self.terminal:
            return

        # if the node is perfectly classified, return
        if len(np.unique(self.data[:, -1])) == 1:
            self.terminal = True
            return

        # if the node is at the maximum depth, return
        if self.depth == self.max_depth:
            self.terminal = True
            return

        # find the best feature to split according to goodness of split
        best_feature = -1
        best_goodness = 0
        best_groups = None
        for feature in range(self.data.shape[1] - 1):
            goodness, groups = self.goodness_of_split(feature)
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature
                best_groups = groups

        # Assigns the best feature for the split to the node's feature
        self.feature = best_feature

        # Handles the case where goodness = 0 for all features in the current data
        if best_groups is None:
            self.terminal = True
            return

        # Case we are chi pruning.
        # Calculates the chi square, compare it to the values in the table and decides if to split the data or not.
        if self.chi != 1:
            # Calculating the Chi square value with a helper function
            chi_square_val = self.chi_test(best_feature)

            # If chi_square <= the value in the chi table, do not split the data and return
            num_feature_values = len(np.unique(self.data[:, best_feature]))
            num_classes = len(np.unique(self.data[:, -1]))
            # Adjust degrees of freedom for chi-square test
            df = (num_feature_values - 1) * (num_classes - 1)
            if chi_square_val <= chi_table[df][self.chi]:
                self.terminal = True
                return

        # Creates the children nodes for the split
        for value, sub_data in best_groups.items():
            child = DecisionNode(sub_data, self.impurity_func, depth=self.depth + 1,
                                 chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(child, value)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def chi_test(self, feature):
        """
        Performs the chi square test and calculate the chi square value
        according to the lecture's formula.

        :param feature: The feature for the chi square test
        :return: The chi square value
        """
        chi_result = 0
        dataset = pd.DataFrame(self.data)
        feature_values = np.unique(dataset[feature])
        feature_classes = np.unique(dataset.iloc[:, -1])
        for feature_val in feature_values:
            # Number of instances that the value in the current feature column equals feature_val
            Df = len(dataset[dataset[feature] == feature_val])
            for class_val in feature_classes:
                # Number of instances that the value in the current feature column equals feature_val
                # and their label equals to class_val
                pf = len(dataset[(dataset[feature] == feature_val) & (dataset.iloc[:, -1] == class_val)])
                # Calculating E.
                E_class_val = Df * len(dataset[dataset.iloc[:, -1] == class_val]) / dataset.shape[0]
                # Adding the calculation to chi_result
                chi_result += (pf - E_class_val) ** 2 / E_class_val

        return chi_result


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio  #
        self.root = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ##########################################################################
        # Initialize the queue of the the nodes to create the tree from.
        nodes_queue = []

        # Creating the first root node and add it to the queue
        self.root = DecisionNode(self.data, self.impurity_func, depth=0, chi=self.chi,
                                 max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        nodes_queue.append(self.root)

        while nodes_queue:
            current_node = nodes_queue.pop()
            # if current node is perfectly classified, continue to the next node
            if len(np.unique(current_node.data[:, -1])) == 1:
                current_node.terminal = True
                continue
            # if current node is a leaf, continue to the next node
            if current_node.depth == self.max_depth:
                current_node.terminal = True
                continue

            # Splitting the node according to the best feature (Details inside the split function)
            current_node.split()

            # Adds the new children to the queue
            for child in current_node.children:
                nodes_queue.append(child)

        # Calculate the feature importance recursively as a post process
        self.calculate_feature_importance(self.root, self.root.data.shape[0])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calculate_feature_importance(self, node, n_total_sample):
        """
        Calculate recursively the feature importance of every node in the tree.

        :param node: The node to calculate the FI
        :param n_total_sample: The total instances in the tree

        This function has no return value, it only assigns the appropriate FI to each node in the tree.
        """
        # Calculates the current node's FI.
        node.calc_feature_importance(n_total_sample)

        # Recursively iterating over all the children of the current node.
        for child in node.children:
            self.calculate_feature_importance(child, n_total_sample)

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # start from the root
        node = self.root

        # while the node is not a leaf (terminal)
        while not node.terminal:
            # get the feature value of the instance to decide which child to go to
            feature_value = instance[node.feature]
            # check if the feature value is in the training data (if not, return the prediction of the current node)
            child_found = False
            # find the child with the corresponding feature value
            for i, child in enumerate(node.children):
                if node.children_values[i] == feature_value:
                    node = child
                    child_found = True
                    break
            if not child_found:  # the feature value is not in the training data
                return node.pred
        # get the prediction of the leaf node
        pred = node.pred
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # iterate over the dataset and predict each instance
        correct = 0
        for instance in dataset:
            if self.predict(instance) == instance[-1]:
                correct += 1
        # calculate the accuracy
        accuracy = correct / len(dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy

    def depth(self):
        return self.root.depth()


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # note that we are already in the loop of the max_depth values

        # create the tree: entropy with gain_ratio bring the best results (from the previous part)
        tree = DecisionTree(X_train, calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        # calculate the accuracy
        training_accuracy, validation_accuracy = tree.calc_accuracy(X_train), tree.calc_accuracy(X_validation)
        # append the accuracies to the lists
        training.append(training_accuracy)
        validation.append(validation_accuracy)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    chi_testing_acc = []

    # iterate over the chi values
    for p_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        # create the tree: entropy + gain_ratio (bring the best results, from the previous part)
        tree = DecisionTree(X_train, calc_entropy, gain_ratio=True, chi=p_val)
        # build the tree
        tree.build_tree()
        # calculate the accuracy
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))
        # get the depth of the tree
        depth.append(tree.depth)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_nodes = 0

    if not node:
        return 0

    if len(node.children) == 0:
        return 1

    for child in node.children:
        n_nodes += count_nodes(child)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes
