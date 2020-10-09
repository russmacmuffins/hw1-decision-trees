import numpy as np

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

    def ID3 (self, features, targets, names, namedict, default):
        num = 0
        for i in targets:
            num += i
        if len(targets) == 0:
            self.value = default
            self.attribute_name = "leaf"
            return self
        elif num == 0:
            self.value = 0
            self.attribute_name = "leaf"
            return self
        elif num == len(targets):
            self.value = 1
            self.attribute_name = "leaf"
            return self
        elif (len(features) == 0) or len(features[0]) == 0:
            self.value = round(num/(len(targets)))
            self.attribute_name = "leaf"
            return self
        if not(len(features) == len(targets)) or not(len(names) == len(features[0])):
            print(len(features[0]) - len(names))
        best = None
        bestVal = 0
        for j in range(len(features[0])):
            newBest = information_gain(features, j, targets)
            if newBest >= bestVal:
                best = j
                bestVal = newBest
        if best is None:
            self.value = default
            self.attribute_name = "leaf"
            return self
        self.attribute_name = names[best]
        self.attribute_index = namedict[self.attribute_name]
        pos_sub, neg_sub, targ_pos, targ_neg, newNames = split(features, targets, names, best)
        #print(pos_sub, neg_sub)
        #print(targ_pos, targ_neg)
        print("new iteration beginning now")
        if not(len(pos_sub) == 0) and not(len(neg_sub) == 0):
            self.branches.append(Node())
            self.branches.append(Node())
            print("pos and neg")
            self.branches[0].ID3(pos_sub, targ_pos, newNames, namedict, 0)
            self.branches[1].ID3(neg_sub, targ_neg, newNames, namedict, 1)
            return self
        elif not(len(pos_sub) == 0):
            self.branches.append(Node())
            self.branches.append(Node(1, "leaf"))
            print("just pos")
            self.branches[0].ID3(pos_sub, targ_pos, newNames, namedict, 0)
            return self
        elif not(len(neg_sub) == 0):
            self.branches.append(Node(0, "leaf"))
            self.branches.append(Node())
            print("just neg")
            self.branches[1].ID3(pos_sub, targ_pos, newNames, namedict, 1)
            return self

    def predict_helper(self, choice):
        #print(self.value)
        if self.attribute_name == "leaf":
            return self.value
        else:
            #print(self.attribute_index)
            if choice[self.attribute_index]:
                self.branches[0].predict_helper(choice)
            else:
                self.branches[1].predict_helper(choice)







class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None



    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)
        namedict = {}
        for i in range(len(self.attribute_names)):
            namedict[self.attribute_names[i]] = i
        self.tree = Node().ID3(features, targets, self.attribute_names, namedict, 1)
        print("\n")
        self.visualize()
        print("\n")


    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons
            for the input data.
        """
        self._check_input(features)
        out = []
        #print("\n")
        #print(features)
        #print("\n")
        for i in features:
            #print(i)
            out.append(self.tree.predict_helper(i))
            #print("\n")
        return np.array(out)



    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = self.tree.value if self.tree.value is not None else -1
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.

    1. num of features with attribute = 1-0
    2. num of test_targets of the feature is represent
    p1 = count of targets that are 1
    p2 count tar 0
    p1_trusplit num of samples classified with a 1 and if that feature is represent
    p1_plitfalse num of samples classified with a 1 and if that feature is not represent
    p2_trusplit num of samples classified with a 0 and if that feature is represent
    p2_plitfalse num of samples classified with a 0 and if that feature is not represent
    """
    p1 = 0
    p2 = 0
    p1_truesplit = 0
    p1_splitfalse = 0
    p2_truesplit = 0
    p2_splitfalse = 0
    for i in range(len(targets)):
        if targets[i]:
            p1 += 1
            #print(features[i])
            if features[i][attribute_index]:
                p1_truesplit += 1
            else:
                p1_splitfalse += 1
        else:
            p2 += 1
            if features[i][attribute_index]:
                p2_truesplit += 1
            else:
                p2_splitfalse += 1
    if ((p1 == 0) and (p2 == 0)) or ((p1_truesplit == 0) and (p2_truesplit == 0)) or ((p1_splitfalse == 0) and (p2_splitfalse == 0)):
        return 0
    else:
        info_gain =  entropy(p1, p2, p1+p2)
        info_gain -= ((p1_truesplit+p2_truesplit)/(p1+p2))*entropy(p1_truesplit, p2_truesplit, p1_truesplit+p2_truesplit)
        info_gain -= ((p1_splitfalse+p2_splitfalse)/(p1+p2))*entropy(p1_splitfalse, p2_splitfalse, p1_splitfalse+p2_splitfalse)
    return info_gain

def entropy(point1, point2, total):
    return -((point1/total)*(np.log2((point1/total))))-((point2/total)*(np.log2((point2/total))))

def split(features, targets, names, index):
    all = [[] for i in range(4)]
    newNames = names.copy()
    newNames.pop(index)
    #print(features)
    for i in range(len(features)):
        if features[i][index] == 1:
            all[0].append(features[i])
            all[1].append(targets[i])
        else:
            all[2].append(features[i])
            all[3].append(targets[i])

    for j in range(len(all[0])):
        #print()
        #print(len(all[0][j]))
        a = np.delete(all[0][j], index)
        all[0][j] = a
        #print(len(all[0][j]))
    for k in range(len(all[2])):
        #print()
        #print(len(all[2][k]))
        b = np.delete(all[2][k], index)
        all[2][k] = b
        #print(len(all[2][k]))

    #print((len(all[0][0])) - len(names))

    return np.array(all[0]), np.array(all[2]), np.array(all[1]),  np.array(all[3]),  newNames




if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
