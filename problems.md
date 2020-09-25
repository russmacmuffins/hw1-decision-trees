# Coding (6 points)
Your task is to implement two machine learning algorithms:

1. Decision tree (in `code/decision_tree.py`)
2. Prior probability (in `code/prior_probability.py`)

You will also write code that reads in data into numpy arrays and code that manipulates
data for training and testing in `code/data.py`.

You will implement evaluation measures in `code/metrics.py`:

1. Confusion matrix (`code/metrics.py -> confusion_matrix`)
2. Precision and recall (`code/metrics.py -> precision_and_recall`)
3. F1-Measure (`code/metrics.py -> f1_measure`)

The entire workflow will be encapsulated in `code/experiment.py -> run`. The run function 
will allow you to run each approach on different datasets easily. You will have to 
implement this `run` function.

Your goal is to pass the test suite (contained in `tests/`). Once the tests are passed, you 
may move on to the next part - reporting your results.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here. Suggested order for passing test_cases:

1. test_load_data
2. test_train_test_split
3. test_confusion_matrix
4. test_accuracy
5. test_precision_and_recall
6. test_f1_measure
7. test_experiment_run_prior_probability
8. test_experiment_run_decision_tree
9. test_experiment_run_and_compare

# Free-response Questions (4 points)

To answer some of these questions, you will have to write extra code (that is not covered by the test cases). The extra code should import your implementation and run experiments on the various datasets (e.g., choosing `ivy-league.csv` for a dataset and doing `experiment.run` with a 80/20 train/test split, averaged over a few trials). **You do not need to submit this extra code.**

1. Assume you have a deterministic function that takes a fixed, finite number of Boolean inputs and returns a Boolean output. Can a Decision Tree be built to represent any such function? Give a simple proof or explanation for your answer. If you choose to give a proof, don't worry about coming up with a very formal or mathematical proof. It is up to you what you want to present as a proof.

2. Let <img src="/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/>  be a voter in the set of voters <img src="/tex/a9a3a4a202d80326bda413b5562d5cd1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.242037049999992pt height=22.465723500000017pt/>. Let each <img src="/tex/6c4adbc36120d62b98deef2a20d5d303.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55786029999999pt height=14.15524440000002pt/> have a value, assigned to either -1 or 1. Let's define Majority-rule as the sign of the sum all voters. If the sum is negative, it returns -, else it returns +. Assume an odd number of voters (so, there are no ties). Can the Majority-rule algorithm be represented by a Decision Tree that considers a single voter at each decision node? Why or why not? Note that the question is not asking if a Decision Tree can learn Majority-rule in general. It is asking if this particular algorithm, as described, can be learned.

3. In the coding section of this assignment, you trained a Decision Tree with the ID3 algorithm on several datasets (`candy-data.csv`, `majority-rule.csv`, `ivy-league.csv`, and `xor.csv`). For each dataset, report the accuracy on the test data, the number of nodes in the tree, and the maximum depth (number of levels) of the tree. 

4. What is the Inductive Bias of the ID3 algorithm?

5. Explain what overfitting is, and describe how one can tell it has occurred.

6. Explain how pre- and post-pruning are done and the reason for doing them.

7. One can modify the simple ID3 algorithm to handle attributes that are real-valued (e.g., height, weight, age). To do this, one must pick a split point for each attribute (e.g., height > 3) and then determine Information Gain given the split point. How would you pick a split point automatically? Why would you do it that way?  

8. Ensemble methods are learning methods that combine the output of multiple learners. The hope is that an ensemble will do better than any one learner, because it reduces the overfitting that a single learner may be susceptible to. One of the most popular ensemble methods is the Random Forest. The high level idea is to build multiple Decision Trees from the training data. One way to build different Decision Trees from the same data is to train several Decision Trees on different random subsets of the features. If, for example, there are 20 measurable features, you randomly pick 10 of the features and build a tree on those 10, then you randomly pick another 10 features and build a tree using those 10 features. If you were building an ensemble of <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> trees this way, how would you combine their decisions in the end? Explain why you would choose this method. Feel free to provide a citation for your choice (if you cite something, please ALSO provide a hyperlink), but also explain the reason this is your choice.
