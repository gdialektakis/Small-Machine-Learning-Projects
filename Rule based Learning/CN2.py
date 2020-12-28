# =============================================================================
# HOMEWORK 3 - RULE-BASED LEARNING
# CN2 ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# For this project, the only thing that we will need to import is the "Orange" library.
# However, before importing it, you must first install the library into Python.
# Read the instructions on how to do that (it might be a bit trickier than usual!)
# =============================================================================


# IMPORT LIBRARY HERE (trivial but necessary...)
import Orange

# =============================================================================



# Load 'wine' dataset
# =============================================================================


# ADD COMMAND TO LOAD TRAIN AND TEST DATA HERE
from Orange import evaluation
from Orange.evaluation import testing
from Orange.classification import rules
wineData = Orange.data.Table("wine")
# =============================================================================




# Define the learner that will be trained with the data.
# Try two different learners: an '(Ordered) Learner' and an 'UnorderedLearner'.
# =============================================================================


# ADD COMMAND TO DEFINE LEARNER HERE
ordered_learner = Orange.classification.CN2Learner()
unordered_learner = Orange.classification.CN2UnorderedLearner()
laplace_learner = Orange.classification.CN2Learner()
# =============================================================================




# At this step we shall configure the parameters of our learner.
# We can set the evaluator/heuristic ('Entropy', 'Laplace' or 'WRAcc'),
# 'beam_width' (in the range of 3-10), 'min_covered_examples' (start from 7-8 and make your way up),
# and 'max_rule_length' (usual values are in the range of 2-5).
# They are located deep inside the 'learner', within the 'rule_finder' class.
# Note: for the evaluator, set it using one of the Evaluator classes in classification.rules
# =============================================================================


# ADD COMMANDS TO CONFIGURE THE LEARNER HERE
ordered_learner.rule_finder.search_algorithm.beam_width = 10
ordered_learner.rule_finder.general_validator.min_covered_examples = 7
ordered_learner.rule_finder.general_validator.max_rule_length = 5
# =============================================================================
unordered_learner.rule_finder.search_algorithm.beam_width = 10
unordered_learner.rule_finder.general_validator.min_covered_examples = 7
unordered_learner.rule_finder.general_validator.max_rule_length = 5
# =============================================================================
laplace_learner.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()
laplace_learner.rule_finder.search_algorithm.beam_width = 9
laplace_learner.rule_finder.general_validator.min_covered_examples = 10
laplace_learner.rule_finder.general_validator.max_rule_length = 5


# We want to test our model now. The CrossValidation() function will do all the
# work in this case, which includes splitting the whole dataset into train and test subsets,
# then train the model, and produce results.
# So, simply initialize the CrossValidation() object from the 'testing' library
# and call it with input arguments 1) the dataset and 2) the learner.
# Note that the 'learner' argument should be in array form, i.e. '[learner]'.

cv = testing.CrossValidation(k=5)
ordered_results = cv(wineData, [ordered_learner])
unordered_results = cv(wineData, [unordered_learner])
laplace_results = cv(wineData, [laplace_learner])

# As for the required metrics, you can get them using the 'evaluation.scoring' library.
# The 'average' parameter of each metric is used while measuring scores to perform
# a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET (AGAIN). USE EITHER
# 'MICRO' OR 'MACRO' (preferably 'macro', at least for final results).
# ============================================================================

# # ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("-------------------------SCORES-----------------------------")
print("-------------------Ordered learner scores-------------------")
print("Precision: %2f" % Orange.evaluation.Precision(results=ordered_results, average="macro"))
print("F1: %2f" % Orange.evaluation.F1(results=ordered_results, average="macro"))
print("Recall: %2f" % Orange.evaluation.Recall(results=ordered_results, average="macro"))
print("Accuracy: %2f" % Orange.evaluation.CA(results=ordered_results))
# =============================================================================
print("\n")
print("-------------------Unordered learner scores-------------------")
print("Precision: %2f" % Orange.evaluation.Precision(results=unordered_results, average="macro"))
print("F1: %2f" % Orange.evaluation.F1(results=unordered_results, average="macro"))
print("Recall: %2f" % Orange.evaluation.Recall(results=unordered_results, average="macro"))
print("Accuracy: %2f" % Orange.evaluation.CA(results=unordered_results))
# =============================================================================
print("\n")
print("-------------------Laplace learner scores-------------------")
print("Precision: %2f" % Orange.evaluation.Precision(results=laplace_results, average="macro"))
print("F1: %2f" % Orange.evaluation.F1(results=laplace_results, average="macro"))
print("Recall: %2f" % Orange.evaluation.Recall(results=laplace_results, average="macro"))
print("Accuracy: %2f" % Orange.evaluation.CA(results=laplace_results))

print("\n")
# Ok, now let's train our learner manually to see how it can classify our data
# using rules.You just want to feed it some data- nothing else.
# =============================================================================


# ADD COMMAND TO TRAIN THE LEARNER HERE
ordered_classifier = ordered_learner(wineData)
unordered_classifier = unordered_learner(wineData)
laplace_classifier = laplace_learner(wineData)
# =============================================================================

print("-------------------------RULES-----------------------------")
print("\n")
# Now we can print the derived rules. To do that, we need to iterate through
# the 'rule_list' of our classifier.
print("------------------Ordered classifier rules------------------")
for rule in ordered_classifier.rule_list:
    print(rule, rule.curr_class_dist.tolist())

print("\n")
print("------------------Unordered classifier rules------------------")
for rule in unordered_classifier.rule_list:
    print(rule, rule.curr_class_dist.tolist())

print("\n")
print("------------------Laplace classifier rules------------------")
for rule in laplace_classifier.rule_list:
    print(rule, rule.curr_class_dist.tolist())