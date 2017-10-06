#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *
from sklearn.grid_search import GridSearchCV
sys.path.append("../tools/")

### Custom made modules
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, load_classifier_and_data, \
    test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

# Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

# Print key properties of the dataset 
poi_count = 0

print "Number of persons: ", len(data_dict.keys())
for person in data_dict.keys():
    if data_dict[person]["poi"]:
        poi_count += 1
print "Number of POI: ", poi_count
print "Number of features per person: ", len(data_dict[data_dict.keys()[0]])
print ""

# Print out number of missing values for each feature
print "FEATURE", " "*(25-len("FEATURE")), "# MISSING VALUES" 
print "-------------------------------------------"
for feature in data_dict[data_dict.keys()[0]]:
    count = 0
    for person in data_dict.iteritems():
        if person[1][feature] == "NaN":
            count += 1
    print feature, " "*(32-len(feature)), count
print ""

### Task 2: Remove outliers

# Plot bonus vs salary
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus, color = "b" )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

data_dict_salary = []
for person in data_dict:
    value = data_dict[person]['salary']
    if value != 'NaN':
        data_dict_salary.append((person, int(value)))

# Print top 5 outliers based on salary
outliers = (sorted(data_dict_salary,key=lambda x:x[1],reverse=True)[:5])
print "NAME", " "*(25-len("NAME")), "SALARY"
print "------------------------------------"
for name, salary in outliers:
    print name, " "*(25-len(name)), salary
print ""

# Remove outliers and re-plot bonus Vs salary 
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

features = ["salary", "bonus", "poi"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    # marking POI as red and non-POI as blue
    if point[2] ==  1:
        poi = plt.scatter( salary, bonus, color = "r" )
    else:
        non_poi = plt.scatter( salary, bonus, color = "b" )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.legend((poi, non_poi),
           ('POI', 'Non-POI'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)
plt.show()

### Task 3: Create new feature(s)
# Create ratio of emails sent to and received from POIs per individual 
for person in data_dict:
    data_dict[person]["from_this_person_to_poi_ratio"]=0
    data_dict[person]["from_poi_to_this_person_ratio"]=0
    
    to_messages = float(data_dict[person]["to_messages"])
    from_messages = float(data_dict[person]["from_messages"])
    from_poi_to_this_person = float(data_dict[person]["from_poi_to_this_person"])
    from_this_person_to_poi = float(data_dict[person]["from_this_person_to_poi"])
    
    if from_messages > 0 and from_this_person_to_poi != "NaN":
        data_dict[person]["from_this_person_to_poi_ratio"] = \
            from_this_person_to_poi/from_messages
    if to_messages > 0 and from_poi_to_this_person != "NaN":
        data_dict[person]["from_poi_to_this_person_ratio"] = \
            from_poi_to_this_person/to_messages

features = ["from_this_person_to_poi_ratio", 
            "from_poi_to_this_person_ratio", "poi"]
data = featureFormat(data_dict, features)

# Plot new feature
for point in data:
    from_this_person_to_poi_ratio = point[0]
    from_poi_to_this_person_ratio = point[1]
    # Marking POI as red and non-POI as blue
    if point[2] ==  1:
        poi = plt.scatter( from_poi_to_this_person_ratio, 
                           from_this_person_to_poi_ratio, color = "r")
    else:
        non_poi = plt.scatter( from_poi_to_this_person_ratio, 
                               from_this_person_to_poi_ratio, color = "b")

plt.xlabel("Ratio of emails received from POI")
plt.ylabel("Ratio of emails sent to POI")
plt.legend((poi, non_poi),
           ('POI', 'Non-POI'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)
plt.show()

# Select and rank features based on SelectKBest
features_list = ['poi', 'salary', 'deferral_payments',
                 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'restricted_stock',
                 'director_fees', 'shared_receipt_with_poi',
                 'from_this_person_to_poi_ratio',
                 'from_poi_to_this_person_ratio']

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scale features via min-max
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
# Feature selection
selection = SelectKBest()
selection.fit(features, labels)

results = zip(features_list[1:], selection.scores_)
results = sorted(results, key=lambda x: x[1], reverse=True)
print "FEATURE", " "*(29-len("FEATURE")), "KBEST SCORE"
print "----------------------------------------------"
for feature, score in results:
    print feature, " "*(32-len(feature)), round(score,2) 
print ""

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Evaluate what is the optimum number of features for each algorithm

######## UNCOMMENT CODE IF WANT TO PLOT SCORES VS # OF FEATURES #########
features_list = ['poi', 'exercised_stock_options', 'total_stock_value',
                 'bonus', 'salary', 'from_this_person_to_poi_ratio',
                 'deferred_income',  'long_term_incentive',
                 'restricted_stock', 'total_payments',
                 'shared_receipt_with_poi', 'loan_advances', 'expenses',
                 'from_poi_to_this_person_ratio', 'director_fees', 
                 'deferral_payments', 'restricted_stock_deferred']
    
#algorithms = ["Gaussian", "RandomForest", "AdaBoost", "KNeighbor"]
#accuracy_values = []
#precision_values = []
#recall_values = []
#f1_values = []

#count = 0
#for algorithm in algorithms:
#    accuracy_values.append([])
#    precision_values.append([])
#    recall_values.append([])
#    f1_values.append([])
#    feature_no = []

#    i = 1
#   while i < len(features_list):
#        features_sublist = features_list[:i+1]

        #Define classifiers
#        if algorithm == "Gaussian":
#            clf = GaussianNB()
#        elif algorithm == "RandomForest":
#            clf = RandomForestClassifier()
#        elif algorithm == "AdaBoost":
#            clf = AdaBoostClassifier()
#        elif algorithm == "KNeighbor":
#            clf = KNeighborsClassifier()
        
#        dump_classifier_and_data(clf, data_dict, features_sublist)
#        clf_test, dataset_test, feature_list_test = load_classifier_and_data()
        
#        accuracy, precision, recall, f1 = test_classifier(clf_test,
#                                                          dataset_test,
#                                                          feature_list_test,
#                                                          False)
#        accuracy_values[count].append(accuracy)
#        precision_values[count].append(precision)
#        recall_values[count].append(recall)
#        f1_values[count].append(f1)
        
#        feature_no.append(i)
#        i += 1
    
#    accuracy_plot = plt.plot( feature_no, accuracy_values[count],
#                              color = "r", label="accuracy")
#    precision_plot = plt.plot( feature_no, precision_values[count],
#                               color = "g", label="precision")
#    recall_plot = plt.plot( feature_no, recall_values[count],
#                            color = "b", label="recall")
#    f1_plot = plt.plot( feature_no, f1_values[count], color = "black",
#                        label="F1")
#    plt.xlabel("# of Features")
#    plt.ylabel("Scores")
#    plt.title(algorithm)
#    plt.legend(bbox_to_anchor=(0., -0.25, 1., .102), loc=3, ncol=4,
#               mode="expand", borderaxespad=0.)
#    plt.show()
    
#    count += 1
######## UNCOMMENT CODE IF WANT TO PLOT SCORES VS # OF FEATURES #########
    
# Evaluate which is the best performing algorithm based on optimum number
# of features
optimum_feature_no = [6, 3, 7, 5]

algorithms = ["Gaussian", "RandomForest", "AdaBoost", "KNeighbor"]
accuracy_optimum_values = []
precision_optimum_values = []
recall_optimum_values = []
f1_optimum_values = []

for i, algorithm in enumerate(algorithms):
    feature_no = []

    features_sublist = features_list[:optimum_feature_no[i]+1]

    #Define classifiers
    if algorithm == "Gaussian":
        clf = GaussianNB()
    elif algorithm == "RandomForest":
        clf = RandomForestClassifier()
    elif algorithm == "AdaBoost":
        clf = AdaBoostClassifier()
    elif algorithm == "KNeighbor":
        clf = KNeighborsClassifier()
        
    dump_classifier_and_data(clf, data_dict, features_sublist)
    clf_test, dataset_test, feature_list_test = load_classifier_and_data()
        
    accuracy, precision, recall, f1 = test_classifier(clf_test, dataset_test,
                                                      feature_list_test, False)
    accuracy_optimum_values.append(accuracy)
    precision_optimum_values.append(precision)
    recall_optimum_values.append(recall)
    f1_optimum_values.append(f1)

print "ALGORITHM", " "*(15-len("ALGORITHM")), "# OF FEATURES", " "* \
    (18-len("# OF FEATURES")), "F1 SCORE"
print "-------------------------------------------------"
for i, algorithm in enumerate(algorithms):
    features = optimum_feature_no[i]
    print algorithm, " "*(20-len(algorithm)), features, " "*13, \
        round(f1_optimum_values[i],4)
    
print ""
print "Gaussian Naive Bayes"
index = 0
print "Number of features: ", optimum_feature_no[index]
print "Accuracy:  ", round(accuracy_optimum_values[index],4)
print "Precision: ", round(precision_optimum_values[index],4)
print "Recall:    ", round(recall_optimum_values[index],4)
print ""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print "Before Fine Tuning Parameters:"
print "KNeighbor"
index = 3
print "Number of features: ", optimum_feature_no[index]
print "Accuracy:  ", round(accuracy_optimum_values[index],4)
print "Precision: ", round(precision_optimum_values[index],4)
print "Recall:    ", round(recall_optimum_values[index],4)
print "F1 score:  ", round(f1_optimum_values[index],4)
print ""

features_sublist = features_list[:optimum_feature_no[index]+1]

parameters = {"n_neighbors":[4, 5, 6], "leaf_size":[20, 30, 40],
              "algorithm":["auto", "ball_tree", "brute"]}
clf_kn = KNeighborsClassifier()
clf = GridSearchCV(clf_kn, parameters)

dump_classifier_and_data(clf, data_dict, features_sublist)
clf_test, dataset_test, feature_list_test = load_classifier_and_data()

accuracy, precision, recall, f1 = test_classifier(clf_test, dataset_test,
                                                  feature_list_test, False)
print "After Fine Tuning Parameters:"
print "KNeighbor"
index = 3
print "Number of features: ", optimum_feature_no[index]
print "Accuracy:  ", round(accuracy,4)
print "Precision: ", round(precision,4)
print "Recall:    ", round(recall,4)
print "F1 score:  ", round(f1,4)
print ""

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = GaussianNB()
features_list = ['poi', 'exercised_stock_options', 'total_stock_value',
                 'bonus', 'salary', 'from_this_person_to_poi_ratio',
                 'deferred_income']

dump_classifier_and_data(clf, data_dict, features_list)
