from sklearn import tree, metrics, datasets
import graphviz

digits = datasets.load_digits()
data = digits.data
i = int(len(data)*0.7)
training_data = data[0:i]
test_data = data[i:len(data)]
target = digits.target
training_labels = target[0:i]
test_labels = target[i:len(target)]


# fit(self, X, y) -> Build a decision tree classifier from the training set (X, y)
# predict(self, X) -> Predict class or regression value for X. For a classification model,
# the predicted class for each sample X is returned.
clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_labels)

dot_data = tree.export_graphviz(clf)
graph = graphviz.Source(dot_data)
graph.render('DigitsTreeSciKitLearn')

predicted_labels = clf.predict(test_data)

print(predicted_labels)

print(metrics.classification_report(test_labels, predicted_labels))
print(metrics.confusion_matrix(test_labels, predicted_labels, labels=digits.target_names))

# ---------------------------------------------------------------------------------------------------------------------

clf2 = tree.DecisionTreeClassifier(min_samples_leaf=3)
clf2.fit(training_data, training_labels)

dot_data = tree.export_graphviz(clf2)
graph = graphviz.Source(dot_data)
graph.render('DigitsTreeSciKitLearn')

predicted_labels = clf.predict(test_data)

print(metrics.classification_report(test_labels, predicted_labels))
print(metrics.confusion_matrix(test_labels, predicted_labels, labels=digits.target_names))
