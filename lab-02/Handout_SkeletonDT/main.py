import ToyData as td
import ID3

import numpy as np
from sklearn import tree, metrics, datasets
import graphviz


def main():

    attributes, classes, data, target, test_data, test_target = td.ToyData().get_data()

    id3 = ID3.ID3DecisionTreeClassifier()

    map_attr = id3.map_attr_index(attributes)

    myTree = id3.fit(data, target, attributes, classes, map_attr, -1)
    print('Root node Toy Data:')
    print(myTree)
    print()
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(test_data, map_attr)
    print('Predicted labels Toy Data:')
    print(predicted)
    print()

    print('Classification Report Toy Data:')
    print(metrics.classification_report(test_target, predicted))
    print('Confusion Matrix Toy Data:')
    print(metrics.confusion_matrix(test_target, predicted, labels=classes))

    # -----------------------------------------------------------------------------------------------------------------
    print()

    digits = datasets.load_digits()
    data_d = digits.data
    i = int(len(data_d) * 0.7)
    training_data = data_d[0:i]
    test_data_d = data_d[i:len(data_d)]
    target_d = digits.target
    training_labels = target_d[0:i]
    test_labels = target_d[i:len(target_d)]
    classes_d = digits.target_names

    attributes_val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    attributes_d = {}
    for j in range(64):
        attributes_d[j] = attributes_val

    id3_d = ID3.ID3DecisionTreeClassifier()
    map_attr_d = id3_d.map_attr_index(attributes_d)

    myTree_d = id3_d.fit(training_data, training_labels, attributes_d, classes_d, map_attr_d, -1)
    print('Root node Digits:')
    print(myTree_d)
    print()
    plot = id3_d.make_dot_data()
    plot.render("testTree_d")
    predicted_d = id3_d.predict(test_data_d, map_attr_d)
    print('Predicted labels Digits:')
    print(predicted_d)
    print()

    print('Classification Report Digits:')
    print(metrics.classification_report(test_labels, predicted_d))
    print('Confusion Matrix Digits:')
    print(metrics.confusion_matrix(test_labels, predicted_d, labels=classes_d))

    # -----------------------------------------------------------------------------------------------------------------
    print()
    attributes_val_2 = ['d', 'g', 'l']
    attributes_d_2 = {}
    for k in range(64):
        attributes_d_2[k] = attributes_val_2

    id3_d_2 = ID3.ID3DecisionTreeClassifier()
    map_attr_d_2 = id3_d_2.map_attr_index(attributes_d_2)

    new_data = []
    for d in data_d:
        row = []
        for l in range(len(d)):
            if d[l] < 5:
                row.append('d')
            elif d[l] > 10:
                row.append('l')
            else:
                row.append('g')
        new_data.append(row)

    training_data_2 = new_data[0:i]
    test_data_d_2 = new_data[i:len(data_d)]

    myTree_d_2 = id3_d_2.fit(training_data_2, training_labels, attributes_d_2, classes_d, map_attr_d_2, -1)
    print('Root node Digits with new attributes:')
    print(myTree_d_2)
    print()
    plot = id3_d_2.make_dot_data()
    plot.render("testTree_d_2")
    predicted_d_2 = id3_d_2.predict(test_data_d_2, map_attr_d_2)
    print('Predicted labels Digits with new attributes:')
    print(predicted_d_2)
    print()

    print('Classification Report Digits with new attributes:')
    print(metrics.classification_report(test_labels, predicted_d_2))
    print('Confusion Matrix Digits with new attributes:')
    print(metrics.confusion_matrix(test_labels, predicted_d_2, labels=classes_d))


if __name__ == "__main__": main()