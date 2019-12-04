from collections import Counter
from graphviz import Digraph
import math


class ID3DecisionTreeClassifier:

    def __init__(self, minSamplesLeaf=1, minSamplesSplit=2):

        self.__nodeCounter = 0

        self.nodes = dict()

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        self.nodes[node['id']] = node
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))



    # make the visualisation available
    def make_dot_data(self):
        return self.__dot

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes, classes, entropy, map_attr):
        # entropy = self.entropy(target, classes)
        info_gain = {'attribute': '', 'info_gain': 0}
        # Change this to make some more sense
        for a in attributes:
            index = map_attr[a]
            gain = entropy - self.inner_entropy(data, target, attributes[a], classes, index)
            if gain >= info_gain['info_gain']:
                info_gain['attribute'] = a
                info_gain['info_gain'] = gain
        return info_gain

    def entropy(self, target, classes):
        entropy = 0.0
        n = len(target)
        for c in classes:
            count_c = 0
            for t in target:
                if t == c:
                    count_c += 1
            if count_c != 0:
                p_k = count_c/n
            else:
                return 0.0
            entropy += -(p_k*math.log(p_k, 2))
        return entropy

    def inner_entropy(self, data, target, attribute_vals, classes, index):
        inner_entropy = 0.0
        for a in attribute_vals:
            entropy = 0.0
            denominator = 0
            for i in range(len(data)):
                if data[i][index] == a:
                    denominator += 1
            if denominator == 0:
                break
            for c in classes:
                numerator = 0
                for t in range(len(target)):
                    if c == target[t] and data[t][index] == a:
                        numerator += 1
                if numerator != 0:
                    entropy += -(numerator/denominator)*math.log(numerator/denominator, 2)
                else:
                    entropy = 0.0
                    break
            inner_entropy += (denominator/len(data))*entropy
        return inner_entropy

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes, map_attr, parent_id):
        # fill in something more sensible here... root should become the output of the recursive tree creation
        root = self.new_ID3_node()
        root['nodes'] = {}
        root['samples'] = len(data)
        root['classCounts'] = self.class_counter(target, classes)
        entropy = self.entropy(target, classes)
        root['entropy'] = entropy
        if self.belong_to_same_class(target):
            root['label'] = target[0]
            self.add_node_to_graph(root, parent_id)
            return root
        if not attributes:
            root['label'] = self.most_common_class(target, classes)
            self.add_node_to_graph(root, parent_id)
            return root
        else:
            a_max_info_gain = self.find_split_attr(data, target, attributes, classes, entropy, map_attr)
            att = a_max_info_gain['attribute']
            root['attribute'] = att
            index = map_attr[att]
            for v in attributes[att]:
                samples = self.samples_per_attribute(data, target, v, index)
                samples_v = samples['data']
                target_v = samples['target']
                if not samples_v:
                    leaf = self.new_ID3_node()
                    leaf['label'] = self.most_common_class(target, classes)
                    leaf['samples'] = 0
                    leaf['classCounts'] = {}
                    root['nodes'][leaf['id']] = v
                    self.add_node_to_graph(leaf, root['id'])
                else:
                    new_atts = self.new_attributes(attributes, att)
                    sub_tree = self.fit(samples_v, target_v, new_atts, classes, map_attr, root['id'])
                    root['nodes'][sub_tree['id']] = v
        self.add_node_to_graph(root, parent_id)
        return root

    def class_counter(self, target, classes):
        counter = {}
        for c in classes:
            class_count = 0
            for t in target:
                if c == t:
                    class_count += 1
            counter[c] = class_count
        return counter

    def map_attr_index(self, attributes):
        i = 0
        d = {}
        for a in attributes:
            d[a] = i
            i += 1
        return d

    def belong_to_same_class(self, target):
        t = target[0]
        for i in range(1, len(target)):
            if t != target[i]:
                return False
        return True

    def new_attributes(self, attributes, a):
        d = {}
        for att in attributes:
            if att != a:
                d[att] = attributes[att]
        return d

    def most_common_class(self, target, classes):
        most_common = {i: 0 for i in classes}
        for i in range(len(target)):
            most_common[target[i]] += 1
        val = list(most_common.values())
        return list(most_common.keys())[val.index(max(val))]

    def samples_per_attribute(self, data, target, a, index):
        samples = {'data': [], 'target': []}
        for i in range(len(data)):
            if data[i][index] == a:
                samples['data'].append(data[i])
                samples['target'].append(target[i])
        return samples

    def predict(self, data, map_attr):
        predicted = list()
        nodes = self.nodes
        for d in data:
            stop = False
            node_id = 0
            while not stop:
                node = nodes[node_id]
                if not node['nodes']:
                    stop = True
                    predicted.append(node['label'])
                else:
                    index = map_attr[node['attribute']]
                    for i in node['nodes']:
                        if node['nodes'][i] == d[index]:
                            node_id = i
                            break

        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted
