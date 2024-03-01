
import numpy as np

class TreeNode:
    def __init__(self, threshold=None, parent=None, left_child=None, right_child=None, label=-1):
        self.threshold = threshold
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.label = label


def calculate_gini_impurity(data, labels, threshold):
    # Function to calculate Gini impurity
    left_labels = labels[data < threshold]
    right_labels = labels[data >= threshold]

    # left gini=p ,right gini=q
    p = np.sum(left_labels) / len(left_labels)
    q = np.sum(right_labels) / len(right_labels)

    return len(left_labels) * p * (1 - p) + len(right_labels) * q * (1 - q)


def find_best_split(data, labels):
    # Function to find the best split based on Gini impurity
    # return np.median(data)
    best_gini_impurity = float('inf')
    best_threshold = None

    for threshold in data:
        gini_impurity = calculate_gini_impurity(data, labels, threshold)

        if gini_impurity < best_gini_impurity:
            best_gini_impurity = gini_impurity
            best_threshold = threshold
    print(best_threshold)
    return best_threshold


def build_decision_tree(data, labels, alpha=8):
    # Recursive function to build the decision tree
    if len(set(labels)) == 1 or len(data) <= alpha:
        labels_count = dict(zip(*np.unique(labels, return_counts=True)))
        return TreeNode(threshold=None, label=max(labels_count, key=labels_count.get))

    threshold = find_best_split(data, labels)

    left_data = data[data < threshold]
    right_data = data[data >= threshold]

    left_child = build_decision_tree(left_data, labels[data < threshold], alpha)
    right_child = build_decision_tree(right_data, labels[data >= threshold], alpha)

    node = TreeNode(threshold=threshold, left_child=left_child, right_child=right_child)

    return node


def predict_label(tree, new_data_point):
    # Function to predict the label for a new data point using the decision tree
    current_node = tree

    while current_node.threshold is not None:
        if new_data_point < current_node.threshold:
            current_node = current_node.left_child
        else:
            current_node = current_node.right_child
    return current_node.label



