import math
import json
import queue as q


class Node:
    def get_node_type(self):
        return self.node_type


class ContinuousNode(Node):
    node_type = 'Continuous'

    def __init__(self, node_id=None, feature_index=None, feature_value=None, max_label=None):
        self.node_id = node_id
        self.feature_index = feature_index
        self.feature_value = feature_value
        self.max_label = max_label
        self.temp_feature0 = None
        self.temp_feature1 = None
        self.temp_label0 = None
        self.temp_label1 = None
        self.left_child = None
        self.right_child = None

    def serialize(self):
        node_dict = {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'feature_index': self.feature_index,
            'feature_value': self.feature_value,
            'max_label': self.max_label,
            'left_child_id': self.left_child.node_id,
            'right_child_id': self.right_child.node_id}
        return node_dict

    def deserialize(self, node_dict):
        self.node_id = node_dict['node_id']
        self.feature_index = node_dict['feature_index']
        self.feature_value = node_dict['feature_value']
        self.max_label = node_dict['max_label']


class DiscreteNode(Node):
    node_type = 'Discrete'

    def __init__(self, node_id=None, feature_index=None, feature_list=None, max_label=None):
        self.node_id = node_id
        self.feature_index = feature_index
        self.feature_list = feature_list
        self.max_label = max_label
        self.temp_features = None
        self.temp_labels = None
        self.child_list = []

    def serialize(self):
        child_id_list = []
        for i in range(len(self.child_list)):
            child_id_list.append(self.child_list[i].node_id)
        node_dict = {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'feature_index': self.feature_index,
            'feature_list': self.feature_list,
            'max_label': self.max_label,
            'child_id_list': child_id_list}
        return node_dict

    def deserialize(self, node_dict):
        self.node_id = node_dict['node_id']
        self.feature_index = node_dict['feature_index']
        for i in range(len(node_dict['feature_list'])):
            self.feature_list.append(node_dict['feature_list'][i])
        self.max_label = node_dict['max_label']


class LeafNode(Node):
    node_type = 'Leaf'

    def __init__(self, node_id=None, label=None):
        self.node_id = node_id
        self.label = label

    def serialize(self):
        node_dict = {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'label': self.label}
        return node_dict

    def deserialize(self, node_dict):
        self.node_id = node_dict['node_id']
        self.label = node_dict['label']


class Tree:
    def __init__(self):
        self.node_id = 0
        self.root = None

    def serialize(self):
        node_list = []
        node_queue = q.Queue()
        node_queue.put(self.root)
        while node_queue.empty() is False:
            node = node_queue.get()
            if node.get_node_type() == 'Continuous':
                if node.left_child is not None:
                    node_queue.put(node.left_child)
                if node.right_child is not None:
                    node_queue.put(node.right_child)
            elif node.get_node_type() == 'Discrete':
                for child in node.child_list:
                    node_queue.put(child)
            node_list.append(node.serialize())
        return node_list

    def deserialize(self, node_list):
        node_map = dict()
        for i in range(len(node_list)):
            node_map[node_list[i]['node_id']] = i
        node_queue = q.Queue()
        if node_list[node_map[0]]['node_type'] == 'Continuous':
            self.root = ContinuousNode(node_id=0)
        elif node_list[node_map[0]]['node_type'] == 'Discrete':
            self.root = DiscreteNode(node_id=0)
        elif node_list[node_map[0]]['node_type'] == 'Leaf':
            self.root = LeafNode(node_id=0)
        else:
            raise Exception('Deserialize error: node type error in root node.')

        self.root.deserialize(node_list[node_map[0]])
        node_queue.put(self.root)
        while node_queue.empty() is False:
            node = node_queue.get()
            if node.get_node_type() == 'Continuous':
                left_child_id = node_list[node_map[node.node_id]]['left_child_id']
                if node_list[node_map[left_child_id]]['node_type'] == 'Continuous':
                    left_child = ContinuousNode()
                elif node_list[node_map[left_child_id]]['node_type'] == 'Discrete':
                    left_child = DiscreteNode()
                elif node_list[node_map[left_child_id]]['node_type'] == 'Leaf':
                    left_child = LeafNode()
                else:
                    raise Exception('Deserialize error: node type error in node %d' % node_list[node_map[left_child_id]]['node_id'])
                left_child.deserialize(node_list[node_map[left_child_id]])
                node.left_child = left_child
                node_queue.put(left_child)
                right_child_id = node_list[node_map[node.node_id]]['right_child_id']
                if node_list[node_map[right_child_id]]['node_type'] == 'Continuous':
                    right_child = ContinuousNode()
                elif node_list[node_map[right_child_id]]['node_type'] == 'Discrete':
                    right_child = DiscreteNode()
                elif node_list[node_map[right_child_id]]['node_type'] == 'Leaf':
                    right_child = LeafNode()
                else:
                    raise Exception('Deserialize error: node type error in node %d' % node_list[node_map[right_child_id]]['node_id'])
                right_child.deserialize(node_list[node_map[right_child_id]])
                node.right_child = right_child
                node_queue.put(right_child)
            elif node.get_node_type() == 'Discrete':
                child_id_list = node_list[node_map[node.node_id]]['child_id_list']
                for i in range(len(child_id_list)):
                    if node_list[node_map[child_id_list[i]]]['node_type'] == 'Continuous':
                        child_node = ContinuousNode()
                    elif node_list[node_map[child_id_list[i]]]['node_type'] == 'Discrete':
                        child_node = DiscreteNode()
                    elif node_list[node_map[child_id_list[i]]]['node_type'] == 'Leaf':
                        child_node = LeafNode()
                    else:
                        raise Exception('Deserialize error: node type error in node %d' % node_list[node_map[child_id_list[i]]]['node_id'])
                    child_node.deserialize(node_list[node_map[node.node_id]])
                    node.child_list.append(child_node)
                    node_queue.put(child_node)

    def save(self, file_name, mode='w'):
        with open(file_name, mode) as f:
            json.dump(self.serialize(), f)

    def load(self, file_name):
        with open(file_name, 'r') as f:
            self.deserialize(json.load(f))

    def get_next_node_id(self):
        self.node_id += 1
        return self.node_id - 1


class DecisionTree(Tree):
    def __init__(self):
        super(DecisionTree, self).__init__()
        self.feature_index = None
        self.feature_attribute = None
        self.feature_list = None
        self.feature_map = None
        self.feature_dim = None
        self.label_dim = None

    def get_tree_type(self):
        return self.decision_tree_type

    def generate_leaf_node(self, feature, label, used_feature_index):
        data_num = len(feature)
        flag = True
        for i in range(1, data_num):
            if label[i] != label[0]:
                flag = False
                break

        if flag:
            return LeafNode(self.get_next_node_id(), label=label[0])
        else:
            flag = True
            for i in range(0, self.feature_dim):
                if i not in used_feature_index:
                    for j in range(1, data_num):
                        if feature[j][i] != feature[0][i]:
                            flag = False
                            break
                    if flag is False:
                        break
            if flag:
                cnt = [0 for _ in range(self.label_dim)]
                for i in range(data_num):
                    cnt[label[i]] += 1
                max_index = 0
                max_cnt = cnt[0]
                for i in range(1, self.label_dim):
                    if cnt[i] > max_cnt:
                        max_cnt = cnt[i]
                        max_index = i
                return LeafNode(self.get_next_node_id(), label=max_index)

        return None

    def test(self, feature, label):
        data_num = len(feature)
        if data_num != len(label):
            raise Exception('Test error: feature num and label num not match.')
        feature_dim = len(feature[0])
        if feature_dim <= 0:
            raise Exception('Test error: feature dim must be greater than 0.')
        if self.root is None:
            raise Exception('Test error: decision tree not initialized.')

        cnt = 0
        result = [-1 for _ in range(data_num)]
        for i in range(data_num):
            node = self.root
            while True:
                if node.node_type == 'Continuous':
                    if feature[i][node.feature_index] <= node.feature_value:
                        node = node.left_child
                    else:
                        node = node.right_child
                elif node.node_type == 'Discrete':
                    if feature[i][node.feature_index] in node.feature_list:
                        node = node.child_list[node.feature_list.index(feature[i][node.feature_index])]
                    else:
                        result[i] = node.max_label
                        if node.max_label == label[i]:
                            cnt += 1
                        break
                else:
                    result[i] = node.label
                    if node.label == label[i]:
                        cnt += 1
                    break

        return cnt, result

    def create_node(self, feature, label, used_feature_index, continuous_gain, discrete_gain):
        data_num = len(feature)
        if data_num == 0:
            raise Exception('Create node error: data num equals to 0.')

        leaf_node = self.generate_leaf_node(feature, label, used_feature_index)
        if leaf_node is not None:
            return leaf_node

        flag = [False for _ in range(self.feature_dim)]
        threshold = [[] for _ in range(self.feature_dim)]
        feature_cnt = [[] for _ in range(self.feature_dim)]
        for i in range(self.feature_dim):
            if i not in used_feature_index:
                for j in range(1, data_num):
                    if feature[j][i] != feature[0][i]:
                        flag[i] = True
                        break
                if self.feature_attribute[i] == 'Continuous':
                    for j in range(data_num):
                        if feature[j][i] not in threshold[i]:
                            threshold[i].append(feature[j][i])
                    threshold[i].sort()
                    threshold_len = len(threshold[i]) - 1
                    for j in range(threshold_len):
                        threshold[i][j] = (threshold[i][j] + threshold[i][j + 1]) / 2
                else:
                    feature_num = len(self.feature_list[i])
                    feature_cnt[i] = [[0 for _ in range(self.label_dim)] for __ in range(feature_num)]
                    for j in range(data_num):
                        feature_cnt[i][self.feature_map[i][feature[j][i]]][label[j]] += 1

        best_gain = None
        best_index = None
        best_value = None
        for i in range(self.feature_dim):
            if i not in used_feature_index and flag[i]:
                if self.feature_attribute[i] == 'Continuous':
                    cnt = [[0 for _ in range(self.label_dim)], [0 for _ in range(self.label_dim)]]
                    temp_data = []
                    for j in range(data_num):
                        temp_data.append([feature[j][i], label[j]])
                    temp_data.sort(key=lambda x: x[0])
                    threshold_index = 0
                    for j in range(data_num):
                        if temp_data[j][0] <= threshold[i][0]:
                            cnt[0][temp_data[j][1]] += 1
                        else:
                            for k in range(j, data_num):
                                cnt[1][temp_data[k][1]] += 1
                            threshold_index = j
                            break
                    gain = continuous_gain(cnt, self.label_dim, data_num)
                    if best_gain is None or gain > best_gain:
                        best_gain = gain
                        best_index = i
                        best_value = threshold[i][0]
                    threshold_len = len(threshold[i]) - 1
                    for j in range(1, threshold_len):
                        while temp_data[threshold_index][0] <= threshold[i][j]:
                            cnt[0][temp_data[threshold_index][1]] += 1
                            cnt[1][temp_data[threshold_index][1]] -= 1
                            threshold_index += 1
                        gain = continuous_gain(cnt, self.label_dim, data_num)
                        if best_gain is None or gain > best_gain:
                            best_gain = gain
                            best_index = i
                            best_value = threshold[i][j]
                elif self.feature_attribute[i] == 'Discrete':
                    gain = discrete_gain(feature_cnt[i], self.label_dim, data_num)
                    if best_gain is None or gain > best_gain:
                        best_gain = gain
                        best_index = i

        if best_index is not None:
            _cnt = [0 for _ in range(self.label_dim)]
            for i in range(data_num):
                _cnt[label[i]] += 1
            max_label = 0
            max_cnt = _cnt[0]
            for i in range(1, self.label_dim):
                if _cnt[i] > max_cnt:
                    max_cnt = _cnt[i]
                    max_label = i
            if self.feature_attribute[best_index] == 'Continuous':
                node = ContinuousNode(node_id=self.get_next_node_id(), feature_index=self.feature_index[best_index], feature_value=best_value, max_label=max_label)
                node.used_feature_index = list(used_feature_index)
                node.used_feature_index.append(best_index)
                node.temp_feature0 = []
                node.temp_feature1 = []
                node.temp_label0 = []
                node.temp_label1 = []
                for j in range(data_num):
                    if feature[j][best_index] <= best_value:
                        node.temp_feature0.append(feature[j])
                        node.temp_label0.append(label[j])
                    else:
                        node.temp_feature1.append(feature[j])
                        node.temp_label1.append(label[j])
                if len(node.temp_feature0) == 0:
                    node.left_child = LeafNode(self.get_next_node_id(), label=max_label)
                if len(node.temp_feature1) == 0:
                    node.right_child = LeafNode(self.get_next_node_id(), label=max_label)
                return node
            else:
                node = DiscreteNode(node_id=self.get_next_node_id(), feature_index=self.feature_index[best_index], feature_list=self.feature_list[best_index], max_label=max_label)
                node.used_feature_index = list(used_feature_index)
                node.used_feature_index.append(best_index)
                feature_num = len(self.feature_list[best_index])
                node.child_list = [None for _ in range(feature_num)]
                node.temp_features = [[] for _ in range(feature_num)]
                node.temp_labels = [[] for _ in range(feature_num)]
                for i in range(data_num):
                    node.temp_features[self.feature_map[best_index][feature[i][best_index]]].append(feature[i])
                    node.temp_labels[self.feature_map[best_index][feature[i][best_index]]].append(label[i])
                for i in range(feature_num):
                    if len(node.temp_features[i]) == 0:
                        node.child_list[i] = LeafNode(self.get_next_node_id(), label=max_label)
                return node
        else:
            raise Exception('Logic error: best gain is None.')

    def build_decision_tree(self, feature, label, feature_index, feature_attribute, feature_list):
        if self.continuous_gain is None or self.discrete_gain is None:
            raise Exception('Logic error: [continuous_gain | discrete_gain] is not implemented in decision tree.\n'
                            'Make sure not call class DecisionTree() and function [continuous_gain & discrete_gain]must be implemented.')
        if len(feature[0]) != len(feature_index):
            raise Exception('Logic error: feature index dimension not match.')

        self.feature_index = feature_index
        self.feature_attribute = feature_attribute
        self.feature_list = feature_list
        self.feature_dim = len(feature_index)
        self.label_dim = len(set(label))
        self.feature_map = [dict() for _ in range(self.feature_dim)]
        for i in range(self.feature_dim):
            feature_num = len(feature_list[i])
            for j in range(feature_num):
                self.feature_map[i][feature_list[i][j]] = j
        self.root = self.create_node(feature, label, [], self.continuous_gain, self.discrete_gain)
        node_queue = q.Queue()
        node_queue.put(self.root)

        while node_queue.empty() is False:
            node = node_queue.get()
            if node.get_node_type() == 'Continuous':
                if len(node.temp_feature0) != 0:
                    left_child_node = self.create_node(node.temp_feature0, node.temp_label0, node.used_feature_index, self.continuous_gain, self.discrete_gain)
                    node.left_child = left_child_node
                    if left_child_node.get_node_type() != 'Leaf':
                        node_queue.put(left_child_node)
                if len(node.temp_feature1) != 0:
                    right_child_node = self.create_node(node.temp_feature1, node.temp_label1, node.used_feature_index, self.continuous_gain, self.discrete_gain)
                    node.right_child = right_child_node
                    if right_child_node.get_node_type() != 'Leaf':
                        node_queue.put(right_child_node)
                node.temp_feature0 = None
                node.temp_feature1 = None
                node.temp_label0 = None
                node.temp_label1 = None
            elif node.get_node_type() == 'Discrete':
                for i in range(len(node.feature_list)):
                    if len(node.temp_features[i]) != 0:
                        child_node = self.create_node(node.temp_features[i], node.temp_labels[i], node.used_feature_index, self.continuous_gain, self.discrete_gain)
                        node.child_list[i] = child_node
                        if child_node.get_node_type() != 'Leaf':
                            node_queue.put(child_node)
                    node.temp_features[i] = None
                    node.temp_labels[i] = None
            else:
                raise Exception('Logic error: node type must be [Continuous | Discrete]')

        self.feature_index = None
        self.feature_attribute = None
        self.feature_list = None
        self.feature_map = None


class ID3Tree(DecisionTree):
    decision_tree_type = 'ID3'

    def __init__(self):
        super(ID3Tree, self).__init__()

    @staticmethod
    def continuous_gain(cnt, label_dim, data_num):
        gain0 = 0
        gain1 = 0
        num0 = 0
        num1 = 0
        for i in range(label_dim):
            num0 += cnt[0][i]
            num1 += cnt[1][i]
        for i in range(label_dim):
            if cnt[0][i] != 0:
                p = cnt[0][i] / num0
                gain0 += p * math.log2(p)
            if cnt[1][i] != 0:
                p = cnt[1][i] / num1
                gain1 += p * math.log2(p)
        return gain0 * (num0 / data_num) + gain1 * (num1 / data_num)

    @staticmethod
    def discrete_gain(cnt, label_dim, data_num):
        class_num = len(cnt)
        gain = [0 for _ in range(class_num)]
        num = [0 for _ in range(class_num)]
        for i in range(class_num):
            for j in range(label_dim):
                num[i] += cnt[i][j]
        for i in range(class_num):
            for j in range(label_dim):
                if cnt[i][j] != 0:
                    p = cnt[i][j] / num[i]
                    gain[i] += p * math.log2(p)
        result = 0
        for i in range(class_num):
            result += gain[i] * (num[i] / data_num)
        return result


class CartTree(DecisionTree):
    decision_tree_type = 'Cart'

    def __init__(self):
        super(CartTree, self).__init__()

    @staticmethod
    def continuous_gain(cnt, label_dim, data_num):
        gain0 = -1
        gain1 = -1
        num0 = 0
        num1 = 0
        for i in range(label_dim):
            num0 += cnt[0][i]
            num1 += cnt[1][i]
        for i in range(label_dim):
            p = cnt[0][i] / num0
            gain0 += p * p
            p = cnt[1][i] / num1
            gain1 += p * p
        return gain0 * (num0 / data_num) + gain1 * (num1 / data_num)

    @staticmethod
    def discrete_gain(cnt, label_dim, data_num):
        class_num = len(cnt)
        gain = [-1 for _ in range(class_num)]
        num = [0 for _ in range(class_num)]
        for i in range(class_num):
            for j in range(label_dim):
                num[i] += cnt[i][j]
        for i in range(class_num):
            for j in range(label_dim):
                p = cnt[i][j] / num[i]
                gain[i] += p * p
        result = 0
        for i in range(class_num):
            result += gain[i] * (num[i] / data_num)
        return result


class C4_5Tree(DecisionTree):
    decision_tree_type = 'C4.5'

    def __init__(self):
        super(C4_5Tree, self).__init__()
