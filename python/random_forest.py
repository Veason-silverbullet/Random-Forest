import math
import random
import json
import utility
import decision_tree as dt


class RandomForest:
    random_forest_type_candidates = ['ID3', 'CART', 'Mixture']

    def __init__(self, random_forest_type=None, random_dim=None, tree_num=None):
        self.random_forest_type = random_forest_type if random_forest_type is not None else 'ID3'
        self.random_dim = random_dim
        self.tree_num = tree_num
        self.decision_trees = []
        self.feature = None
        self.label = None
        self.data_num = None
        self.feature_attribute = None
        self.feature_list = None
        self.feature_dim = None
        self.label_dim = None

    def get_random_forest_type(self):
        return self.random_forest_type

    def set_random_forest_type(self, random_forest_type):
        self.random_forest_type = random_forest_type

    def set_tree_num(self, tree_num):
        self.tree_num = tree_num

    def set_random_dim(self, random_dim):
        self.random_dim = random_dim

    def load_data(self, feature, label, feature_attribute, feature_list, feature_dim, label_dim):
        if len(feature) == 0:
            raise Exception('Logic error: feature is empty')
        else:
            self.feature = feature
            self.data_num = len(feature)
        if len(label) < self.data_num:
            raise Exception('Logic error: label num is less than feature num')
        else:
            self.label = label
        if len(feature_attribute) < feature_dim or len(feature[0]) < feature_dim:
            raise Exception('Logic error: feature dim error')
        for attribute in feature_attribute:
            if attribute != 'Continuous' and attribute != 'Discrete':
                raise Exception('Logic error: feature attribute must be one type of [Continuous | Discrete]')
        for i in range(len(feature_attribute)):
            if feature_attribute[i] == 'Discrete' and (feature_list[i] is None or len(feature_list[i]) == 0):
                raise Exception('Data error: discrete feature[%d]\' class num is not specific.' % i)
        self.feature_attribute = feature_attribute
        self.feature_list = feature_list
        self.feature_dim = feature_dim
        self.label_dim = label_dim

    def build_random_forest(self):
        if self.feature is None or self.label is None or self.feature_attribute is None:
            raise Exception('Logic error: [feature | label | feature_attribute] is None.')
        for i in range(len(self.feature_attribute)):
            if self.feature_attribute[i] == 'Discrete' and (self.feature_list[i] is None or len(self.feature_list[i]) == 0):
                raise Exception('Data error: discrete feature[%d]\' class num is not specific.' % i)
        if self.random_dim is None:
            self.random_dim = math.ceil(math.sqrt(self.feature_dim))
        if self.tree_num is None:
            self.tree_num = math.ceil(math.sqrt(self.data_num) * math.log2(self.data_num))

        for i in range(self.tree_num):
            sample_feature, feature_index, sample_feature_attribute, sample_feature_list = \
                utility.generate_training_batch(self.feature, self.data_num, self.feature_dim, self.random_dim, self.feature_attribute, self.feature_list)
            if self.random_forest_type == 'ID3':
                decision_tree = dt.ID3Tree()
            elif self.random_forest_type == 'CART':
                decision_tree = dt.CartTree()
            else:
                if random.randint(0, 1) == 0:
                    decision_tree = dt.ID3Tree()
                else:
                    decision_tree = dt.CartTree()
            decision_tree.build_decision_tree(sample_feature, self.label, feature_index, sample_feature_attribute, sample_feature_list)
            decision_tree.tree_id = i
            self.decision_trees.append(decision_tree)

    def test(self, feature, label, feature_attribute, feature_dim, label_dim):
        if len(feature) == 0:
            raise Exception('Logic error: feature is empty')
        else:
            data_num = len(feature)
        if len(label) < data_num:
            raise Exception('Logic error: label num is less than feature num')
        if len(feature_attribute) < feature_dim or len(feature[0]) < feature_dim:
            raise Exception('Logic error: test feature dim error')
        for i in range(len(feature_attribute)):
            if feature_attribute[i] != 'Continuous' and feature_attribute[i] != 'Discrete':
                raise Exception('Logic error: feature attribute must be one type of [Continuous | Discrete]')
            if feature_attribute[i] != self.feature_attribute[i]:
                raise Exception('Logic error: test feature attribute not match')

        sum_result = [[0 for _ in range(label_dim)] for __ in range(data_num)]
        for i in range(self.tree_num):
            cnt, result = self.decision_trees[i].test(feature, label)
            for j in range(data_num):
                sum_result[j][result[j]] += 1
        # print(sum_result)
        test_cnt = 0
        prediction = [0 for _ in range(data_num)]
        for i in range(data_num):
            prediction[i] = 0
            max_result = sum_result[i][0]
            for j in range(label_dim):
                if sum_result[i][j] > max_result:
                    max_result = sum_result[i][j]
                    prediction[i] = j
            if prediction[i] == label[i]:
                test_cnt += 1
        print('acc =', test_cnt / data_num)

    def load(self, file_name):
        with open(file_name, 'r') as f:
            forest_dict = json.load(f)
            self.set_random_forest_type(forest_dict['forest_type'])
            self.set_tree_num(forest_dict['tree_num'])
            tree_dict_list = forest_dict['trees']
            if self.tree_num != len(tree_dict_list):
                raise Exception('Tree num error: expected tree num is ' + str(self.tree_num) + ', but ' + str(len(tree_dict_list)) + ' was found in tree list.')
            self.decision_trees = [None for _ in range(self.tree_num)]
            for i in range(self.tree_num):
                tree_id = tree_dict_list[i]['tree_id']
                if tree_dict_list[i]['tree_type'] == 'ID3':
                    self.decision_trees[tree_id] = dt.ID3Tree()
                elif tree_dict_list[i]['tree_type'] == 'CART':
                    self.decision_trees[tree_id] = dt.CartTree()
                elif tree_dict_list[i]['tree_type'] == 'C4.5':
                    self.decision_trees[tree_id] = dt.C4_5Tree()
                else:
                    raise Exception('Random forest type error: randomForestType must be [ID3 | Cart | Mixture].')
                self.decision_trees[tree_id].tree_id = tree_id
                self.decision_trees[tree_id].deserialize(tree_dict_list[i]['nodes'])

    def save(self, file_name):
        with open(file_name, 'w') as f:
            tree_dict_list = list()
            for i in range(self.tree_num):
                tree_dict = {
                    'tree_id': self.decision_trees[i].tree_id,
                    'tree_type': self.decision_trees[i].get_tree_type(),
                    'nodes': self.decision_trees[i].serialize()
                }
                tree_dict_list.append(tree_dict)
            forest_dict = {
                'forest_type': self.get_random_forest_type(),
                'tree_num': self.tree_num,
                'trees': tree_dict_list
            }
            json.dump(forest_dict, f)
