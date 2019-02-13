import os
import csv
import random
import re


def get_map(labels):
    map_dict = dict()
    remap_dict = dict()
    label_dim = len(labels)
    for i in range(label_dim):
        if str(i) in labels:
            map_dict[str(i)] = i
            remap_dict[i] = str(i)
        else:
            return None, None
    return map_dict, remap_dict


def load_map_file(map_file, labels):
    map_dict = dict()
    remap_dict = dict()
    with open(map_file, 'r') as f:
        pattern1 = re.compile(r'([\s]*[\S]+)[\s]+(0|[1-9][0-9]*)[\s]*')
        pattern2 = re.compile(r'[\s]*([\S]+)[\s]*')
        false_pattern = re.compile(r'[\S]*([\s]+)')
        num_pattern = re.compile(r'[\S]*(0|[1-9][0-9]*)[\S]*')
        line_index = 0
        for line in f:
            line_index += 1
            line = line.strip('\n')
            result = pattern1.match(line)
            if result:
                index_key = result.group(1)
                index_value = result.group(2)
                if num_pattern.match(index_value) is None:
                    raise Exception('Map file error: illegal format in line %d:\n%s' % (line_index, line))
                index_value = int(index_value)
            else:
                if false_pattern.match(line):
                    raise Exception('Map file error: illegal format in line %d:\n%s' % (line_index, line))
                else:
                    result = pattern2.match(line)
                    if result:
                        index_key = result.group(1)
                        index_value = line_index - 1
            if index_key in map_dict and map_dict[index_key] != index_value:
                raise Exception('Map file error: different map value for key \"%s\" in line %d:\n%s' % (index_key, line_index, line))
            if index_value in remap_dict:
                raise Exception('Map file error: the same map value %d in line %d:\n%s' % (index_value, line_index, line))
            map_dict[index_key] = index_value
            remap_dict[index_value] = index_key
        for label in labels:
            if label not in map_dict:
                raise Exception('Map file error: label \"%s\" not found in map file' % label)
    return map_dict, remap_dict


def load_feature_attribute_file(feature_info_file, feature_dim):
    feature_attribute = [None for _ in range(feature_dim)]
    continuous_pattern1 = re.compile(r'[\s]*continuous[\s]*', re.I)
    continuous_pattern2 = re.compile(r'[\s]*(0|[1-9][0-9]*)[\s]+continuous[\s]*', re.I)
    discrete_pattern1 = re.compile(r'[\s]*discrete[\s]*', re.I)
    discrete_pattern2 = re.compile(r'[\s]*(0|[1-9][0-9]*)[\s]+discrete[\s]*', re.I)
    illegal_pattern = re.compile(r'[\s]*[\S]+[\s]*')
    line_index = 0
    with open(feature_info_file, 'r') as f:
        for line in f:
            line_index += 1
            if continuous_pattern1.match(line):
                if line_index <= feature_dim:
                    feature_attribute[line_index - 1] = 'Continuous'
                else:
                    print('Warning: feature dimension is %d. Line[%d]: \'%s\' is invalid.' % (feature_dim, line_index, line))
            elif discrete_pattern1.match(line):
                if line_index <= feature_dim:
                    feature_attribute[line_index - 1] = 'Discrete'
                else:
                    print('Warning: feature dimension is %d. Line[%d]: \'%s\' is invalid.' % (feature_dim, line_index, line))
            else:
                result = continuous_pattern2.match(line)
                if result:
                    index = int(result.group(1))
                    if index < feature_dim:
                        feature_attribute[index] = 'Continuous'
                    else:
                        print('Warning: feature dimension is %d. Line[%d]: \'%s\' is invalid.' % (feature_dim, line_index, line))
                else:
                    result = discrete_pattern2.match(line)
                    if result:
                        index = int(result.group(1))
                        if index < feature_dim:
                            feature_attribute[index] = 'Discrete'
                        else:
                            print('Warning: feature dimension is %d. Line[%d]: \'%s\' is invalid.' % (feature_dim, line_index, line))
                    else:
                        result = illegal_pattern.match(line)
                        if result:
                            print('Warning: illegal format in line[%d]: \'%s\', which will be ignored.' % (line_index, line))

    for i in range(feature_dim):
        if feature_attribute[i] is None:
            raise Exception('Feature attribute error: feature[%d]\'s attribute is not specific.' % i)
    return feature_attribute


def load_csv_data(csv_file_path, csv_header=True, map_file_path=None, feature_attribute_file_path=None):
    feature = []
    label = []
    data_num = 0
    labels = set()
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        flag = True
        for row in reader:
            if csv_header and flag:
                flag = False
            else:
                feature.append(row[0:-1])
                label.append(row[-1])
                labels.add(label[data_num])
                data_num += 1
    assert data_num > 0
    map_dict, remap_dict = get_map(labels)
    if map_dict is None:
        if map_file_path is None or os.path.exists(map_file_path) is False:
            raise Exception('Map file error: label is required as [0, ... , (label_num - 1)]\nMap file not found.')
        map_dict, remap_dict = load_map_file(map_file_path, labels)
    for i in range(data_num):
        label[i] = map_dict[label[i]]

    feature_dim = len(feature[0])
    if feature_attribute_file_path is None or os.path.exists(feature_attribute_file_path) is False:
        feature_attribute = ['Continuous' for _ in range(feature_dim)]
    else:
        feature_attribute = load_feature_attribute_file(feature_attribute_file_path, feature_dim)
    feature_list = [[] for _ in range(feature_dim)]
    label_dim = len(labels)
    print('Data num =', data_num)
    print('Data dim =', feature_dim)
    print('Label dim =', label_dim)
    for i in range(data_num):
        if len(feature[i]) != len(feature[0]):
            raise Exception('Data file error: feature dimension not match at feature[%d].' % i)
        for j in range(feature_dim):
            if feature_attribute[j] == 'Continuous':
                try:
                    feature[i][j] = float(feature[i][j])
                except:
                    print('Data error: feature[%d][%d] is not a real number.' % (i, j))
            else:
                if feature[i][j] not in feature_list[j]:
                    feature_list[j].append(feature[i][j])

    return feature, label, feature_attribute, feature_list, map_dict, remap_dict, data_num, feature_dim, label_dim


def generate_training_batch(feature, data_num, feature_dim, dim, feature_attribute, feature_list):
    assert feature_dim >= dim
    feature_index = [0 for _ in range(dim)]
    sample_feature_attribute = [None for _ in range(dim)]
    sample_feature_list = [[] for _ in range(dim)]
    for i in range(dim):
        while True:
            index = random.randint(0, feature_dim - 1)
            if index not in feature_index:
                feature_index[i] = index
                sample_feature_attribute[i] = feature_attribute[index]
                sample_feature_list[i] = feature_list[index]
                break
    sample_feature = [[0 for _ in range(dim)] for __ in range(data_num)]

    for i in range(data_num):
        for j in range(dim):
            sample_feature[i][j] = feature[i][feature_index[j]]

    return sample_feature, feature_index, sample_feature_attribute, sample_feature_list
