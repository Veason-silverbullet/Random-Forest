import time
import math
import utility
import random_forest as rf

display_time = True  # Determine whether display the used time

training_data_file = '../data/data.csv'  # Training csv file path
testing_data_file = '../data/data.csv'  # Testing csv file path
training_csv_header = True  # Determine whether discard training csv file header
testing_csv_header = True  # Determine whether discard testing csv file header
mapping_file = '../data/index.txt'  # Map file path, to convert the unordered label to ordered label as [0, 1, 2,..., N - 1], supposing N labels
feature_attribute_file = '../data/feature_attribute.txt'  # Feature attribute file path, to determine the feature attribute is continuous or discrete

random_forest_type = 'Mixture'  # Random_forest_type can be 'ID3' or 'Cart' or 'Mixture'
feature_use_rate = 0.75  # Feature use rate in random forest. To use (feature_use_rate * N) of N features in building decision tree.


if __name__ == '__main__':
    if display_time:
        start_time = time.time()

    # Training phase
    training_feature, training_label, feature_attribute, feature_list, map_dict, remap_dict, training_data_num, training_feature_dim, training_label_dim = \
        utility.load_csv_data(training_data_file, csv_header=training_csv_header, map_file_path=mapping_file, feature_attribute_file_path=feature_attribute_file)
    random_forest = rf.RandomForest()
    feature_dim = len(training_feature[0])
    random_forest.set_random_forest_type(random_forest_type)
    random_forest.set_random_dim(int(max(min(feature_dim, math.ceil(feature_dim * feature_use_rate)), 1)))
    random_forest.load_data(training_feature, training_label, feature_attribute, feature_list, training_feature_dim, training_label_dim)
    random_forest.build_random_forest()

    # Testing phase
    testing_feature, testing_label, feature_attribute, feature_list, map_dict, remap_dict, testing_data_num, testing_feature_dim, testing_label_dim = \
        utility.load_csv_data(testing_data_file, csv_header=testing_csv_header, map_file_path=mapping_file, feature_attribute_file_path=feature_attribute_file)
    random_forest.test(testing_feature, testing_label, feature_attribute, testing_feature_dim, testing_label_dim)

    if display_time:
        end_time = time.time()
        print('Used time : %.6fs.' % (end_time - start_time))
