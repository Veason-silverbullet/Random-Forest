import decision_tree as dt
import utility

tree1 = dt.ID3Tree()
tree1.root = dt.ContinuousNode(0, 0, 1.0, 0)
tree1.root.left_child = dt.ContinuousNode(1, 2, 3.0, 0)
tree1.root.right_child = dt.ContinuousNode(2, 4, 5.0, 0)
tree1.root.left_child.left_child = dt.ContinuousNode(3, 6, 7.0, 0)
tree1.root.left_child.right_child = dt.LeafNode(4, 0)
tree1.root.right_child.left_child = dt.LeafNode(5, 1)
tree1.root.right_child.right_child = dt.LeafNode(6, 2)
tree1.root.left_child.left_child.left_child = dt.LeafNode(7, 3)
tree1.root.left_child.left_child.right_child = dt.LeafNode(8, 4)
tree1.save('../data/test1.json')

tree2 = dt.ID3Tree()
tree2.load('../data/test1.json')
print(tree2.root)
tree2.save('../data/test2.json')

feature, label, feature_attribute, feature_list, map_dict, remap_dict, data_num, feature_dim, label_dim = \
    utility.load_csv_data('../data/data_test.csv', csv_header=False, feature_attribute_file_path='../data/feature_attribute_test.txt')
ID3_tree = dt.ID3Tree()
feature_index = [0, 1, 2, 3]
ID3_tree.build_decision_tree(feature, label, feature_index, feature_attribute, feature_list)
ID3_tree.save('../data/test_ID3.json')
cnt, result = ID3_tree.test(feature, label)
print(cnt, result)
Cart_tree = dt.CartTree()
Cart_tree.build_decision_tree(feature, label, feature_index, feature_attribute, feature_list)
Cart_tree.save('../data/test_Cart.json')
cnt, result = Cart_tree.test(feature, label)
print(cnt, result)

training_data_file = '../data/data.csv'
testing_data_file = '../data/data.csv'
mapping_file = '../data/index.txt'
feature_attribute_file = '../data/feature_attribute.txt'
feature_index = [0, 1, 2, 3, 4, 5, 6, 7]
training_feature, training_label, feature_attribute, feature_list, map_dict, remap_dict, training_data_num, training_feature_dim, training_label_dim = \
        utility.load_csv_data(training_data_file, csv_header=True, map_file_path=mapping_file, feature_attribute_file_path=feature_attribute_file)
testing_feature, testing_label, feature_attribute, feature_list, map_dict, remap_dict, testing_data_num, testing_feature_dim, testing_label_dim = \
    utility.load_csv_data(testing_data_file, csv_header=True, map_file_path=mapping_file, feature_attribute_file_path=feature_attribute_file)
tree3 = dt.ID3Tree()
tree3.build_decision_tree(training_feature, training_label, feature_index, feature_attribute, feature_list)
tree3.save('../data/test3.json')
cnt, result = tree3.test(testing_feature, testing_label)
print(cnt, result)
tree4 = dt.CartTree()
tree4.build_decision_tree(training_feature, training_label, feature_index, feature_attribute, feature_list)
tree4.save('../data/test4.json')
cnt, result = tree4.test(testing_feature, testing_label)
print(cnt, result)
