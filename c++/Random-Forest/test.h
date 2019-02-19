#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <map>
#include "utility.h"
#include "decisionTreeNodes.h"
#include "decisionTree.h"
#include "randomForest.h"
#include "rapidjson/include/rapidjson/document.h"
#include "rapidjson/include/rapidjson/writer.h"
#include "rapidjson/include/rapidjson/stringbuffer.h"
#include "rapidjson/include/rapidjson/filereadstream.h"
#include "rapidjson/include/rapidjson/reader.h"
#include "rapidjson/include/rapidjson/filewritestream.h"
#include "rapidjson/include/rapidjson/writer.h"
using namespace std;
using namespace randomForest::utility;
using namespace randomForest::decisionTreeNodes;
using namespace randomForest::decisionTree;
using namespace randomForest::randomForest;
using namespace rapidjson;

template <class Type>
void test()
{
    ID3Tree<Type> tree1 = ID3Tree<Type>();
    tree1.root = new ContinuousNode<Type>(0, 0, 1.0, 0);
    ContinuousNode<Type>* node0 = (ContinuousNode<Type>*)tree1.root;
    node0->leftChild = new ContinuousNode<Type>(1, 2, 3.0, 0);
    node0->rightChild = new ContinuousNode<Type>(2, 4, 5.0, 0);
    ContinuousNode<Type>* node1 = (ContinuousNode<Type>*)node0->leftChild;
    ContinuousNode<Type>* node2 = (ContinuousNode<Type>*)node0->rightChild;
    node1->leftChild = new ContinuousNode<Type>(3, 6, 7.0, 0);
    node1->rightChild = new LeafNode<Type>(4, 0);
    ContinuousNode<Type>* node3 = (ContinuousNode<Type>*)node1->leftChild;
    LeafNode<Type>* node4 = (LeafNode<Type>*)node1->rightChild;
    node2->leftChild = new LeafNode<Type>(5, 1);
    node2->rightChild = new LeafNode<Type>(6, 2);
    LeafNode<Type>* node5 = (LeafNode<Type>*)node2->leftChild;
    LeafNode<Type>* node6 = (LeafNode<Type>*)node2->rightChild;
    node3->leftChild = new LeafNode<Type>(7, 3);
    node3->rightChild = new LeafNode<Type>(8, 4);
    LeafNode<Type>* node7 = (LeafNode<Type>*)node3->leftChild;
    LeafNode<Type>* node8 = (LeafNode<Type>*)node3->rightChild;
    tree1.save("../../data/test1.json");

    ID3Tree<Type> tree2 = ID3Tree<Type>();
    tree2.load("../../data/test1.json");
    tree2.save("../../data/test2.json");

    vector<unsigned int>featureIndex = { 0,1,2,3 };
    vector<unsigned int>result;
    unsigned int cnt = 0;
    Data<Type>* feature = new Data<Type>();
    Data<Type>* label = new Data<Type>();
    map<string, unsigned int> labelMap;
    Data<Type>::loadCsvData(*feature, *label, labelMap, "../../data/data_test.csv", false, "../../data/feature_attribute_test.txt");
    ID3Tree<Type> id3Tree = ID3Tree<Type>();
    id3Tree.buildDecisionTree(feature, label, 8, featureIndex);
    id3Tree.save("../../data/test_ID3.json");
    cnt = id3Tree.test(*feature, *label, result);
    cout << cnt << " [";
    for (size_t i(0); i < result.size(); ++i)
        cout << result[i] << (i != result.size() - 1 ? ", " : "]\n");

    CartTree<Type> cartTree = CartTree<Type>();
    cartTree.buildDecisionTree(feature, label, 8, featureIndex);
    cartTree.save("../../data/test_Cart.json");
    cnt = cartTree.test(*feature, *label, result);
    cout << cnt << " [";
    for (size_t i(0); i < result.size(); ++i)
        cout << result[i] << (i != result.size() - 1 ? ", " : "]\n");

    Data<Type>* trainingFeature = new Data<Type>();
    Data<Type>* trainingLabel = new Data<Type>();
    Data<Type>* testingFeature = new Data<Type>();
    Data<Type>* testingLabel = new Data<Type>();
    Data<Type>::loadCsvData(*trainingFeature, *trainingLabel, labelMap, "../../data/data.csv", true, "../../data/feature_attribute.txt", "../../data/index.txt");
    Data<Type>::loadCsvData(*testingFeature, *testingLabel, labelMap, "../../data/data.csv", true, "../../data/feature_attribute.txt", "../../data/index.txt");
    featureIndex = { 0,1,2,3,4,5,6,7 };
    ID3Tree<Type> tree3 = ID3Tree<Type>();
    tree3.buildDecisionTree(trainingFeature, trainingLabel, 2, featureIndex);
    tree3.save("../../data/test3.json");
    cnt = tree3.test(*testingFeature, *testingLabel, result);
    cout << cnt << endl;
    CartTree<Type> tree4 = CartTree<Type>();
    tree4.buildDecisionTree(trainingFeature, trainingLabel, 2, featureIndex);
    cnt = tree4.test(*testingFeature, *testingLabel, result);
    cout << cnt << endl;
    tree4.save("../../data/test4.json");

    RandomForest<Type> forest1 = RandomForest<Type>();
    forest1.setRandomForestType(RandomForestType::Mixture);
    forest1.setTreeNum(2);
    forest1.setRandomDim(3);
    forest1.buildRandomForest(feature, label, 8);
    forest1.save("../../data/test5.json");

    RandomForest<Type> forest2 = RandomForest<Type>();
    forest2.load("../../data/test5.json");
    forest2.save("../../data/test6.json");
}