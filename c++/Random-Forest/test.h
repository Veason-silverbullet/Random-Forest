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


void test()
{
    ID3Tree<float> tree1 = ID3Tree<float>();
    tree1.root = new ContinuousNode<float>(0, 0, 1.0, 0);
    ContinuousNode<float>* node0 = (ContinuousNode<float>*)tree1.root;
    node0->leftChild = new ContinuousNode<float>(1, 2, 3.0, 0);
    node0->rightChild = new ContinuousNode<float>(2, 4, 5.0, 0);
    ContinuousNode<float>* node1 = (ContinuousNode<float>*)node0->leftChild;
    ContinuousNode<float>* node2 = (ContinuousNode<float>*)node0->rightChild;
    node1->leftChild = new ContinuousNode<float>(3, 6, 7.0, 0);
    node1->rightChild = new LeafNode<float>(4, 0);
    ContinuousNode<float>* node3 = (ContinuousNode<float>*)node1->leftChild;
    LeafNode<float>* node4 = (LeafNode<float>*)node1->rightChild;
    node2->leftChild = new LeafNode<float>(5, 1);
    node2->rightChild = new LeafNode<float>(6, 2);
    LeafNode<float>* node5 = (LeafNode<float>*)node2->leftChild;
    LeafNode<float>* node6 = (LeafNode<float>*)node2->rightChild;
    node3->leftChild = new LeafNode<float>(7, 3);
    node3->rightChild = new LeafNode<float>(8, 4);
    LeafNode<float>* node7 = (LeafNode<float>*)node3->leftChild;
    LeafNode<float>* node8 = (LeafNode<float>*)node3->rightChild;
    tree1.save("../../data/test1.json");

    ID3Tree<float> tree2 = ID3Tree<float>();
    tree2.load("../../data/test1.json");
    tree2.save("../../data/test2.json");

    vector<unsigned int>featureIndex = { 0,1,2,3 };
    vector<unsigned int>result;
    unsigned int cnt = 0;
    Data<float>* feature = new Data<float>();
    Data<float>* label = new Data<float>();
    map<string, unsigned int> labelMap;
    Data<float>::loadCsvData(*feature, *label, labelMap, "../../data/data_test.csv", false, "../../data/feature_attribute_test.txt");
    ID3Tree<float> id3Tree = ID3Tree<float>();
    id3Tree.buildDecisionTree(feature, label, 8, featureIndex);
    id3Tree.save("../../data/test_ID3.json");
    cnt = id3Tree.test(*feature, *label, result);
    cout << cnt << " [";
    for (size_t i(0); i < result.size(); ++i)
        cout << result[i] << (i != result.size() - 1 ? ", " : "]\n");

    CartTree<float> cartTree = CartTree<float>();
    cartTree.buildDecisionTree(feature, label, 8, featureIndex);
    cartTree.save("../../data/test_Cart.json");
    cnt = cartTree.test(*feature, *label, result);
    cout << cnt << " [";
    for (size_t i(0); i < result.size(); ++i)
        cout << result[i] << (i != result.size() - 1 ? ", " : "]\n");

    Data<float>* trainingFeature = new Data<float>();
    Data<float>* trainingLabel = new Data<float>();
    Data<float>* testingFeature = new Data<float>();
    Data<float>* testingLabel = new Data<float>();
    Data<float>::loadCsvData(*trainingFeature, *trainingLabel, labelMap, "../../data/data.csv", true, "../../data/feature_attribute.txt", "../../data/index.txt");
    Data<float>::loadCsvData(*testingFeature, *testingLabel, labelMap, "../../data/data.csv", true, "../../data/feature_attribute.txt", "../../data/index.txt");
    featureIndex = { 0,1,2,3,4,5,6,7 };
    ID3Tree<float> tree3 = ID3Tree<float>();
    tree3.buildDecisionTree(trainingFeature, trainingLabel, 2, featureIndex);
    tree3.save("../../data/test3.json");
    cnt = tree3.test(*testingFeature, *testingLabel, result);
    cout << cnt << endl;
    CartTree<float> tree4 = CartTree<float>();
    tree4.buildDecisionTree(trainingFeature, trainingLabel, 2, featureIndex);
    cnt = tree4.test(*testingFeature, *testingLabel, result);
    cout << cnt << endl;
    tree4.save("../../data/test4.json");
}