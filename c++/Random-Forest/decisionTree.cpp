#define _CRT_SECURE_NO_WARNINGS
#include "decisionTreeNodes.h"
#include "decisionTree.h"
#include "utility.h"
#include "assert.h"
#include <iostream>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <queue>
#include <map>
#include <algorithm>
#include "rapidjson/include/rapidjson/document.h"
#include "rapidjson/include/rapidjson/writer.h"
#include "rapidjson/include/rapidjson/stringbuffer.h"
#include "rapidjson/include/rapidjson/filereadstream.h"
#include "rapidjson/include/rapidjson/reader.h"
#include "rapidjson/include/rapidjson/filewritestream.h"
#include "rapidjson/include/rapidjson/writer.h"
#define JsonBufferSize (1 << 16)
using namespace std;
using namespace randomForest::decisionTreeNodes;
using namespace randomForest::utility;
using namespace rapidjson;

namespace randomForest { namespace decisionTree {

    template <class Type>
    Tree<Type>::Tree()
    {
        this->nodeId = 0;
    }

    template <class Type>
    void Tree<Type>::serialize(map<NodeId, NodeInfo<Type>*> & nodeMap)
    {
        queue<Node<Type>*> nodeQueue;
        assert(this->root != NULL);
        nodeQueue.push(this->root);
        while (!nodeQueue.empty())
        {
            Node<Type>* node = nodeQueue.front();
            nodeQueue.pop();
            if (node->nodeType == NodeType::Continuous)
            {
                nodeQueue.push(((ContinuousNode<Type>*)node)->leftChild);
                nodeQueue.push(((ContinuousNode<Type>*)node)->rightChild);
                ContinuousNodeInfo<Type>* nodeInfo = new ContinuousNodeInfo<Type>();
                node->serialize(nodeInfo);
                nodeMap.insert({ node->nodeId, nodeInfo });
            }
            else if (node->nodeType == NodeType::Discrete)
            {
                DiscreteNode<Type>* nodePtr = (DiscreteNode<Type>*)node;
                for (unsigned int i(0); i < nodePtr->childList.size(); ++i)
                    nodeQueue.push(nodePtr->childList[i]);
                DiscreteNodeInfo<Type>* nodeInfo = new DiscreteNodeInfo<Type>();
                node->serialize(nodeInfo);
                nodeMap.insert({ node->nodeId, nodeInfo });
            }
            else
            {
                LeafNodeInfo<Type>* nodeInfo = new LeafNodeInfo<Type>();
                node->serialize(nodeInfo);
                nodeMap.insert({ node->nodeId, nodeInfo });
            }
        }
    }

    template <class Type>
    void Tree<Type>::deserialize(const map<NodeId, NodeInfo<Type>*> & nodeMap)
    {
        queue<Node<Type>*> nodeQueue;
        auto it = nodeMap.find((NodeId)0);
        auto* nodeInfo = it->second;
        if (it != nodeMap.end())
        {
            if (nodeInfo->nodeType == NodeType::Continuous)
                this->root = new ContinuousNode<Type>();
            else if (nodeInfo->nodeType == NodeType::Discrete)
                this->root = new DiscreteNode<Type>();
            else
                this->root = new LeafNode<Type>();
            this->root->deserialize(nodeInfo);
            nodeQueue.push(this->root);
        }
        else
            throw exception("Deserialize error: root node was not found");

        while (!nodeQueue.empty())
        {
            Node<Type>* node = nodeQueue.front();
            nodeQueue.pop();
            if (node->nodeType == NodeType::Continuous)
            {
                it = nodeMap.find(((ContinuousNode<Type>*)(node))->nodeId);
                assert(it != nodeMap.end());
                auto* nodeInfo = it->second;
                auto _it = nodeMap.find(((ContinuousNodeInfo<Type>*)(nodeInfo))->leftChildId);
                assert(_it != nodeMap.end());
                auto _nodeInfo = _it->second;
                if (_nodeInfo->nodeType == NodeType::Continuous)
                    ((ContinuousNode<Type>*)(node))->leftChild = new ContinuousNode<Type>();
                else if (_nodeInfo->nodeType == NodeType::Discrete)
                    ((ContinuousNode<Type>*)(node))->leftChild = new DiscreteNode<Type>();
                else
                    ((ContinuousNode<Type>*)(node))->leftChild = new LeafNode<Type>();
                ((ContinuousNode<Type>*)(node))->leftChild->deserialize(_nodeInfo);
                nodeQueue.push(((ContinuousNode<Type>*)(node))->leftChild);
                _it = nodeMap.find(((ContinuousNodeInfo<Type>*)(nodeInfo))->rightChildId);
                assert(_it != nodeMap.end());
                _nodeInfo = _it->second;
                if (_nodeInfo->nodeType == NodeType::Continuous)
                    ((ContinuousNode<Type>*)(node))->rightChild = new ContinuousNode<Type>();
                else if (_nodeInfo->nodeType == NodeType::Discrete)
                    ((ContinuousNode<Type>*)(node))->rightChild = new DiscreteNode<Type>();
                else
                    ((ContinuousNode<Type>*)(node))->rightChild = new LeafNode<Type>();
                ((ContinuousNode<Type>*)(node))->rightChild->deserialize(_nodeInfo);
                nodeQueue.push(((ContinuousNode<Type>*)(node))->rightChild);
            }
            else if (node->nodeType == NodeType::Discrete)
            {
                it = nodeMap.find(((DiscreteNode<Type>*)(node))->nodeId);
                assert(it != nodeMap.end());
                nodeInfo = it->second;
                for (unsigned int i(0); i < ((DiscreteNodeInfo<Type>*)(nodeInfo))->childIdList.size(); ++i)
                {
                    auto _it = nodeMap.find(((DiscreteNodeInfo<Type>*)(nodeInfo))->childIdList[i]);
                    assert(_it != nodeMap.end());
                    auto _nodeInfo = _it->second;
                    if (_nodeInfo->nodeType == NodeType::Continuous)
                        ((DiscreteNode<Type>*)(node))->childList.push_back(new ContinuousNode<Type>());
                    else if (_nodeInfo->nodeType == NodeType::Discrete)
                        ((DiscreteNode<Type>*)(node))->childList.push_back(new DiscreteNode<Type>());
                    else
                        ((DiscreteNode<Type>*)(node))->childList.push_back(new LeafNode<Type>());
                    ((DiscreteNode<Type>*)(node))->childList[i]->deserialize(_nodeInfo);
                    nodeQueue.push(((DiscreteNode<Type>*)(node))->childList[i]);
                }
            }
        }
    }

    template <class Type>
    void Tree<Type>::load(string fileName)
    {
        FILE* file = fopen(fileName.c_str(), "rb");
        assert(file != NULL);
        char readBuffer[JsonBufferSize];
        FileReadStream jsonReader(file, readBuffer, sizeof(readBuffer));
        Document doc;
        doc.ParseStream(jsonReader);
        map<NodeId, NodeInfo<Type>*> nodeMap;
        for (SizeType i(0); i < doc.Size(); ++i)
        {
            assert(doc[i].HasMember("node_type") && doc[i]["node_type"].IsString());
            assert(doc[i].HasMember("node_id") && doc[i]["node_id"].IsInt());
            string nodeType = doc[i]["node_type"].GetString();
            NodeId nodeId = doc[i]["node_id"].GetInt();
            if (nodeType == "Continuous")
            {
                assert(doc[i].HasMember("feature_index") && doc[i]["feature_index"].IsInt());
                assert(doc[i].HasMember("feature_value") && doc[i]["feature_value"].IsNumber());
                assert(doc[i].HasMember("max_label") && doc[i]["max_label"].IsInt());
                assert(doc[i].HasMember("left_child_id") && doc[i]["left_child_id"].IsInt());
                assert(doc[i].HasMember("right_child_id") && doc[i]["right_child_id"].IsInt());
                ContinuousNodeInfo<Type>* nodeInfo = new ContinuousNodeInfo<Type>(nodeId, NodeType::Continuous,
                    doc[i]["feature_index"].GetInt(), (Type)(doc[i]["feature_value"].GetDouble()), doc[i]["max_label"].GetInt(), doc[i]["left_child_id"].GetInt(), doc[i]["right_child_id"].GetInt());
                nodeMap.insert({ nodeId,nodeInfo });
            }
            else if (nodeType == "Discrete")
            {
                assert(doc[i].HasMember("feature_index") && doc[i]["feature_index"].IsInt());
                assert(doc[i].HasMember("feature_list") && doc[i]["feature_list"].IsArray());
                vector<string> featureList;
                for (SizeType j(0); j < doc[i]["feature_list"].Size(); ++j)
                    featureList.push_back(string(doc[i]["feature_list"][j].GetString()));
                assert(doc[i].HasMember("max_label") && doc[i]["max_label"].IsInt());
                assert(doc[i].HasMember("child_id_list") && doc[i]["child_id_list"].IsArray());
                vector<NodeId> childIdList;
                for (SizeType j(0); j < doc[i]["child_id_list"].Size(); ++j)
                    childIdList.push_back(doc[i]["child_id_list"][j].GetInt());
                DiscreteNodeInfo<Type>* nodeInfo = new DiscreteNodeInfo<Type>(nodeId, NodeType::Discrete,
                    doc[i]["feature_index"].GetInt(), featureList, doc[i]["max_label"].GetInt(), childIdList);
                nodeMap.insert({ nodeId,nodeInfo });
            }
            else if (nodeType == "Leaf")
            {
                assert(doc[i].HasMember("label") && doc[i]["label"].IsInt());
                LeafNodeInfo<Type>* nodeInfo = new LeafNodeInfo<Type>(nodeId, NodeType::Leaf,
                    doc[i]["label"].GetInt());
                nodeMap.insert({ nodeId,nodeInfo });
            }
            else
                throw exception(("Node type error: unexpected node type [" + nodeType + "] is not [Continuous | Discrete | Leaf].").c_str());
        }
        fclose(file);

        this->deserialize(nodeMap);
        for (auto it = nodeMap.begin(); it != nodeMap.end(); ++it)
            delete it->second;
    }

    template <class Type>
    void Tree<Type>::save(string fileName)
    {
        map<NodeId, NodeInfo<Type>*> nodeMap;
        this->serialize(nodeMap);

        FILE* file = fopen(fileName.c_str(), "wb");
        assert(file != NULL);
        char writeBuffer[JsonBufferSize];
        FileWriteStream jsonWriter(file, writeBuffer, sizeof(writeBuffer));
        Document doc;
        Document::AllocatorType& docAllocator = doc.GetAllocator();
        doc.SetArray();
        for (auto it = nodeMap.begin(); it != nodeMap.end(); ++it)
        {
            Value val(kObjectType);
            NodeInfo<Type>* nodeInfo = it->second;
            val.AddMember("node_id", nodeInfo->nodeId, docAllocator);
            Value s;
            s.SetString(StringRef(nodeTypeStr[nodeInfo->nodeType].c_str(), nodeTypeStr[nodeInfo->nodeType].length()));
            val.AddMember("node_type", s, docAllocator);
            if (nodeInfo->nodeType == NodeType::Continuous)
            {
                ContinuousNodeInfo<Type>* continuousNodeInfo = (ContinuousNodeInfo<Type>*)nodeInfo;
                val.AddMember("feature_index", continuousNodeInfo->featureIndex, docAllocator);
                val.AddMember("feature_value", continuousNodeInfo->featureValue, docAllocator);
                val.AddMember("max_label", continuousNodeInfo->maxLabel, docAllocator);
                val.AddMember("left_child_id", continuousNodeInfo->leftChildId, docAllocator);
                val.AddMember("right_child_id", continuousNodeInfo->rightChildId, docAllocator);
            }
            else if (nodeInfo->nodeType == NodeType::Discrete)
            {
                DiscreteNodeInfo<Type>* discreteNodeInfo = (DiscreteNodeInfo<Type>*)nodeInfo;
                val.AddMember("feature_index", discreteNodeInfo->featureIndex, docAllocator);
                Value featureArr(kArrayType);
                for (unsigned int i(0); i < discreteNodeInfo->featureList.size(); ++i)
                {
                    Value str;
                    str.SetString(StringRef(discreteNodeInfo->featureList[i].c_str(), discreteNodeInfo->featureList[i].length()));
                    featureArr.PushBack(str, docAllocator);
                }
                val.AddMember("feature_list", featureArr, docAllocator);
                val.AddMember("max_label", discreteNodeInfo->maxLabel, docAllocator);
                Value childIdArr(kArrayType);
                for (unsigned int i(0); i < discreteNodeInfo->childIdList.size(); ++i)
                    childIdArr.PushBack(discreteNodeInfo->childIdList[i], docAllocator);
                val.AddMember("child_id_list", childIdArr, docAllocator);
            }
            else if (nodeInfo->nodeType == NodeType::Leaf)
            {
                LeafNodeInfo<Type>* leafNodeInfo = (LeafNodeInfo<Type>*)nodeInfo;
                val.AddMember("label", leafNodeInfo->label, docAllocator);
            }
            else
                throw exception(("Node type error: unexpected node type [" + nodeTypeStr[nodeInfo->nodeType] + "] is not [Continuous | Discrete | Leaf].").c_str());
            doc.PushBack(val, docAllocator);
        }
        Writer<FileWriteStream> writer(jsonWriter);
        doc.Accept(writer);
        fclose(file);
        for (auto it = nodeMap.begin(); it != nodeMap.end(); ++it)
            delete it->second;
    }

    template <class Type>
    void Tree<Type>::release()
    {
        if (this->root == NULL)
            return;
        queue<Node<Type>*> nodeQueue;
        nodeQueue.push(this->root);
        while (!nodeQueue.empty())
        {
            Node<Type>* node = nodeQueue.front();
            nodeQueue.pop();
            if (node->nodeType == NodeType::Continuous)
            {
                nodeQueue.push(((ContinuousNode<Type>*)(node))->leftChild);
                nodeQueue.push(((ContinuousNode<Type>*)(node))->rightChild);
            }
            else if (node->nodeType == NodeType::Discrete)
            {
                DiscreteNode<Type>* nodePtr = (DiscreteNode<Type>*)node;
                for (unsigned int i(0); i < nodePtr->childList.size(); ++i)
                    nodeQueue.push(nodePtr->childList[i]);
            }
            delete node;
        }
        this->root = NULL;
    }

    template <class Type>
    Tree<Type>::~Tree()
    {
        this->release();
    }

    template <class Type>
    DecisionTreeType DecisionTree<Type>::getDecisionTreeType()
    {
        return this->decisionTreeType;
    }

    template <class Type>
    LeafNode<Type>* DecisionTree<Type>::generateLeafNode(vector<unsigned int> & dataIndex, vector<bool> & usedFeature)
    {
        unsigned int dataNum = (unsigned int)dataIndex.size();
        unsigned int dataDim = feature->dim;
        unsigned int* labelPtr = (unsigned int*)(label->dataPtr[0]);
        bool flag = true;
        for (unsigned int i(1); i < dataNum; ++i)
        {
            if (labelPtr[dataIndex[i]] != labelPtr[dataIndex[0]])
            {
                flag = false;
                break;
            }
        }

        if (flag)
            return new LeafNode<Type>(this->nodeId++, labelPtr[dataIndex[0]]);
        else
        {
            flag = true;
            for (unsigned int i(0); i < dataDim; ++i)
            {
                if (!usedFeature[i])
                {
                    if (feature->dataType[i] == DataType::Real)
                    {
                        Type* featurePtr = (Type*)(feature->dataPtr[i]);
                        for (unsigned int j(1); j < dataNum; ++j)
                        {
                            if (featurePtr[dataIndex[j]] != featurePtr[dataIndex[0]])
                            {
                                flag = false;
                                break;
                            }
                        }
                    }
                    else if (feature->dataType[i] == DataType::String)
                    {
                        string* featurePtr = (string*)(feature->dataPtr[i]);
                        for (unsigned int j(1); j < dataNum; ++j)
                        {
                            if (featurePtr[dataIndex[j]] != featurePtr[dataIndex[0]])
                            {
                                flag = false;
                                break;
                            }
                        }
                    }
                    else
                        throw exception("Data type error: unexpected type, data type must be in [DataType::Real | DataType::String]");
                }
                if (!flag)
                    break;
            }
            if (flag)
            {
                unsigned int* cnt = new unsigned int[labelDim];
                memset(cnt, 0, sizeof(unsigned int) * labelDim);
                for (unsigned int i(0); i < dataNum; ++i)
                    ++cnt[labelPtr[dataIndex[i]]];
                unsigned int maxCnt = cnt[0];
                unsigned int maxLabel = 0;
                for (unsigned int i(1); i < labelDim; ++i)
                {
                    if (cnt[i] > maxCnt)
                    {
                        maxCnt = cnt[i];
                        maxLabel = i;
                    }
                }
                delete[] cnt;
                return new LeafNode<Type>(this->nodeId++, maxLabel);
            }
        }

        return NULL;
    }

    template <class Type>
    Node<Type>* DecisionTree<Type>::createNode(vector<unsigned int> & dataIndex, vector<vector<string>> & featureList, vector<map<string, unsigned int>> & featureMap, vector<bool> & usedFeature)
    {
        LeafNode<Type>* leafNode = generateLeafNode(dataIndex, usedFeature);
        if (leafNode != NULL)
            return leafNode;

        unsigned int dataNum = (unsigned int)dataIndex.size();
        unsigned int dataDim = feature->dim;
        unsigned int* labelPtr = (unsigned int*)label->dataPtr[0];
        vector<bool> flag;
        flag.resize(dataDim);
        fill(flag.begin(), flag.end(), false);
        vector<vector<Type>> thresold;
        thresold.resize(dataDim);
        for (unsigned int i(0); i < dataDim; ++i)
        {
            if (!usedFeature[i])
            {
                if (feature->dataType[i] == DataType::Real)
                {
                    Type* dataPtr = (Type*)feature->dataPtr[i];
                    for (unsigned int j(0); j < dataNum; ++j)
                    {
                        if (dataPtr[dataIndex[j]] != dataPtr[dataIndex[0]])
                            flag[i] = true;
                        thresold[i].push_back(dataPtr[dataIndex[j]]);
                    }
                    sort(thresold[i].begin(), thresold[i].end());
                    auto iter = unique(thresold[i].begin(), thresold[i].end());
                    thresold[i].erase(iter, thresold[i].end());
                    unsigned int thresoldLen = (unsigned int)thresold[i].size() - 1;
                    for (unsigned int j(0); j < thresoldLen; ++j)
                        thresold[i][j] = (thresold[i][j] + thresold[i][j + 1]) / 2;
                }
                else
                {
                    string* dataPtr = (string*)feature->dataPtr[i];
                    unsigned int featureSize = (unsigned int)featureList[i].size();
                    for (unsigned int j(0); j < featureSize; ++j)
                        memset(multiCnt[i][j], 0, sizeof(unsigned int) * labelDim);
                    for (unsigned int j(0); j < dataNum; ++j)
                    {
                        if (dataPtr[dataIndex[j]] != dataPtr[dataIndex[0]])
                            flag[i] = true;
                        ++multiCnt[i][featureMap[i][dataPtr[dataIndex[j]]]][labelPtr[dataIndex[j]]];
                    }
                }
            }
        }

        double bestGain = -1.0;
        int bestIndex = -1;
        Type bestValue = 0;
        FeatureLabelBundle<Type>* tempData = new FeatureLabelBundle<Type>[dataNum];
        for (unsigned int i(0); i < dataDim; ++i)
        {
            if (!usedFeature[i] && flag[i])
            {
                if (feature->dataType[i] == DataType::Real)
                {
                    Type* dataPtr = (Type*)feature->dataPtr[i];
                    memset(binCnt[0], 0, sizeof(unsigned int) * labelDim);
                    memset(binCnt[1], 0, sizeof(unsigned int) * labelDim);
                    for (unsigned int j(0); j < dataNum; ++j)
                    {
                        tempData[j].feature = dataPtr[dataIndex[j]];
                        tempData[j].label = labelPtr[dataIndex[j]];
                    }
                    sort(tempData, tempData + dataNum);
                    unsigned int thresoldIndex = 0;
                    for (unsigned int j(0); j < dataNum; ++j)
                    {
                        if (tempData[j].feature <= thresold[i][0])
                            ++binCnt[0][tempData[j].label];
                        else
                        {
                            for (unsigned int k(j); k < dataNum; ++k)
                                ++binCnt[1][tempData[k].label];
                            thresoldIndex = j;
                            break;
                        }
                    }
                    double gain = continuousGainPtr(binCnt, labelDim, dataNum);
                    if (bestIndex == -1 || gain > bestGain)
                    {
                        bestGain = gain;
                        bestIndex = i;
                        bestValue = thresold[i][0];
                    }
                    unsigned int thresoldLen = (unsigned int)thresold[i].size() - 1;
                    for (unsigned int j(1); j < thresoldLen; ++j)
                    {
                        while (tempData[thresoldIndex].feature <= thresold[i][j])
                        {
                            ++binCnt[0][tempData[thresoldIndex].label];
                            --binCnt[1][tempData[thresoldIndex].label];
                            ++thresoldIndex;
                        }
                        gain = continuousGainPtr(binCnt, labelDim, dataNum);
                        if (bestIndex == -1 || gain > bestGain)
                        {
                            bestGain = gain;
                            bestIndex = i;
                            bestValue = thresold[i][j];
                        }
                    }
                }
                else
                {
                    double gain = discreteGainPtr(multiCnt[i], (unsigned int)featureList[i].size(), labelDim, dataNum);
                    if (bestIndex == -1 || gain > bestGain)
                    {
                        bestGain = gain;
                        bestIndex = i;
                    }
                }
            }
        }
        delete[] tempData;

        if (bestIndex != -1)
        {
            unsigned int* cnt = new unsigned int[labelDim];
            memset(cnt, 0, sizeof(unsigned int) * labelDim);
            for (unsigned int i(0); i < dataNum; ++i)
                ++cnt[labelPtr[dataIndex[i]]];
            unsigned int maxCnt = cnt[0];
            unsigned int maxLabel = 0;
            for (unsigned int i(1); i < labelDim; ++i)
            {
                if (cnt[i] > maxCnt)
                {
                    maxCnt = cnt[i];
                    maxLabel = i;
                }
            }
            delete[] cnt;

            if (feature->dataType[bestIndex] == DataType::Real)
            {
                ContinuousNode<Type>* node = new ContinuousNode<Type>(this->nodeId++, featureIndex[bestIndex], bestValue, maxLabel);
                node->usedFeature.assign(usedFeature.begin(), usedFeature.end());
                node->usedFeature[bestIndex] = true;
                Type* dataPtr = (Type*)feature->dataPtr[bestIndex];
                for (unsigned int i(0); i < dataNum; ++i)
                {
                    if (dataPtr[dataIndex[i]] <= bestValue)
                        node->tempIndex0.push_back(dataIndex[i]);
                    else
                        node->tempIndex1.push_back(dataIndex[i]);
                }
                if (node->tempIndex0.size() == 0)
                    node->leftChild = new LeafNode<Type>(this->nodeId++, maxLabel);
                if (node->tempIndex1.size() == 0)
                    node->rightChild = new LeafNode<Type>(this->nodeId++, maxLabel);
                return node;
            }
            else
            {
                DiscreteNode<Type>* node = new DiscreteNode<Type>(this->nodeId++, featureIndex[bestIndex], featureList[bestIndex], maxLabel);
                node->usedFeature.assign(usedFeature.begin(), usedFeature.end());
                node->usedFeature[bestIndex] = true;
                string* dataPtr = (string*)feature->dataPtr[bestIndex];
                unsigned int featureNum = (unsigned int)featureList[bestIndex].size();
                node->tempIndex.resize(featureNum);
                map<string, unsigned int> & _featureMap = featureMap[bestIndex];
                for (unsigned int i(0); i < dataNum; ++i)
                    node->tempIndex[_featureMap[dataPtr[dataIndex[i]]]].push_back(dataIndex[i]);
                node->childList.resize(featureNum);
                for (unsigned int i(0); i < featureNum; ++i)
                {
                    if (node->tempIndex[i].size() == 0)
                        node->childList[i] = new LeafNode<Type>(this->nodeId++, maxLabel);
                }
                return node;
            }
        }
        else
            throw exception("Logic error: best gain is None.");
    }

    template <class Type>
    void DecisionTree<Type>::buildDecisionTree(Data<Type>* feature, Data<Type>* label, unsigned int labelDim, const vector<unsigned int> & featureIndex)
    {
        this->feature = feature;
        this->label = label;
        this->labelDim = labelDim;
        this->featureIndex.assign(featureIndex.begin(), featureIndex.end());
        unsigned int dataNum = feature->num;
        unsigned int dataDim = feature->dim;

        binCnt = new unsigned int*[2];
        binCnt[0] = new unsigned int[labelDim];
        binCnt[1] = new unsigned int[labelDim];
        multiCnt.resize(dataDim);
        vector<vector<string>> featureList;
        vector<map<string, unsigned int>> featureMap;
        featureList.resize(dataDim);
        featureMap.resize(dataDim);
        for (unsigned int i(0); i < dataDim; ++i)
        {
            if (feature->dataType[i] == DataType::String)
            {
                string* dataPtr = (string*)feature->dataPtr[i];
                for (unsigned int j(0); j < dataNum; ++j)
                {
                    if (featureMap[i].find(dataPtr[j]) == featureMap[i].end())
                    {
                        featureMap[i][dataPtr[j]] = (unsigned int)featureList[i].size();
                        featureList[i].push_back(dataPtr[j]);
                    }
                }
                unsigned int featureSize = (unsigned int)featureList[i].size();
                multiCnt[i] = new unsigned int*[featureSize];
                for (unsigned int j(0); j < featureSize; ++j)
                    multiCnt[i][j] = new unsigned int[labelDim];
            }
        }
        vector<unsigned int> initDataIndex;
        initDataIndex.resize(dataNum);
        for (unsigned int i(0); i < dataNum; ++i)
            initDataIndex[i] = i;
        vector<bool> initUsedFeature(feature->dim);
        fill(initUsedFeature.begin(), initUsedFeature.end(), false);
        this->root = createNode(initDataIndex, featureList, featureMap, initUsedFeature);
        queue<Node<Type>*> nodeQueue;
        nodeQueue.push(this->root);

        while (!nodeQueue.empty())
        {
            Node<Type>* node = nodeQueue.front();
            nodeQueue.pop();
            if (NodeType::Continuous == node->nodeType)
            {
                ContinuousNode<Type>* nodePtr = (ContinuousNode<Type>*)node;
                if (nodePtr->tempIndex0.size() != 0)
                {
                    Node<Type>* leftChildNode = createNode(nodePtr->tempIndex0, featureList, featureMap, nodePtr->usedFeature);
                    nodePtr->leftChild = leftChildNode;
                    if(leftChildNode->nodeType != NodeType::Leaf)
                        nodeQueue.push(leftChildNode);
                }
                if (nodePtr->tempIndex1.size() != 0)
                {
                    Node<Type>* rightChildNode = createNode(nodePtr->tempIndex1, featureList, featureMap, nodePtr->usedFeature);
                    nodePtr->rightChild = rightChildNode;
                    if (rightChildNode->nodeType != NodeType::Leaf)
                        nodeQueue.push(rightChildNode);
                }
            }
            else if (NodeType::Discrete == node->nodeType)
            {
                DiscreteNode<Type>* nodePtr = (DiscreteNode<Type>*)node;
                unsigned int childNum = (unsigned int)nodePtr->childList.size();
                for (unsigned int i(0); i < childNum; ++i)
                {
                    if (nodePtr->tempIndex[i].size() != 0)
                    {
                        Node<Type>* childNode = createNode(nodePtr->tempIndex[i], featureList, featureMap, nodePtr->usedFeature);
                        nodePtr->childList[i] = childNode;
                        if (childNode->nodeType != NodeType::Leaf)
                            nodeQueue.push(childNode);
                    }
                }
            }
        }

        delete binCnt[0];
        delete binCnt[1];
        delete binCnt;
        for (unsigned int i(0); i < dataDim; ++i)
        {
            if (feature->dataType[i] == DataType::String)
            {
                unsigned int featureSize = (unsigned int)featureList[i].size();
                for (unsigned int j(0); j < featureSize; ++j)
                    delete[] multiCnt[i][j];
                delete[] multiCnt[i];
                multiCnt[i] = NULL;
            }
        }
    }

    template <class Type>
    unsigned int DecisionTree<Type>::test(Data<Type> & testFeature, Data<Type> & testLabel, vector<unsigned int> & result)
    {
        assert(testFeature.num == testLabel.num && this->root != NULL);
        unsigned int num = testFeature.num;
        result.resize(num);

        for (unsigned int i(0); i < num; ++i)
        {
            Node<Type>* node = this->root;
            while (true)
            {
                if (node->nodeType == NodeType::Continuous)
                {
                    ContinuousNode<Type>* nodePtr = (ContinuousNode<Type>*)node;
                    if (((Type*)(testFeature.dataPtr[nodePtr->featureIndex]))[i] <= nodePtr->featureValue)
                        node = nodePtr->leftChild;
                    else
                        node = nodePtr->rightChild;
                }
                else if (node->nodeType == NodeType::Discrete)
                {
                    DiscreteNode<Type>* nodePtr = (DiscreteNode<Type>*)node;
                    string* dataPtr = (string*)(testFeature.dataPtr[nodePtr->featureIndex]);
                    bool flag = true;
                    vector<string> & featureList = nodePtr->featureList;
                    for (size_t j(0); j < featureList.size(); ++j)
                    {
                        if (dataPtr[i] == featureList[j])
                        {
                            node = nodePtr->childList[j];
                            flag = false;
                            break;
                        }
                    }
                    if (flag)
                    {
                        result[i] = nodePtr->maxLabel;
                        break;
                    }
                }
                else
                {
                    result[i] = ((LeafNode<Type>*)node)->label;
                    break;
                }
            }
        }

        unsigned int cnt = 0;
        unsigned int* labelPtr = (unsigned int*)(testLabel.dataPtr[0]);
        for (unsigned int i(0); i < num; ++i)
        {
            if (labelPtr[i] == result[i])
                ++cnt;
        }

        return cnt;
    }

    template <class Type>
    ID3Tree<Type>::ID3Tree()
    {
        this->decisionTreeType = DecisionTreeType::ID3;
        this->continuousGainPtr = &(this->continuousGain);
        this->discreteGainPtr = &(this->discreteGain);
    }

    template <class Type>
    double ID3Tree<Type>::continuousGain(unsigned int** cnt, unsigned int labelDim, unsigned int dataNum)
    {
        double gain0 = 0.0;
        double gain1 = 0.0;
        unsigned int num0 = 0;
        unsigned int num1 = 0;
        for (unsigned int i(0); i < labelDim; ++i)
        {
            num0 += cnt[0][i];
            num1 += cnt[1][i];
        }
        for (unsigned int i(0); i < labelDim; ++i)
        {
            if (cnt[0][i] != 0)
            {
                double p = ((double)cnt[0][i]) / num0;
                gain0 += p * log2(p);
            }
            if (cnt[1][i] != 0)
            {
                double p = ((double)cnt[1][i]) / num1;
                gain1 += p * log2(p);
            }
        }
        return gain0 * (((double)num0) / dataNum) + gain1 * (((double)num1) / dataNum);
    }

    template <class Type>
    double ID3Tree<Type>::discreteGain(unsigned int** cnt, unsigned int classNum, unsigned int labelDim, unsigned int dataNum)
    {
        double* gain = new double[classNum];
        unsigned int* num = new unsigned int[classNum];
        memset(gain, 0, sizeof(double) * classNum);
        memset(num, 0, sizeof(unsigned int) * classNum);
        for (unsigned int i(0); i < classNum; ++i)
        {
            for (unsigned int j(0); j < labelDim; ++j)
                num[i] += cnt[i][j];
        }
        for (unsigned int i(0); i < classNum; ++i)
        {
            for (unsigned int j(0); j < labelDim; ++j)
            {
                if (cnt[i][j] != 0)
                {
                    double p = ((double)cnt[i][j]) / num[i];
                    gain[i] += p * log2(p);
                }
            }
        }
        double result = 0;
        for (unsigned int i(0); i < classNum; ++i)
            result += gain[i] * (((double)num[i]) / dataNum);
        delete[] gain;
        delete[] num;
        return result;
    }

    template <class Type>
    CartTree<Type>::CartTree()
    {
        this->decisionTreeType = DecisionTreeType::CART;
        this->continuousGainPtr = &(this->continuousGain);
        this->discreteGainPtr = &(this->discreteGain);
    }

    template <class Type>
    double CartTree<Type>::continuousGain(unsigned int** cnt, unsigned int labelDim, unsigned int dataNum)
    {
        double gain0 = -1.0;
        double gain1 = -1.0;
        unsigned int num0 = 0;
        unsigned int num1 = 0;
        for (unsigned int i(0); i < labelDim; ++i)
        {
            num0 += cnt[0][i];
            num1 += cnt[1][i];
        }
        for (unsigned int i(0); i < labelDim; ++i)
        {
            if (cnt[0][i] != 0)
            {
                double p = ((double)cnt[0][i]) / num0;
                gain0 += p * p;
            }
            if (cnt[1][i] != 0)
            {
                double p = ((double)cnt[1][i]) / num1;
                gain1 += p * p;
            }
        }
        return gain0 * (((double)num0) / dataNum) + gain1 * (((double)num1) / dataNum);
    }

    template <class Type>
    double CartTree<Type>::discreteGain(unsigned int** cnt, unsigned int classNum, unsigned int labelDim, unsigned int dataNum)
    {
        double* gain = new double[classNum];
        unsigned int* num = new unsigned int[classNum];
        memset(num, 0, sizeof(unsigned int) * classNum);
        for (unsigned int i(0); i < classNum; ++i)
        {
            gain[i] = -1;
            for (unsigned int j(0); j < labelDim; ++j)
                num[i] += cnt[i][j];
        }
        for (unsigned int i(0); i < classNum; ++i)
        {
            for (unsigned int j(0); j < labelDim; ++j)
            {
                if (cnt[i][j] != 0)
                {
                    double p = ((double)cnt[i][j]) / num[i];
                    gain[i] += p * p;
                }
            }
        }
        double result = 0;
        for (unsigned int i(0); i < classNum; ++i)
            result += gain[i] * (((double)num[i]) / dataNum);
        delete[] gain;
        delete[] num;
        return result;
    }

    template <class Type>
    C4_5Tree<Type>::C4_5Tree()
    {
        this->decisionTreeType = DecisionTreeType::C4_5;
        this->continuousGainPtr = &(this->continuousGain);
        this->discreteGainPtr = &(this->discreteGain);
    }

    template <class Type>
    double C4_5Tree<Type>::continuousGain(unsigned int** cnt, unsigned int labelDim, unsigned int dataNum)
    {
        throw exception("Not implemented yet.");
        return 0;
    }

    template <class Type>
    double C4_5Tree<Type>::discreteGain(unsigned int** cnt, unsigned int classNum, unsigned int labelDim, unsigned int dataNum)
    {
        throw exception("Not implemented yet.");
        return 0;
    }

    template class Tree<float>;
    template class Tree<double>;
    template class DecisionTree<float>;
    template class DecisionTree<double>;
    template class ID3Tree<float>;
    template class ID3Tree<double>;
    template class CartTree<float>;
    template class CartTree<double>;
    template class C4_5Tree<float>;
    template class C4_5Tree<double>;
}}