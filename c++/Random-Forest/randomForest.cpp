#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4018)
#include "utility.h"
#include "randomForest.h"
#include "assert.h"
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <random>
#include <ctime>
#include <iostream>
#include "rapidjson/include/rapidjson/document.h"
#include "rapidjson/include/rapidjson/writer.h"
#include "rapidjson/include/rapidjson/stringbuffer.h"
#include "rapidjson/include/rapidjson/filereadstream.h"
#include "rapidjson/include/rapidjson/reader.h"
#include "rapidjson/include/rapidjson/filewritestream.h"
#include "rapidjson/include/rapidjson/writer.h"
#define JsonBufferSize (1 << 16)
using namespace std;
using namespace randomForest::utility;
using namespace randomForest::decisionTree;
using namespace randomForest::decisionTreeNodes;
using namespace randomForest::utility;
using namespace rapidjson;

namespace randomForest { namespace randomForest {

    template<class Type>
    RandomForest<Type>::RandomForest() : randomForestType(RandomForestType::ID3)
    {
    }

    template<class Type>
    RandomForest<Type>::RandomForest(RandomForestType randomForestType) : randomForestType(randomForestType)
    {
    }

    template<class Type>
    RandomForestType RandomForest<Type>::getRandomForestType()
    {
        return this->randomForestType;
    }

    template<class Type>
    void RandomForest<Type>::setRandomForestType(RandomForestType randomForestType)
    {
        this->randomForestType = randomForestType;
    }

    template<class Type>
    void RandomForest<Type>::setTreeNum(unsigned int treeNum)
    {
        this->treeNum = treeNum;
    }

    template<class Type>
    void RandomForest<Type>::setRandomDim(unsigned int randomDim)
    {
        this->randomDim = randomDim;
    }

    template<class Type>
    void RandomForest<Type>::buildRandomForest(Data<Type>* feature, Data<Type>* label, unsigned int labelDim)
    {
        unsigned int featureDim = feature->dim;
        assert(feature->num == label->num && feature->num > 0 && labelDim > 0 && featureDim > 0);
        if (0 == randomDim)
            randomDim = (unsigned int)ceil(sqrt((double)featureDim));
        if (0 == treeNum)
            treeNum = (unsigned int)ceil((sqrt((double)feature->num) * log2((double)feature->num)));
        decisionTree.resize(treeNum);
        vector<unsigned int> featureIndex;
        featureIndex.resize(randomDim);
        srand((unsigned int)time(NULL));

        for (unsigned int i(0); i < treeNum; ++i)
        {
            set<unsigned int> randSet;
            for (unsigned int j(0); j < randomDim; ++j)
            {
                while (true)
                {
                    unsigned int k = rand() % featureDim;
                    if (randSet.find(k) == randSet.end())
                    {
                        featureIndex[j] = k;
                        randSet.insert(k);
                        break;
                    }
                }
            }

            if (RandomForestType::ID3 == randomForestType)
                decisionTree[i] = new ID3Tree<Type>();
            else if (RandomForestType::Cart == randomForestType)
                decisionTree[i] = new CartTree<Type>();
            else if (RandomForestType::Mixture == randomForestType)
            {
                if (rand() % 2 == 0)
                    decisionTree[i] = new ID3Tree<Type>();
                else
                    decisionTree[i] = new CartTree<Type>();
            }
            else
                throw exception("Random forest type error: randomForestType must be [ID3 | Cart | Mixture].");
            Data<Type>* randomFeature = feature->getRefColumnSlices(featureIndex);
            decisionTree[i]->buildDecisionTree(randomFeature, label, labelDim, featureIndex);
            decisionTree[i]->treeId = i;
            delete randomFeature;
        }
    }

    template<class Type>
    unsigned int RandomForest<Type>::test(Data<Type>* feature, Data<Type>* label, vector<unsigned int> & result)
    {
        assert(feature->num == label->num && feature->num > 0);
        unsigned int dataNum = feature->num;
        result.resize(dataNum);
        vector<unsigned int> res;
        unsigned int* labelPtr = (unsigned int*)label->dataPtr[0];

        vector<map<unsigned int, unsigned int>>resultCnt;
        resultCnt.resize(dataNum);
        for (unsigned int i(0); i < treeNum; ++i)
        {
            decisionTree[i]->test(*feature, *label, res);
            for (unsigned int j(0); j < dataNum; ++j)
            {
                if(resultCnt[j].find(res[j]) == resultCnt[j].end())
                    resultCnt[j][res[j]] = 1;
                else
                    ++resultCnt[j][res[j]];
            }
        }

        /*
        for (unsigned int i(0); i < dataNum; ++i)
        {
            for (auto iter = resultCnt[i].begin(); iter != resultCnt[i].end(); ++iter)
                cout << iter->second << " ";
            puts("");
        }
        */

        for (unsigned int i(0); i < dataNum; ++i)
        {
            unsigned int maxLabel = 0;
            unsigned int maxCnt = 0;
            for (auto iter = resultCnt[i].begin(); iter != resultCnt[i].end(); ++iter)
            {
                if (iter->second > maxCnt)
                {
                    maxLabel = iter->first;
                    maxCnt = iter->second;
                }
            }
            result[i] = maxLabel;
        }
        unsigned int cnt = 0;
        for (unsigned int i(0); i < dataNum; ++i)
        {
            if (labelPtr[i] == result[i])
                ++cnt;
        }
        return cnt;
    }

    template<class Type>
    void RandomForest<Type>::load(string fileName)
    {
        FILE* file = fopen(fileName.c_str(), "rb");
        assert(file != NULL);
        char readBuffer[JsonBufferSize];
        FileReadStream jsonReader(file, readBuffer, sizeof(readBuffer));
        Document doc;
        doc.ParseStream(jsonReader);

        assert(doc.HasMember("forest_type") && doc["forest_type"].IsString());
        assert(doc.HasMember("tree_num") && doc["tree_num"].IsInt());
        assert(doc.HasMember("trees") && doc["trees"].IsArray());
        this->release();
        string forestType = doc["forest_type"].GetString();
        if ("ID3" == forestType)
            randomForestType = RandomForestType::ID3;
        else if ("CART" == forestType)
            randomForestType = RandomForestType::Cart;
        else if ("Mixture" == forestType)
            randomForestType = RandomForestType::Mixture;
        else
            throw exception(("Random forest type error: unexpected forest type [" + forestType + "] is not [ID3 | CART | Mixture].").c_str());
        treeNum = doc["tree_num"].GetInt();
        decisionTree.resize(treeNum);
        auto trees = doc["trees"].GetArray();
        if (trees.Size() != treeNum)
            throw exception(("Tree num error: expected tree num is " + to_string(treeNum) + ", but " + to_string(trees.Size()) + " was found in tree list.").c_str());

        for (unsigned int i(0); i < treeNum; ++i)
        {
            assert(trees[i].HasMember("tree_type") && trees[i]["tree_type"].IsString());
            assert(trees[i].HasMember("tree_id") && trees[i]["tree_id"].IsInt());
            assert(trees[i].HasMember("nodes") && trees[i]["nodes"].IsArray());
            string treeType = trees[i]["tree_type"].GetString();
            int treeId = trees[i]["tree_id"].GetInt();
            assert(treeId >= 0 && treeId < treeNum);
            if ("ID3" == treeType)
                decisionTree[treeId] = new ID3Tree<Type>();
            else if ("CART" == treeType)
                decisionTree[treeId] = new CartTree<Type>();
            else if ("C4.5" == treeType)
                decisionTree[treeId] = new C4_5Tree<Type>();
            else
                throw exception(("Tree type error: unexpected tree type [" + treeType + "] is not [ID3 | CART | C4.5].").c_str());
            decisionTree[treeId]->treeId = treeId;

            auto nodes = trees[i]["nodes"].GetArray();
            map<NodeId, NodeInfo<Type>*> nodeMap;
            for (unsigned int i(0); i < nodes.Size(); ++i)
            {
                assert(nodes[i].HasMember("node_type") && nodes[i]["node_type"].IsString());
                assert(nodes[i].HasMember("node_id") && nodes[i]["node_id"].IsInt());
                string nodeType = nodes[i]["node_type"].GetString();
                NodeId nodeId = nodes[i]["node_id"].GetInt();
                if ("Continuous" == nodeType)
                {
                    assert(nodes[i].HasMember("feature_index") && nodes[i]["feature_index"].IsInt());
                    assert(nodes[i].HasMember("feature_value") && nodes[i]["feature_value"].IsNumber());
                    assert(nodes[i].HasMember("max_label") && nodes[i]["max_label"].IsInt());
                    assert(nodes[i].HasMember("left_child_id") && nodes[i]["left_child_id"].IsInt());
                    assert(nodes[i].HasMember("right_child_id") && nodes[i]["right_child_id"].IsInt());
                    ContinuousNodeInfo<Type>* nodeInfo = new ContinuousNodeInfo<Type>(nodeId, NodeType::Continuous,
                        nodes[i]["feature_index"].GetInt(), (Type)(nodes[i]["feature_value"].GetDouble()), nodes[i]["max_label"].GetInt(), nodes[i]["left_child_id"].GetInt(), nodes[i]["right_child_id"].GetInt());
                    nodeMap.insert({ nodeId,nodeInfo });
                }
                else if ("Discrete" == nodeType)
                {
                    assert(nodes[i].HasMember("feature_index") && nodes[i]["feature_index"].IsInt());
                    assert(nodes[i].HasMember("feature_list") && nodes[i]["feature_list"].IsArray());
                    vector<string> featureList;
                    for (SizeType j(0); j < nodes[i]["feature_list"].Size(); ++j)
                        featureList.push_back(string(nodes[i]["feature_list"][j].GetString()));
                    assert(nodes[i].HasMember("max_label") && nodes[i]["max_label"].IsInt());
                    assert(nodes[i].HasMember("child_id_list") && nodes[i]["child_id_list"].IsArray());
                    vector<NodeId> childIdList;
                    for (SizeType j(0); j < nodes[i]["child_id_list"].Size(); ++j)
                        childIdList.push_back(nodes[i]["child_id_list"][j].GetInt());
                    DiscreteNodeInfo<Type>* nodeInfo = new DiscreteNodeInfo<Type>(nodeId, NodeType::Discrete,
                        nodes[i]["feature_index"].GetInt(), featureList, nodes[i]["max_label"].GetInt(), childIdList);
                    nodeMap.insert({ nodeId,nodeInfo });
                }
                else if ("Leaf" == nodeType)
                {
                    assert(nodes[i].HasMember("label") && nodes[i]["label"].IsInt());
                    LeafNodeInfo<Type>* nodeInfo = new LeafNodeInfo<Type>(nodeId, NodeType::Leaf,
                        nodes[i]["label"].GetInt());
                    nodeMap.insert({ nodeId,nodeInfo });
                }
                else
                    throw exception(("Node type error: unexpected node type [" + nodeType + "] is not [Continuous | Discrete | Leaf].").c_str());
            }
            decisionTree[treeId]->deserialize(nodeMap);
            for (auto it = nodeMap.begin(); it != nodeMap.end(); ++it)
                delete it->second;
        }

        fclose(file);
    }

    template<class Type>
    void RandomForest<Type>::save(string fileName)
    {
        FILE* file = fopen(fileName.c_str(), "wb");
        assert(file != NULL);
        char writeBuffer[JsonBufferSize];
        FileWriteStream jsonWriter(file, writeBuffer, sizeof(writeBuffer));
        Document doc;
        Document::AllocatorType& docAllocator = doc.GetAllocator();
        doc.SetObject();
        Value _str;
        _str.SetString(StringRef(forestTypeStr[randomForestType].c_str(), forestTypeStr[randomForestType].length()));
        doc.AddMember("forest_type", _str, docAllocator);
        doc.AddMember("tree_num", treeNum, docAllocator);

        Value _treeVal(kArrayType);
        for (unsigned int i(0); i < treeNum; ++i)
        {
            Value treeVal(kObjectType);
            assert(decisionTree[i]->treeId != -1);
            treeVal.AddMember("tree_id", decisionTree[i]->treeId, docAllocator);
            Value _s;
            _s.SetString(StringRef(treeTypeStr[decisionTree[i]->getDecisionTreeType()].c_str(), treeTypeStr[decisionTree[i]->getDecisionTreeType()].length()));
            treeVal.AddMember("tree_type", _s, docAllocator);
            Value treeNodeVal(kArrayType);
            map<NodeId, NodeInfo<Type>*> nodeMap;
            decisionTree[i]->serialize(nodeMap);
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
                treeNodeVal.PushBack(val, docAllocator);
            }
            treeVal.AddMember("nodes", treeNodeVal, docAllocator);
            _treeVal.PushBack(treeVal, docAllocator);
            for (auto it = nodeMap.begin(); it != nodeMap.end(); ++it)
                delete it->second;
        }
        doc.AddMember("trees", _treeVal, docAllocator);

        Writer<FileWriteStream> writer(jsonWriter);
        doc.Accept(writer);
        fclose(file);
    }

    template<class Type>
    void RandomForest<Type>::release()
    {
        for (unsigned int i(0); i < treeNum; ++i)
            decisionTree[i]->release();
        decisionTree.clear();
        treeNum = 0;
    }

    template <class Type>
    RandomForest<Type>::~RandomForest()
    {
        this->release();
    }

    template class RandomForest<float>;
    template class RandomForest<double>;
}}