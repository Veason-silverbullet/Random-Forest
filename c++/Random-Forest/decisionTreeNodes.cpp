#include "decisionTreeNodes.h"
#include "assert.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <queue>
using namespace std;

namespace randomForest { namespace decisionTreeNodes {

    template <class Type>
    NodeInfo<Type>::NodeInfo()
    {
    }

    template <class Type>
    NodeInfo<Type>::NodeInfo(NodeId nodeId, NodeType nodeType) : nodeId(nodeId), nodeType(nodeType)
    {
    }

    template <class Type>
    void NodeInfo<Type>::printInfo()
    {
        cout << "node_type = " << nodeTypeStr[this->nodeType] << endl;
        cout << "node_id = " << this->nodeId << endl;
    }

    template <class Type>
    ContinuousNodeInfo<Type>::ContinuousNodeInfo()
    {
        this->nodeType = NodeType::Continuous;
    }

    template <class Type>
    ContinuousNodeInfo<Type>::ContinuousNodeInfo(NodeId nodeId, NodeType nodeType,
        unsigned int featureIndex, Type featureValue, unsigned int maxLabel, NodeId leftChildId, NodeId rightChildId) :
        NodeInfo<Type>(nodeId, nodeType),
        featureIndex(featureIndex), featureValue(featureValue), maxLabel(maxLabel), leftChildId(leftChildId), rightChildId(rightChildId)
    {
        assert(this->nodeType == NodeType::Continuous);
    }

    template <class Type>
    void ContinuousNodeInfo<Type>::printInfo()
    {
        cout << "node_type = " << nodeTypeStr[this->nodeType] << endl;
        cout << "feature_index = " << this->featureIndex << endl;
        cout << "feature_value = " << this->featureValue << endl;
        cout << "max_label = " << this->maxLabel << endl;
        cout << "left_child_id = " << this->leftChildId << endl;
        cout << "right_child_id = " << this->rightChildId << endl;
    }

    template <class Type>
    DiscreteNodeInfo<Type>::DiscreteNodeInfo()
    {
        this->nodeType = NodeType::Discrete;
    }

    template <class Type>
    DiscreteNodeInfo<Type>::DiscreteNodeInfo(NodeId nodeId, NodeType nodeType,
        unsigned int featureIndex, const vector<string> & featureList, unsigned int maxLabel, const vector<NodeId> & childIdList) :
        NodeInfo<Type>(nodeId, nodeType),
        featureIndex(featureIndex), maxLabel(maxLabel)
    {
        assert(this->nodeType == NodeType::Discrete);
        assert(0 == this->featureList.size());
        for (int i(0); i < featureList.size(); ++i)
            this->featureList.push_back(featureList[i]);
        assert(0 == this->childIdList.size());
        for (int i(0); i < childIdList.size(); ++i)
            this->childIdList.push_back(childIdList[i]);
    }

    template <class Type>
    void DiscreteNodeInfo<Type>::printInfo()
    {
        cout << "node_type = " << nodeTypeStr[this->nodeType] << endl;
        cout << "node_id = " << this->nodeId << endl;
        cout << "feature_index = " << this->featureIndex << endl;
        cout << "feature_list = [";
        for (int i(0); i < featureList.size(); ++i)
            cout << featureList[i] << ((i != featureList.size() - 1) ? ", " : "]\n");
        cout << "max_label = " << this->maxLabel << endl;
        cout << "child_id_list = [";
        for (int i(0); i < featureList.size(); ++i)
            cout << featureList[i] << ((i != featureList.size() - 1) ? ", " : "]\n");
    }

    template <class Type>
    LeafNodeInfo<Type>::LeafNodeInfo()
    {
        this->nodeType = NodeType::Leaf;
    }

    template <class Type>
    LeafNodeInfo<Type>::LeafNodeInfo(NodeId nodeId, NodeType nodeType,
        unsigned int label) :
        NodeInfo<Type>(nodeId, nodeType),
        label(label)
    {
        assert(this->nodeType == NodeType::Leaf);
    }

    template <class Type>
    void LeafNodeInfo<Type>::printInfo()
    {
        cout << "node_type = " << nodeTypeStr[this->nodeType] << endl;
        cout << "node_id = " << this->nodeId << endl;
        cout << "label = " << this->label << endl;
    }

    template <class Type>
    ContinuousNode<Type>::ContinuousNode()
    {
        this->nodeType = NodeType::Continuous;
    }

    template <class Type>
    ContinuousNode<Type>::ContinuousNode(NodeId nodeId)
    {
        this->nodeType = NodeType::Continuous;
        this->nodeId = nodeId;
    }

    template <class Type>
    ContinuousNode<Type>::ContinuousNode(NodeId nodeId, unsigned int featureIndex, Type featureValue, unsigned int maxLabel) :
        featureIndex(featureIndex), featureValue(featureValue), maxLabel(maxLabel)
    {
        this->nodeType = NodeType::Continuous;
        this->nodeId = nodeId;
    }

    template <class Type>
    NodeType ContinuousNode<Type>::getNodeType()
    {
        assert(this->nodeType == NodeType::Continuous);
        return NodeType::Continuous;
    }

    template <class Type>
    void ContinuousNode<Type>::serialize(NodeInfo<Type>* nodeInfo)
    {
        assert(nodeInfo->nodeType == NodeType::Continuous);
        ((ContinuousNodeInfo<Type>*)nodeInfo)->nodeId = this->nodeId;
        ((ContinuousNodeInfo<Type>*)nodeInfo)->nodeType = this->nodeType;
        ((ContinuousNodeInfo<Type>*)nodeInfo)->featureIndex = this->featureIndex;
        ((ContinuousNodeInfo<Type>*)nodeInfo)->featureValue = this->featureValue;
        ((ContinuousNodeInfo<Type>*)nodeInfo)->maxLabel = this->maxLabel;
        ((ContinuousNodeInfo<Type>*)nodeInfo)->leftChildId = this->leftChild->nodeId;
        ((ContinuousNodeInfo<Type>*)nodeInfo)->rightChildId = this->rightChild->nodeId;
    }

    template <class Type>
    void ContinuousNode<Type>::deserialize(const NodeInfo<Type>* nodeInfo)
    {
        assert(nodeInfo->nodeType == NodeType::Continuous);
        this->nodeId = ((ContinuousNodeInfo<Type>*)nodeInfo)->nodeId;
        this->nodeType = ((ContinuousNodeInfo<Type>*)nodeInfo)->nodeType;
        this->featureIndex = ((ContinuousNodeInfo<Type>*)nodeInfo)->featureIndex;
        this->featureValue = ((ContinuousNodeInfo<Type>*)nodeInfo)->featureValue;
        this->maxLabel = ((ContinuousNodeInfo<Type>*)nodeInfo)->maxLabel;
    }

    template <class Type>
    DiscreteNode<Type>::DiscreteNode()
    {
        this->nodeType = NodeType::Discrete;
    }

    template <class Type>
    DiscreteNode<Type>::DiscreteNode(NodeId nodeId)
    {
        this->nodeType = NodeType::Discrete;
        this->nodeId = nodeId;
    }

    template <class Type>
    DiscreteNode<Type>::DiscreteNode(NodeId nodeId, unsigned int featureIndex, const vector<string> & featureList, unsigned int maxLabel) :
        featureIndex(featureIndex), maxLabel(maxLabel)
    {
        this->nodeType = NodeType::Discrete;
        this->nodeId = nodeId;
        assert(0 == this->featureList.size());
        for (int i(0); i < featureList.size(); ++i)
            this->featureList.push_back(featureList[i]);
    }

    template <class Type>
    NodeType DiscreteNode<Type>::getNodeType()
    {
        assert(this->nodeType == NodeType::Discrete);
        return NodeType::Discrete;
    }

    template <class Type>
    void DiscreteNode<Type>::serialize(NodeInfo<Type>* nodeInfo)
    {
        assert(nodeInfo->nodeType == NodeType::Discrete);
        ((DiscreteNodeInfo<Type>*)nodeInfo)->nodeId = this->nodeId;
        ((DiscreteNodeInfo<Type>*)nodeInfo)->nodeType = this->nodeType;
        ((DiscreteNodeInfo<Type>*)nodeInfo)->featureIndex = this->featureIndex;
        for (unsigned int i(0); i < this->featureList.size(); ++i)
            ((DiscreteNodeInfo<Type>*)nodeInfo)->featureList.push_back(this->featureList[i]);
        for (NodeId nodeId(0); nodeId < this->childList.size(); ++nodeId)
            ((DiscreteNodeInfo<Type>*)nodeInfo)->childIdList.push_back((this->childList[nodeId])->nodeId);
        ((DiscreteNodeInfo<Type>*)nodeInfo)->maxLabel = this->maxLabel;
    }

    template <class Type>
    void DiscreteNode<Type>::deserialize(const NodeInfo<Type>* nodeInfo)
    {
        assert(nodeInfo->nodeType == NodeType::Discrete);
        this->nodeId = ((DiscreteNodeInfo<Type>*)nodeInfo)->nodeId;
        this->nodeType = ((DiscreteNodeInfo<Type>*)nodeInfo)->nodeType;
        this->featureIndex = ((DiscreteNodeInfo<Type>*)nodeInfo)->featureIndex;
        for (unsigned int i(0); i < ((DiscreteNodeInfo<Type>*)nodeInfo)->featureList.size(); ++i)
            this->featureList.push_back(((DiscreteNodeInfo<Type>*)nodeInfo)->featureList[i]);
        this->maxLabel = ((DiscreteNodeInfo<Type>*)nodeInfo)->maxLabel;
    }

    template <class Type>
    LeafNode<Type>::LeafNode()
    {
        this->nodeType = NodeType::Leaf;
    }

    template <class Type>
    LeafNode<Type>::LeafNode(NodeId nodeId)
    {
        this->nodeType = NodeType::Leaf;
        this->nodeId = nodeId;
    }

    template <class Type>
    LeafNode<Type>::LeafNode(NodeId nodeId, unsigned int label) :
        label(label)
    {
        this->nodeType = NodeType::Leaf;
        this->nodeId = nodeId;
    }

    template <class Type>
    NodeType LeafNode<Type>::getNodeType()
    {
        assert(this->nodeType == NodeType::Leaf);
        return NodeType::Leaf;
    }

    template <class Type>
    void LeafNode<Type>::serialize(NodeInfo<Type>* nodeInfo)
    {
        assert(nodeInfo->nodeType == NodeType::Leaf);
        ((LeafNodeInfo<Type>*)nodeInfo)->nodeId = this->nodeId;
        ((LeafNodeInfo<Type>*)nodeInfo)->nodeType = this->nodeType;
        ((LeafNodeInfo<Type>*)nodeInfo)->label = this->label;
    }

    template <class Type>
    void LeafNode<Type>::deserialize(const NodeInfo<Type>* nodeInfo)
    {
        assert(nodeInfo->nodeType == NodeType::Leaf);
        this->nodeId = ((LeafNodeInfo<Type>*)nodeInfo)->nodeId;
        this->nodeType = ((LeafNodeInfo<Type>*)nodeInfo)->nodeType;
        this->label = ((LeafNodeInfo<Type>*)nodeInfo)->label;
    }


    template class NodeInfo<float>;
    template class NodeInfo<double>;
    template class ContinuousNodeInfo<float>;
    template class ContinuousNodeInfo<double>;
    template class DiscreteNodeInfo<float>;
    template class DiscreteNodeInfo<double>;
    template class LeafNodeInfo<float>;
    template class LeafNodeInfo<double>;
    template class ContinuousNode<float>;
    template class ContinuousNode<double>;
    template class DiscreteNode<float>;
    template class DiscreteNode<double>;
    template class LeafNode<float>;
    template class LeafNode<double>;
}}