#pragma once
#include <vector>
#include <string>
#include <map>
using namespace std;

namespace randomForest { namespace decisionTreeNodes {

    typedef unsigned int NodeId;
    enum NodeType
    {
        Continuous,
        Discrete,
        Leaf
    };
    static const string nodeTypeStr[3] = { "Continuous", "Discrete", "Leaf" };

    template <class Type>
    class NodeInfo
    {
    public:
        NodeId nodeId;
        NodeType nodeType;

        NodeInfo();
        NodeInfo(NodeId nodeId, NodeType nodeType);
        void printInfo();
    };

    template <class Type>
    class ContinuousNodeInfo : public NodeInfo<Type>
    {
    public:
        unsigned int featureIndex;
        Type featureValue;
        unsigned int maxLabel;
        NodeId leftChildId;
        NodeId rightChildId;

        ContinuousNodeInfo();
        ContinuousNodeInfo(NodeId nodeId, NodeType nodeType,
            unsigned int featureIndex, Type featureValue, unsigned int maxLabel, NodeId leftChildId, NodeId rightChildId);
        void printInfo();
    };

    template <class Type>
    class DiscreteNodeInfo : public NodeInfo<Type>
    {
    public:
        unsigned int featureIndex;
        vector<string> featureList;
        unsigned int maxLabel;
        vector<NodeId> childIdList;

        DiscreteNodeInfo();
        DiscreteNodeInfo(NodeId nodeId, NodeType nodeType,
            unsigned int featureIndex, const vector<string> & featureList, unsigned int maxLabel, const vector<NodeId> & childIdList);
        void printInfo();
    };

    template <class Type>
    class LeafNodeInfo : public NodeInfo<Type>
    {
    public:
        unsigned int label;

        LeafNodeInfo();
        LeafNodeInfo(NodeId nodeId, NodeType nodeType,
            unsigned int label);
        void printInfo();
    };

    template <class Type>
    class Node
    {
    public:
        NodeId nodeId;
        NodeType nodeType;
        vector<bool> usedFeature;

        virtual NodeType getNodeType() = 0;
        virtual void serialize(NodeInfo<Type>* nodeInfo) = 0;
        virtual void deserialize(const NodeInfo<Type>* nodeInfo) = 0;
    };

    template <class Type>
    class ContinuousNode : public Node<Type>
    {
    public:
        unsigned int featureIndex;
        Type featureValue;
        unsigned int maxLabel;
        Node<Type>* leftChild;
        Node<Type>* rightChild;
        vector<unsigned int> tempIndex0;
        vector<unsigned int> tempIndex1;

        ContinuousNode();
        explicit ContinuousNode(NodeId nodeId);
        ContinuousNode(NodeId nodeId, unsigned int featureIndex, Type featureValue, unsigned int maxLabel);
        NodeType getNodeType();
        void serialize(NodeInfo<Type>* nodeInfo);
        void deserialize(const NodeInfo<Type>* nodeInfo);
    };

    template <class Type>
    class DiscreteNode : public Node<Type>
    {
    public:
        unsigned int featureIndex;
        vector<string> featureList;
        unsigned int maxLabel;
        vector<Node<Type>*> childList;
        vector<vector<unsigned int>>tempIndex;

        DiscreteNode();
        explicit DiscreteNode(NodeId nodeId);
        DiscreteNode(NodeId nodeId, unsigned int featureIndex, const vector<string> & featureList, unsigned int maxLabel);
        NodeType getNodeType();
        void serialize(NodeInfo<Type>* nodeInfo);
        void deserialize(const NodeInfo<Type>* nodeInfo);
    };

    template <class Type>
    class LeafNode : public Node<Type>
    {
    public:
        unsigned int label;

        LeafNode();
        explicit LeafNode(NodeId nodeId);
        LeafNode(NodeId nodeId, unsigned int label);
        NodeType getNodeType();
        void serialize(NodeInfo<Type>* nodeInfo);
        void deserialize(const NodeInfo<Type>* nodeInfo);
    };
}}