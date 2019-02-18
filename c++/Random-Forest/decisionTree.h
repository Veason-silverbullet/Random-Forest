#pragma once
# include "decisionTreeNodes.h"
# include "utility.h"
# include <vector>
# include <map>
using namespace std;
using namespace randomForest::decisionTreeNodes;
using namespace randomForest::utility;

namespace randomForest { namespace decisionTree {

    enum DecisionTreeType
    {
        ID3,
        CART,
        C4_5
    };
    static const string treeTypeStr[3] = { "ID3", "CART", "C4.5" };

    template <class Type>
    struct FeatureLabelBundle
    {
        Type feature;
        unsigned int label;
        bool friend operator<(const FeatureLabelBundle & a, const FeatureLabelBundle & b)
        {
            return a.feature < b.feature;
        }
    };

    template <class Type>
    class Tree
    {
    public:
        int treeId;
        unsigned int nodeId;
        Node<Type>* root;

        Tree<Type>();
        void setTreeId(int treeId);
        void serialize(map<NodeId, NodeInfo<Type>*> & nodeMap);
        void deserialize(const map<NodeId, NodeInfo<Type>*> & nodeMap);
        void load(string fileName);
        void save(string fileName);
        void release();
        ~Tree();
    };

    template <class Type>
    class DecisionTree : public Tree<Type>
    {
    public:
        Data<Type>* feature;
        Data<Type>* label;
        unsigned int labelDim;
        vector<unsigned int> featureIndex;
        unsigned int** binCnt;
        vector<unsigned int**> multiCnt;
        typedef double(*continuousGainFunPtr)(unsigned int**, unsigned int, unsigned int);
        continuousGainFunPtr continuousGainPtr;
        typedef double(*discreteGainFunPtr)(unsigned int**, unsigned int, unsigned int, unsigned int);
        discreteGainFunPtr discreteGainPtr;

        DecisionTreeType decisionTreeType;
        DecisionTreeType getDecisionTreeType();
        LeafNode<Type>* generateLeafNode(vector<unsigned int> & dataIndex, vector<bool> & usedFeature);
        Node<Type>* createNode(vector<unsigned int> & dataIndex, vector<vector<string>> & featureList, vector<map<string, unsigned int>> & featureMap, vector<bool> & usedFeature);
        void buildDecisionTree(Data<Type>* feature, Data<Type>* label, unsigned int labelDim, const vector<unsigned int> & featureIndex);
        unsigned int test(Data<Type> & feature, Data<Type> & label, vector<unsigned int> & result);
    };

    template <class Type>
    class ID3Tree : public DecisionTree<Type>
    {
    public:
        ID3Tree();
        static double continuousGain(unsigned int** cnt, unsigned int labelDim, unsigned int dataNum);
        static double discreteGain(unsigned int** cnt, unsigned int classNum, unsigned int labelDim, unsigned int dataNum);
    };

    template <class Type>
    class CartTree : public DecisionTree<Type>
    {
    public:
        CartTree();
        static double continuousGain(unsigned int** cnt, unsigned int labelDim, unsigned int dataNum);
        static double discreteGain(unsigned int** cnt, unsigned int classNum, unsigned int labelDim, unsigned int dataNum);
    };

    template <class Type>
    class C4_5Tree : public DecisionTree<Type>
    {
    public:
        C4_5Tree();
        static double continuousGain(unsigned int** cnt, unsigned int labelDim, unsigned int dataNum);
        static double discreteGain(unsigned int** cnt, unsigned int classNum, unsigned int labelDim, unsigned int dataNum);
    };
}}