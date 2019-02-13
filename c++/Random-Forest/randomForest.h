#pragma once
#include "utility.h"
#include "decisionTree.h"
using namespace std;
using namespace randomForest::utility;
using namespace randomForest::decisionTree;

namespace randomForest { namespace randomForest {

    enum RandomForestType
    {
        ID3,
        Cart,
        Mixture
    };

    template<class Type>
    class RandomForest
    {
    public:
        RandomForestType randomForestType;
        unsigned int treeNum;
        unsigned int randomDim;
        unsigned int labelDim;
        vector<DecisionTree<Type>*> decisionTree;

        RandomForest();
        RandomForest(RandomForestType randomForestType);
        void setRandomForestType(RandomForestType randomForestType);
        void setTreeNum(unsigned int treeNum);
        void setRandomDim(unsigned int randomDim);
        void buildRandomForest(Data<Type>* feature, Data<Type>* label, unsigned int labelDim);
        unsigned int test(Data<Type>* feature, Data<Type>* label, vector<unsigned int> & result);
    };
}}