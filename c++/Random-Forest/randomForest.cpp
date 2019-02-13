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
using namespace std;
using namespace randomForest::utility;
using namespace randomForest::decisionTree;

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

    template class RandomForest<float>;
    template class RandomForest<double>;
}}