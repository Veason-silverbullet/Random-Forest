#include "utility.h"
#include "decisionTree.h"
#include "randomForest.h"
#include "test.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <time.h>
#define DisplayTime
using namespace std;
using namespace randomForest::utility;
using namespace randomForest::decisionTree;
using namespace randomForest::randomForest;

int main()
{
    test<float>();

#ifdef DisplayTime
    clock_t startTime = clock();
#endif

    Data<double>* feature = new Data<double>();
    Data<double>* label = new Data<double>();
    map<string, unsigned int> labelMap;
    Data<double>::loadCsvData(*feature, *label, labelMap, "../../data/data.csv", true, "../../data/feature_attribute.txt", "../../data/index.txt");
    RandomForest<double> randomForest = RandomForest<double>();
    randomForest.setRandomForestType(RandomForestType::Mixture);
    randomForest.setRandomDim(max(min(feature->dim, (unsigned int)ceil(feature->dim * 0.75)), (unsigned int)1));
    randomForest.buildRandomForest(feature, label, 2);
    vector<unsigned int> result;
    unsigned int cnt = randomForest.test(feature, label, result);
    printf("%.16lf\n", ((double)cnt) / feature->num);
    randomForest.save("../../data/model.json");

#ifdef DisplayTime
    clock_t endTime = clock();
    printf("Used time : %.6lfs.\n", ((double)(endTime - startTime)) / 1000);
#endif

    system("pause");
    return 0;
}