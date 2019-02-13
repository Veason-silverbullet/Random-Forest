#pragma once
#include <vector>
#include <string>
#include <map>
using namespace std;

namespace randomForest { namespace utility {

    enum DataType
    {
        Unsigned32,
        Unsigned64,
        Real,
        String
    };

    template <class Type>
    class Data
    {
    public:
        vector<DataType> dataType;
        vector<void*> dataPtr;
        unsigned int dim;
        unsigned int num;
        bool isRef;

        Data<Type>();
        Data<Type>(unsigned int dim, unsigned int num);
        Data<Type>(bool isRef);
        Data<Type>(unsigned int dim, unsigned int num, bool isRef);
        ~Data<Type>();
        Data<Type>& getRowSlices(const vector<unsigned int> & rowIndex);
        Data<Type>& getColumnSlices(const vector<unsigned int> & columnIndex);
        Data<Type>* getRefColumnSlices(const vector<unsigned int> & columnIndex);
        Data<Type>& getSlices(const vector<unsigned int> & rowIndex, const vector<unsigned int> & columnIndex);
        static void loadCsvData(Data<Type> & feature, Data<Type> & label, map<string, unsigned int> & labelMap, string csvFilePath, bool csvHeader = true, string featureAttributeFilePath = "", string mapFilePath = "");
        void printData();
        void release();
    };
}}