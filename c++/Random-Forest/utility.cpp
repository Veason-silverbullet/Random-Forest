#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4244)
#include "utility.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <io.h>
#include <regex>
using namespace std;

namespace randomForest { namespace utility {

    template <class Type>
    Data<Type>::Data() : dim(0), num(0), isRef(false)
    {
    }

    template <class Type>
    Data<Type>::Data(unsigned int dim, unsigned int num) : dim(dim), num(num), isRef(false)
    {
    }

    template <class Type>
    Data<Type>::Data(bool isRef) : dim(0), num(0), isRef(isRef)
    {
    }

    template <class Type>
    Data<Type>::Data(unsigned int dim, unsigned int num, bool isRef) : dim(dim), num(num), isRef(isRef)
    {
    }

    template <class Type>
    Data<Type>::~Data<Type>()
    {
        release();
    }

    template <class Type>
    Data<Type>& Data<Type>::getRowSlices(const vector<unsigned int> & rowIndex)
    {
        unsigned int rowNum = (unsigned int)rowIndex.size();
        unsigned int columnNum = dim;
        Data<Type>* data = new Data<Type>(columnNum, rowNum);
        data->dataType.resize(columnNum);
        data->dataPtr.resize(columnNum);
        for (unsigned int i(0); i < columnNum; ++i)
        {
            data->dataType[i] = dataType[i];
            if (DataType::Unsigned32 == dataType[i])
            {
                unsigned int* srcDataPtr = (unsigned int*)dataPtr[i];
                unsigned int* dstDataPtr = new unsigned int[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[rowIndex[j]];
                data->dataPtr[i] = dstDataPtr;
            }
            else if (DataType::Unsigned64 == dataType[i])
            {
                unsigned long* srcDataPtr = (unsigned long*)dataPtr[i];
                unsigned long* dstDataPtr = new unsigned long[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[rowIndex[j]];
                data->dataPtr[i] = dstDataPtr;
            }
            else if (DataType::Real == dataType[i])
            {
                Type* srcDataPtr = (Type*)dataPtr[i];
                Type* dstDataPtr = new Type[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[rowIndex[j]];
                data->dataPtr[i] = dstDataPtr;
            }
            else if (DataType::String == dataType[i])
            {
                string* srcDataPtr = (string*)dataPtr[i];
                string* dstDataPtr = new string[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[rowIndex[j]];
                data->dataPtr[i] = dstDataPtr;
            }
            else
                throw exception("Unexpected type error: data type must be [Unsigned32 | Unsigned64 | Real | String]");
        }
        return *data;
    }

    template <class Type>
    Data<Type>& Data<Type>::getColumnSlices(const vector<unsigned int> & columnIndex)
    {
        unsigned int rowNum = num;
        unsigned int columnNum = (unsigned int)columnIndex.size();
        Data<Type>* data = new Data<Type>(columnNum, rowNum);
        data->dataType.resize(columnNum);
        data->dataPtr.resize(columnNum);
        for (unsigned int i(0); i < columnNum; ++i)
        {
            data->dataType[i] = dataType[columnIndex[i]];
            if (DataType::Unsigned32 == dataType[columnIndex[i]])
            {
                unsigned int* srcDataPtr = (unsigned int*)dataPtr[columnIndex[i]];
                unsigned int* dstDataPtr = new unsigned int[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[j];
                data->dataPtr[i] = dstDataPtr;
            }
            else if (DataType::Unsigned64 == dataType[columnIndex[i]])
            {
                unsigned long* srcDataPtr = (unsigned long*)dataPtr[columnIndex[i]];
                unsigned long* dstDataPtr = new unsigned long[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[j];
                data->dataPtr[i] = dstDataPtr;
            }
            else if (DataType::Real == dataType[columnIndex[i]])
            {
                Type* srcDataPtr = (Type*)dataPtr[columnIndex[i]];
                Type* dstDataPtr = new Type[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[j];
                data->dataPtr[i] = dstDataPtr;
            }
            else if (DataType::String == dataType[columnIndex[i]])
            {
                string* srcDataPtr = (string*)dataPtr[columnIndex[i]];
                string* dstDataPtr = new string[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[j];
                data->dataPtr[i] = dstDataPtr;
            }
            else
                throw exception("Unexpected type error: data type must be [Unsigned32 | Unsigned64 | Real | String]");
        }
        return *data;
    }

    template <class Type>
    Data<Type>* Data<Type>::getRefColumnSlices(const vector<unsigned int> & columnIndex)
    {
        unsigned int rowNum = num;
        unsigned int columnNum = (unsigned int)columnIndex.size();
        Data<Type>* data = new Data<Type>(columnNum, rowNum, true);
        data->dataType.resize(columnNum);
        data->dataPtr.resize(columnNum);
        for (unsigned int i(0); i < columnNum; ++i)
        {
            data->dataType[i] = dataType[columnIndex[i]];
            data->dataPtr[i] = dataPtr[columnIndex[i]];
        }
        return data;
    }

    template <class Type>
    Data<Type>& Data<Type>::getSlices(const vector<unsigned int> & rowIndex, const vector<unsigned int> & columnIndex)
    {
        unsigned int rowNum = (unsigned int)rowIndex.size();
        unsigned int columnNum = (unsigned int)columnIndex.size();
        Data<Type>* data = new Data<Type>(columnNum, rowNum);
        data->dataType.resize(columnNum);
        data->dataPtr.resize(columnNum);
        for (unsigned int i(0); i < columnNum; ++i)
        {
            data->dataType[i] = dataType[columnIndex[i]];
            if (DataType::Unsigned32 == dataType[columnIndex[i]])
            {
                unsigned int* srcDataPtr = (unsigned int*)dataPtr[columnIndex[i]];
                unsigned int* dstDataPtr = new unsigned int[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[rowIndex[j]];
                data->dataPtr[i] = dstDataPtr;
            }
            else if (DataType::Unsigned64 == dataType[columnIndex[i]])
            {
                unsigned long* srcDataPtr = (unsigned long*)dataPtr[columnIndex[i]];
                unsigned long* dstDataPtr = new unsigned long[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[rowIndex[j]];
                data->dataPtr[i] = dstDataPtr;
            }
            else if (DataType::Real == dataType[columnIndex[i]])
            {
                Type* srcDataPtr = (Type*)dataPtr[columnIndex[i]];
                Type* dstDataPtr = new Type[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[rowIndex[j]];
                data->dataPtr[i] = dstDataPtr;
            }
            else if (DataType::String == dataType[columnIndex[i]])
            {
                string* srcDataPtr = (string*)dataPtr[columnIndex[i]];
                string* dstDataPtr = new string[rowNum];
                for (unsigned int j(0); j < rowNum; ++j)
                    dstDataPtr[j] = srcDataPtr[rowIndex[j]];
                data->dataPtr[i] = dstDataPtr;
            }
            else
                throw exception("Unexpected type error: data type must be [Unsigned32 | Unsigned64 | Real | String]");
        }
        return *data;
    }

    template <class Type>
    void Data<Type>::printData()
    {
        for (unsigned int i(0); i < num; ++i)
        {
            for (unsigned int j(0); j < dim; ++j)
            {
                if (DataType::Unsigned32 == dataType[j])
                    cout << ((unsigned int*)dataPtr[j])[i] << " ";
                else if (DataType::Unsigned64 == dataType[j])
                    cout << ((unsigned long*)dataPtr[j])[i] << " ";
                else if (DataType::Real == dataType[j])
                    cout << ((Type*)dataPtr[j])[i] << " ";
                else if (DataType::String == dataType[j])
                    cout << ((string*)dataPtr[j])[i] << " ";
                else
                    throw exception("Unexpected type error: data type must be [Unsigned32 | Unsigned64 | Real | String]");
            }
            cout << endl;
        }
    }

    size_t split(string & line, vector<string> & strVec)
    {
        size_t size = strVec.size();
        size_t len = line.length();
        string temp = "";
        size_t i = 0;
        while (i < len && (line[i] == ' ' || line[i] == '\t' || line[i] == '\n' || line[i] == ','))
            ++i;
        for (; i < len; ++i)
        {
            if (line[i] == ' ' || line[i] == '\t' || line[i] == '\n' || line[i] == ',')
            {
                strVec.push_back(temp);
                while (i < len && (line[i] == ' ' || line[i] == '\t' || line[i] == '\n' || line[i] == ','))
                    ++i;
                temp = line[i];
            }
            else
                temp += line[i];
        }
        if (temp != "")
            strVec.push_back(temp);
        return strVec.size() - size;
    }

    void loadMapFile(map<string, unsigned int> & labelMap, string mapFilePath)
    {
        ifstream mapFile(mapFilePath, ios::in);
        string line;
        unsigned int lineIndex = 0;
        while (getline(mapFile, line))
        {
            vector<string> strVec;
            split(line, strVec);
            ++lineIndex;
            if (strVec.size() == 0)
                continue;
            else if (strVec.size() == 2)
                labelMap[strVec[0]] = stoi(strVec[1]);
            else
                throw exception(("Map file error: illegal format in map file, line " + to_string(lineIndex) + ".").c_str());
        }
        mapFile.close();
    }

    template <class Type>
    void loadFeatureAttributeFile(string featureAttributeFilePath, Data<Type> & data)
    {
        ifstream featureAttributeFile(featureAttributeFilePath, ios::in);
        string line;
        vector<string> strVec;
        while (getline(featureAttributeFile, line))
        {
            strVec.clear();
            split(line, strVec);
            if (strVec.size() == 0)
                continue;
            else if (strVec.size() == 1)
            {
                auto lowcase = [](string str)
                {
                    string _str = str;
                    size_t len = _str.length();
                    for (size_t i(0); i < len; ++i)
                    {
                        if (_str[i] >= 'A' && _str[i] <= 'Z')
                            _str[i] += 'a' - 'A';
                    }
                    return _str;
                };
                if (lowcase(strVec[0]) == "continuous")
                    data.dataType.push_back(DataType::Real);
                else if (lowcase(strVec[0]) == "discrete")
                    data.dataType.push_back(DataType::String);
                else
                    throw exception("Feature attribute error: feature attribute must be [Continuous | Discrete].");
            }
            else
                throw exception("Feature attribute error: illegal format.");
        }
        if (data.dataType.size() < data.dim)
            throw exception(("Feature attribute error: expected attribute num is " + to_string(data.dim) + ", but only " + to_string(data.dataType.size()) + "was provided.").c_str());
        else if (data.dataType.size() > data.dim)
            cout << "Warning: expected attribute num is " + to_string(data.dim) + ", but " + to_string(data.dataType.size()) + " was provided." << endl;
        featureAttributeFile.close();
    }

    template <class Type>
    void Data<Type>::loadCsvData(Data<Type> & feature, Data<Type> & label, map<string, unsigned int> & labelMap, string csvFilePath, bool csvHeader, string featureAttributeFilePath, string mapFilePath)
    {
        ifstream inFile(csvFilePath, ios::in);
        string line;
        bool flag = false;
        unsigned int lineIndex = 0;
        while (getline(inFile, line))
        {
            ++lineIndex;
            if (line != "")
            {
                flag = true;
                break;
            }
        }
        if (!flag)
            return;
        vector<string> strBuffer;
        split(line, strBuffer);
        feature.dim = (unsigned int)strBuffer.size() - 1;
        feature.dataPtr.resize(feature.dim);
        label.dim = 1;
        label.dataPtr.resize(1);

        bool hasMapFile = false;
        if (mapFilePath != "")
        {
            if ((_access(mapFilePath.c_str(), 0)) != -1)
                loadMapFile(labelMap, mapFilePath);
            else
                throw exception(("Map file error: file not exist in " + mapFilePath + ".").c_str());
            hasMapFile = true;
        }

        if (featureAttributeFilePath != "")
        {
            if ((_access(featureAttributeFilePath.c_str(), 0)) != -1)
                loadFeatureAttributeFile(featureAttributeFilePath, feature);
            else
                throw exception(("Feature attribute error: file not exist in " + featureAttributeFilePath + ".").c_str());
        }
        else
        {
            feature.dataType.resize(feature.dim);
            for (DataType & type : feature.dataType)
                type = DataType::Real;
        }
        label.dataType.push_back(DataType::Unsigned32);

        if (csvHeader)
            strBuffer.clear();
        while (getline(inFile, line))
        {
            ++lineIndex;
            size_t elementNum = split(line, strBuffer);
            if (0 == elementNum)
                continue;
            if(elementNum - 1 != feature.dim)
                throw exception(("Csv data error: element num not match in line " + to_string(lineIndex) + ".").c_str());
        }

        feature.num = label.num = (unsigned int)strBuffer.size() / (feature.dim + 1);
        unsigned int index = 0;
        for (unsigned int i(0); i < feature.dim; ++i)
        {
            if (DataType::Real == feature.dataType[i])
                feature.dataPtr[i] = new Type[feature.num];
            else if (DataType::String == feature.dataType[i])
                feature.dataPtr[i] = new string[feature.num];
            else
                throw exception("Feature attribute error: feature attribute must be [Continuous | Discrete].");
        }
        label.dataPtr[0] = new unsigned int[label.num];
        unsigned int* labelPtr = (unsigned int*)label.dataPtr[0];
        for (unsigned int i(0); i < feature.num; ++i)
        {
            for (unsigned int j(0); j < feature.dim; ++j)
            {
                if (DataType::Real == feature.dataType[j])
                    ((Type*)(feature.dataPtr[j]))[i] = stod(strBuffer[index++]);
                else if (DataType::String == feature.dataType[j])
                    ((string*)(feature.dataPtr[j]))[i] = strBuffer[index++];
                else
                    throw exception("Feature attribute error: feature attribute must be [Continuous | Discrete].");
            }
            if (hasMapFile)
            {
                if (labelMap.find(strBuffer[index]) == labelMap.end())
                    throw exception(("Label map error: label " + strBuffer[index] + "not found in map file.").c_str());
            }
            else
            {
                if (labelMap.find(strBuffer[index]) == labelMap.end())
                    labelMap[strBuffer[index]] = stoi(strBuffer[index]);
            }
            labelPtr[i] = labelMap[strBuffer[index++]];
        }

        inFile.close();
    }

    template <class Type>
    void Data<Type>::release()
    {
        if (isRef)
            return;
        for (unsigned int i(0); i < dim; ++i)
        {
            if (dataPtr[i] != NULL)
            {
                if (DataType::Unsigned32 == dataType[i])
                    delete[] (unsigned int*)dataPtr[i];
                else if (DataType::Unsigned64 == dataType[i])
                    delete[] (unsigned long*)dataPtr[i];
                else if (DataType::Real == dataType[i])
                    delete[] (Type*)dataPtr[i];
                else if (DataType::String == dataType[i])
                    delete[] (string*)dataPtr[i];
                else
                    throw exception("Unexpected type error: data type must be [Unsigned32 | Unsigned64 | Real | String]");
                dataPtr[i] = NULL;
            }
        }
    }

    template class Data<float>;
    template class Data<double>;
}}