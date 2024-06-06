
#ifndef BASECLASSIFIER_H__
#define BASECLASSIFIER_H__

#include <iostream>

#include "CountVectorizer.h"
#include "TfidfVectorizer.h"

using namespace std;

#define ID_CLASSIFIER_NAIVEBAYESCLASSIFIER              1
#define ID_CLASSIFIER_LOGISTICREGRESSIONCLASSIFIER      2
#define ID_CLASSIFIER_SVCCLASSIFIER                     3
#define ID_CLASSIFIER_KNNCLASSIFIER                     4
#define ID_CLASSIFIER_RANDOMFORESTCLASSIFIER            5
#define ID_CLASSIFIER_GRADIENTBOOSTINGCLASSIFIER        6

struct Prediction
{
    int label;
    double probability;
};

class BaseClassifier
{
public:  // protected
    BaseClassifier();
    ~BaseClassifier();
    BaseVectorizer* pVec;
    CountVectorizer countVectorizerObj;
    TfidfVectorizer tfidfVectorizerObj;
    void shape();
    void head();
    virtual void fit(string abs_filepath_to_features, string abs_filepath_to_labels) = 0;
    virtual void predict(string abs_filepath_to_features, string abs_filepath_to_labels) = 0;
    virtual Prediction predict(string sentence) = 0;
    virtual void save(const std::string& filename) const = 0;
    virtual void load(const std::string& filename) = 0;
};

#endif // BASECLASSIFIER_H__