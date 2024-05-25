#ifndef BASECLASSIFIER_H__
#define BASECLASSIFIER_H__

#include <iostream>
#include "CountVectorizer.h"

using namespace std;

class BaseClassifier
{
public:  // protected
    BaseClassifier();
    ~BaseClassifier();
    CountVectorizer CV;
    void shape();
    void head();
    virtual void fit(string abs_filepath_to_features, string abs_filepath_to_labels) = 0;
    virtual void predict(string abs_filepath_to_features, string abs_filepath_to_labels) = 0;
    virtual int predict(string sentence) = 0;
    virtual void save(const std::string& filename) const = 0;
    virtual void load(const std::string& filename) = 0;
};

#endif // BASECLASSIFIER_H__