#ifndef NAIVEBAYESCLASSIFIER_H__
#define NAIVEBAYESCLASSIFIER_H__

#include <cmath>
#include "BaseClassifier.h"

class NaiveBayesClassifier: public BaseClassifier
{
public:
    NaiveBayesClassifier();
    ~NaiveBayesClassifier();
    void fit(string abs_filepath_to_features, string abs_filepath_to_labels) override;
    void predict(string abs_filepath_to_features, string abs_filepath_to_labels) override;
    int predict(string sentence) override;
    void save(const std::string& filename) const override;
    void load(const std::string& filename) override;

private:
    int total_words_of_type_true;
    float logp_true;
    int total_words_of_type_false;
    float logp_false;
    float smoothing_param_m;
    float smoothing_param_p;
};

#endif // NAIVEBAYESCLASSIFIER_H__