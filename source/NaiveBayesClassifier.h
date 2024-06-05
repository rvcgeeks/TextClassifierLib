#ifndef NAIVEBAYESCLASSIFIER_H__
#define NAIVEBAYESCLASSIFIER_H__

#include <cmath>
#include <vector>
#include <unordered_map>
#include "BaseClassifier.h"

class NaiveBayesClassifier : public BaseClassifier
{
public:
    NaiveBayesClassifier();
    ~NaiveBayesClassifier();
    void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels);
    void predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels) override;
    Prediction predict(std::string sentence) override;
    void save(const std::string& filename) const override;
    void load(const std::string& filename) override;

private:
    std::unordered_map<int, double> log_prob_pos;
    std::unordered_map<int, double> log_prob_neg;
    double log_prior_pos;
    double log_prior_neg;

    double calculate_log_probability(const std::vector<int>& features, bool is_positive) const;
};

#endif // NAIVEBAYESCLASSIFIER_H__
