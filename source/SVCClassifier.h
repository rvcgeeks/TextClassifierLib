#ifndef LINEARSVCCLASSIFIER_H__
#define LINEARSVCCLASSIFIER_H__

#include <vector>
#include <random> 
#include "BaseClassifier.h"

class SVCClassifier: public BaseClassifier
{
public:
    SVCClassifier();
    ~SVCClassifier();
    void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels);
    void predict(string abs_filepath_to_features, string abs_filepath_to_labels) override;
    Prediction predict(string sentence) override;
    void save(const std::string& filename) const override;
    void load(const std::string& filename) override;

private:
    std::vector<double> weights;
    double bias;
    int epochs;
    double learning_rate;
    double regularization_param;
    double predict_margin(const std::vector<int>& features) const;
};

#endif // LINEARSVCCLASSIFIER_H__
