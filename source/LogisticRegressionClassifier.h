#ifndef LOGISTICREGRESSIONCLASSIFIER_H__
#define LOGISTICREGRESSIONCLASSIFIER_H__

#include <vector>
#include <cmath>
#include "BaseClassifier.h"

class LogisticRegressionClassifier: public BaseClassifier
{
public:
    LogisticRegressionClassifier();
    ~LogisticRegressionClassifier();
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
    double sigmoid(double z) const;
    double predict_proba(const std::vector<int>& features) const;
};

#endif // LOGISTICREGRESSIONCLASSIFIER_H__
