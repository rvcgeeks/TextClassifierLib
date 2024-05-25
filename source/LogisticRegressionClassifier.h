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
    void fit(string abs_filepath_to_features, string abs_filepath_to_labels) override;
    void predict(string abs_filepath_to_features, string abs_filepath_to_labels) override;
    int predict(string sentence) override;
    void save(const std::string& filename) const override;
    void load(const std::string& filename) override;

private:
    std::vector<float> weights;
    float bias;
    float learning_rate;
    int epochs;
    float sigmoid(float z) const;
    void update_weights(std::vector<int>& features, bool label);
};

#endif // LOGISTICREGRESSIONCLASSIFIER_H__
