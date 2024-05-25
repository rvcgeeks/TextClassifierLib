#ifndef LINEARSVCCLASSIFIER_H__
#define LINEARSVCCLASSIFIER_H__

#include <vector>
#include <random> 
#include "BaseClassifier.h"

class LinearSVCClassifier: public BaseClassifier
{
public:
    LinearSVCClassifier();
    ~LinearSVCClassifier();
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
    void update_weights(std::vector<int>& features, bool label);
};

#endif // LINEARSVCCLASSIFIER_H__
