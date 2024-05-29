#ifndef KNNCLASSIFIER_H__
#define KNNCLASSIFIER_H__

#include <vector>
#include <string>
#include "BaseClassifier.h"
#include "KDTree.h"

class KNNClassifier : public BaseClassifier
{
public:
    KNNClassifier(int k = 3);
    ~KNNClassifier();
    void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels);
    void predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels) override;
    int predict(std::string sentence) override;
    void save(const std::string& filename) const override;
    void load(const std::string& filename) override;

private:
    int k;
    std::vector<std::vector<int>> training_features;
    std::vector<int> training_labels;
    KDTree kd_tree;

    int getLabel(const std::vector<int>& features) const;
    double calculateDistance(const std::vector<int>& a, const std::vector<int>& b) const;
};

#endif // KNNCLASSIFIER_H__
