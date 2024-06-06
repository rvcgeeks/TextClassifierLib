#ifndef GRADIENTBOOSTINGCLASSIFIER_H__
#define GRADIENTBOOSTINGCLASSIFIER_H__

#include <vector>
#include <memory>

#include "BaseClassifier.h"
#include "DecisionTree.h"

class GradientBoostingClassifier : public BaseClassifier
{
public:
    GradientBoostingClassifier(int vectorizerid=ID_VECTORIZER_COUNT);
    ~GradientBoostingClassifier();
    
    void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels) override;
    void predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels) override;
    Prediction predict(std::string sentence) override;
    void save(const std::string& filename) const override;
    void load(const std::string& filename) override;

private:
    std::vector<std::unique_ptr<DecisionTree>> trees;
    std::vector<double> tree_weights;
    int n_trees;
    int max_depth;
    double learning_rate;

    double predict_tree(const DecisionTree& tree, const std::vector<double>& features) const;
    double predict_proba(const std::vector<double>& features) const;
};

#endif // GRADIENTBOOSTINGCLASSIFIER_H__
