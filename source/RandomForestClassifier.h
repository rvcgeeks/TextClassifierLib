#ifndef RANDOMFORESTCLASSIFIER_H__
#define RANDOMFORESTCLASSIFIER_H__

#include <vector>
#include <memory>

#include "BaseClassifier.h"
#include "DecisionTree.h"

class RandomForestClassifier : public BaseClassifier
{
public:
    RandomForestClassifier(int vectorizerid = ID_VECTORIZER_COUNT, int num_trees = 50, int max_depth = 5);
    ~RandomForestClassifier();
    
    void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels);
    void predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels) override;
    Prediction predict(std::string sentence) override;
    void save(const std::string& filename) const override;
    void load(const std::string& filename) override;

private:
    int num_trees;
    int max_depth;
    std::vector<std::shared_ptr<DecisionTree>> trees;
};

#endif // RANDOMFORESTCLASSIFIER_H__
