#ifndef DECISIONTREE_H__
#define DECISIONTREE_H__

#include <vector>
#include <memory>
#include <string>
#include <fstream>

#include "BaseClassifier.h"
#include "BaseVectorizer.h"

class DecisionTree
{
public:
    DecisionTree(int max_depth = 10);
    ~DecisionTree();

    void fit(const std::vector<std::shared_ptr<Sentence>>& sentences);
    Prediction predict(const std::vector<double>& features) const;
    void save(std::ofstream& outFile) const;
    void load(std::ifstream& inFile);

private:
    struct Node
    {
        int feature_index;
        int label;
        int total_samples;
        int pos_samples;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;

        Node(int feature_index = -1, int label = -1, int total_samples = 0, int pos_samples = 0)
            : feature_index(feature_index), label(label), total_samples(total_samples), pos_samples(pos_samples) {}
    };

    std::shared_ptr<Node> root;
    int max_depth;

    std::shared_ptr<Node> buildTree(const std::vector<std::shared_ptr<Sentence>>& sentences, int depth);
    int majorityClass(const std::vector<std::shared_ptr<Sentence>>& sentences, int& total_samples, int& pos_samples) const;
    double giniIndex(const std::vector<std::shared_ptr<Sentence>>& left, const std::vector<std::shared_ptr<Sentence>>& right) const;
    void split(const std::vector<std::shared_ptr<Sentence>>& sentences, int feature_index, std::vector<std::shared_ptr<Sentence>>& left, std::vector<std::shared_ptr<Sentence>>& right) const;
    Prediction predictNode(const std::shared_ptr<Node>& node, const std::vector<double>& features) const;
    void saveNode(std::ofstream& outFile, const std::shared_ptr<Node>& node) const;
    std::shared_ptr<Node> loadNode(std::ifstream& inFile);
};

#endif // DECISIONTREE_H__
