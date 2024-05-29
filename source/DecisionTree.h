#ifndef DECISIONTREE_H__
#define DECISIONTREE_H__

#include <vector>
#include <memory>
#include <string>
#include <fstream>

#include "CountVectorizer.h"

class DecisionTree
{
public:
    DecisionTree(int max_depth = 10);
    ~DecisionTree();

    void fit(const CountVectorizer& CV, const std::vector<std::shared_ptr<Sentence>>& sentences);
    int predict(const std::vector<int>& features) const;
    void save(std::ofstream& outFile) const;
    void load(std::ifstream& inFile);

private:
    struct Node
    {
        int feature_index;
        int label;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;

        Node(int feature_index = -1, int label = -1) : feature_index(feature_index), label(label) {}
    };

    std::shared_ptr<Node> root;
    int max_depth;

    std::shared_ptr<Node> buildTree(const std::vector<std::shared_ptr<Sentence>>& sentences, int depth);
    int majorityClass(const std::vector<std::shared_ptr<Sentence>>& sentences) const;
    double giniIndex(const std::vector<std::shared_ptr<Sentence>>& left, const std::vector<std::shared_ptr<Sentence>>& right) const;
    void split(const std::vector<std::shared_ptr<Sentence>>& sentences, int feature_index, std::vector<std::shared_ptr<Sentence>>& left, std::vector<std::shared_ptr<Sentence>>& right) const;
    int predictNode(const std::shared_ptr<Node>& node, const std::vector<int>& features) const;
    void saveNode(std::ofstream& outFile, const std::shared_ptr<Node>& node) const;
    std::shared_ptr<Node> loadNode(std::ifstream& inFile);
};

#endif // DECISIONTREE_H__
