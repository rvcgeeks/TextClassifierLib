
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#include "DecisionTree.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>

DecisionTree::DecisionTree(int max_depth)
    : max_depth(max_depth), root(nullptr)
{
}

DecisionTree::~DecisionTree()
{
}

void DecisionTree::fit(const std::vector<std::shared_ptr<Sentence>>& sentences)
{
    root = buildTree(sentences, 0);
}

Prediction DecisionTree::predict(const std::vector<double>& features) const
{
    return predictNode(root, features);
}

void DecisionTree::save(std::ofstream& outFile) const
{
    saveNode(outFile, root);
}

void DecisionTree::load(std::ifstream& inFile)
{
    root = loadNode(inFile);
}

std::shared_ptr<DecisionTree::Node> DecisionTree::buildTree(const std::vector<std::shared_ptr<Sentence>>& sentences, int depth)
{
    int total_samples, pos_samples;
    int majority_class = majorityClass(sentences, total_samples, pos_samples);

    if (sentences.empty() || depth >= max_depth)
    {
        return std::make_shared<Node>(-1, majority_class, total_samples, pos_samples);
    }

    double best_gini = 1.0;
    int best_feature = -1;
    std::vector<std::shared_ptr<Sentence>> best_left, best_right;

    size_t num_features = sentences[0]->sentence_map.size();
    for (size_t i = 0; i < num_features; ++i)
    {
        std::vector<std::shared_ptr<Sentence>> left, right;
        split(sentences, i, left, right);

        if (left.empty() || right.empty())
        {
            continue;
        }

        double gini = giniIndex(left, right);
        if (gini < best_gini)
        {
            best_gini = gini;
            best_feature = i;
            best_left = left;
            best_right = right;
        }
    }

    if (best_feature == -1)
    {
        return std::make_shared<Node>(-1, majority_class, total_samples, pos_samples);
    }

    auto node = std::make_shared<Node>(best_feature, -1, total_samples, pos_samples);
    node->left = buildTree(best_left, depth + 1);
    node->right = buildTree(best_right, depth + 1);

    return node;
}

int DecisionTree::majorityClass(const std::vector<std::shared_ptr<Sentence>>& sentences, int& total_samples, int& pos_samples) const
{
    pos_samples = std::count_if(sentences.begin(), sentences.end(), [](const std::shared_ptr<Sentence>& s) { return s->label == 1; });
    total_samples = sentences.size();
    int neg_count = total_samples - pos_samples;
    return pos_samples > neg_count ? 1 : 0;
}

double DecisionTree::giniIndex(const std::vector<std::shared_ptr<Sentence>>& left, const std::vector<std::shared_ptr<Sentence>>& right) const
{
    auto gini = [](const std::vector<std::shared_ptr<Sentence>>& group) {
        if (group.empty()) return 0.0;
        int pos_count = std::count_if(group.begin(), group.end(), [](const std::shared_ptr<Sentence>& s) { return s->label == 1; });
        int neg_count = group.size() - pos_count;
        double p1 = static_cast<double>(pos_count) / group.size();
        double p2 = static_cast<double>(neg_count) / group.size();
        return 1.0 - p1 * p1 - p2 * p2;
    };

    double total_size = left.size() + right.size();
    return (left.size() / total_size) * gini(left) + (right.size() / total_size) * gini(right);
}

void DecisionTree::split(const std::vector<std::shared_ptr<Sentence>>& sentences, int feature_index, std::vector<std::shared_ptr<Sentence>>& left, std::vector<std::shared_ptr<Sentence>>& right) const
{
    for (const auto& sentence : sentences)
    {
        if (sentence->sentence_map.count(feature_index))
        {
            left.push_back(sentence);
        }
        else
        {
            right.push_back(sentence);
        }
    }
}

Prediction DecisionTree::predictNode(const std::shared_ptr<Node>& node, const std::vector<double>& features) const
{
    if (!node)
    {
        throw std::runtime_error("Node is null");
    }
    if (node->feature_index == -1)
    {
        double probability = static_cast<double>(node->pos_samples) / node->total_samples;
        return { node->label, probability };
    }

    if (features[node->feature_index] > 0)
    {
        return predictNode(node->left, features);
    }
    else
    {
        return predictNode(node->right, features);
    }
}

void DecisionTree::saveNode(std::ofstream& outFile, const std::shared_ptr<Node>& node) const
{
    char null_flag;
    if (!node)
    {
        null_flag = 1;
        int dummy = 0;
        outFile.write(reinterpret_cast<const char*>(&null_flag), sizeof(null_flag));
        outFile.write(reinterpret_cast<const char*>(&dummy), sizeof(dummy));
        outFile.write(reinterpret_cast<const char*>(&dummy), sizeof(dummy));
        return;
    }
    null_flag = 0;
    outFile.write(reinterpret_cast<const char*>(&null_flag), sizeof(null_flag));
    outFile.write(reinterpret_cast<const char*>(&node->feature_index), sizeof(node->feature_index));
    outFile.write(reinterpret_cast<const char*>(&node->label), sizeof(node->label));
    outFile.write(reinterpret_cast<const char*>(&node->total_samples), sizeof(node->total_samples));
    outFile.write(reinterpret_cast<const char*>(&node->pos_samples), sizeof(node->pos_samples));
    saveNode(outFile, node->left);
    saveNode(outFile, node->right);
}

std::shared_ptr<DecisionTree::Node> DecisionTree::loadNode(std::ifstream& inFile)
{
    char null_flag;
    int feature_index;
    int label;
    int total_samples;
    int pos_samples;
    int dummy;

    inFile.read(reinterpret_cast<char*>(&null_flag), sizeof(null_flag));
    inFile.read(reinterpret_cast<char*>(&feature_index), sizeof(feature_index));
    inFile.read(reinterpret_cast<char*>(&label), sizeof(label));
    if (null_flag == 1)
    {
        return std::shared_ptr<Node>(nullptr);
    }
    inFile.read(reinterpret_cast<char*>(&total_samples), sizeof(total_samples));
    inFile.read(reinterpret_cast<char*>(&pos_samples), sizeof(pos_samples));

    auto node = std::make_shared<Node>(feature_index, label, total_samples, pos_samples);
    node->left = loadNode(inFile);
    node->right = loadNode(inFile);
    return node;
}
