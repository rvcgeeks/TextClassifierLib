/**
 * @file DecisionTree.h
 * @brief Header file for the DecisionTree class.
 */

#ifndef DECISIONTREE_H__
#define DECISIONTREE_H__

#include <vector>
#include <memory>
#include <string>
#include <fstream>

#include "BaseClassifier.h"
#include "BaseVectorizer.h"

/**
 * @class DecisionTree
 * @brief Implementation of a Decision Tree classifier.
 *
 * This class provides functionality to build and use a Decision Tree classifier
 * for classification tasks.
 */
class DecisionTree
{
public:
    /**
     * @brief Constructor.
     *
     * @param max_depth Maximum depth of the decision tree.
     */
    DecisionTree(int max_depth = 10);

    /**
     * @brief Destructor.
     */
    ~DecisionTree();

    /**
     * @brief Fit the decision tree on the provided dataset.
     *
     * @param sentences Vector of shared pointers to Sentence objects.
     */
    void fit(const std::vector<std::shared_ptr<Sentence>>& sentences);

    /**
     * @brief Predict the class label for the given features.
     *
     * @param features Vector of feature values.
     * @return Prediction object containing the predicted label and probability.
     */
    Prediction predict(const std::vector<double>& features) const;

    /**
     * @brief Save the decision tree model to a file.
     *
     * @param outFile Output file stream to save the model.
     */
    void save(std::ofstream& outFile) const;

    /**
     * @brief Load the decision tree model from a file.
     *
     * @param inFile Input file stream to load the model.
     */
    void load(std::ifstream& inFile);

private:
    /**
     * @struct Node
     * @brief Structure representing a node in the decision tree.
     */
    struct Node
    {
        int feature_index; /**< Index of the feature used for splitting. */
        int label; /**< Predicted label for the node. */
        int total_samples; /**< Total number of samples in the node. */
        int pos_samples; /**< Number of positive samples in the node. */
        std::shared_ptr<Node> left; /**< Pointer to the left child node. */
        std::shared_ptr<Node> right; /**< Pointer to the right child node. */

        /**
         * @brief Constructor for Node struct.
         *
         * @param feature_index Index of the feature used for splitting.
         * @param label Predicted label for the node.
         * @param total_samples Total number of samples in the node.
         * @param pos_samples Number of positive samples in the node.
         */
        Node(int feature_index = -1, int label = -1, int total_samples = 0, int pos_samples = 0)
            : feature_index(feature_index), label(label), total_samples(total_samples), pos_samples(pos_samples) {}
    };

    std::shared_ptr<Node> root; /**< Pointer to the root node of the decision tree. */
    int max_depth; /**< Maximum depth of the decision tree. */

    /**
     * @brief Build the decision tree recursively.
     *
     * @param sentences Vector of shared pointers to Sentence objects.
     * @param depth Current depth of the tree.
     * @return Pointer to the root node of the built tree.
     */
    std::shared_ptr<Node> buildTree(const std::vector<std::shared_ptr<Sentence>>& sentences, int depth);

    /**
     * @brief Determine the majority class in the dataset.
     *
     * @param sentences Vector of shared pointers to Sentence objects.
     * @param total_samples Total number of samples.
     * @param pos_samples Number of positive samples.
     * @return Majority class label.
     */
    int majorityClass(const std::vector<std::shared_ptr<Sentence>>& sentences, int& total_samples, int& pos_samples) const;

    /**
     * @brief Calculate the Gini index for a split.
     *
     * @param left Vector of shared pointers to Sentence objects on the left side of the split.
     * @param right Vector of shared pointers to Sentence objects on the right side of the split.
     * @return Gini index value.
     */
    double giniIndex(const std::vector<std::shared_ptr<Sentence>>& left, const std::vector<std::shared_ptr<Sentence>>& right) const;

    /**
     * @brief Split the dataset based on a feature.
     *
     * @param sentences Vector of shared pointers to Sentence objects.
     * @param feature_index Index of the feature to split on.
     * @param left Vector to store the samples on the left side of the split.
     * @param right Vector to store the samples on the right side of the split.
     */
    void split(const std::vector<std::shared_ptr<Sentence>>& sentences, int feature_index, std::vector<std::shared_ptr<Sentence>>& left, std::vector<std::shared_ptr<Sentence>>& right) const;

    /**
     * @brief Predict the class label for a given set of features at a node.
     *
     *  * @param node Pointer to the current node in the decision tree.
     * @param features Vector of feature values.
     * @return Prediction object containing the predicted label and probability.
     */
    Prediction predictNode(const std::shared_ptr<Node>& node, const std::vector<double>& features) const;

    /**
     * @brief Save a node of the decision tree to a file recursively.
     *
     * @param outFile Output file stream to save the node.
     * @param node Pointer to the current node.
     */
    void saveNode(std::ofstream& outFile, const std::shared_ptr<Node>& node) const;

    /**
     * @brief Load a node of the decision tree from a file recursively.
     *
     * @param inFile Input file stream to load the node.
     * @return Pointer to the loaded node.
     */
    std::shared_ptr<Node> loadNode(std::ifstream& inFile);
};

#endif // DECISIONTREE_H__
