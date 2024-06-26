#ifndef RANDOMFORESTCLASSIFIER_H__
#define RANDOMFORESTCLASSIFIER_H__

#include <vector>
#include <memory>

#include "BaseClassifier.h"
#include "DecisionTree.h"

/**
 * @file RandomForestClassifier.h
 * @brief Declaration of RandomForestClassifier class.
 */

/**
 * @brief Random forest classifier implementation.
 *
 * RandomForestClassifier is an ensemble learning method that constructs
 * a multitude of decision trees during training and outputs the class
 * that is the mode of the classes (classification) or mean prediction
 * (regression) of the individual trees.
 */
class RandomForestClassifier : public BaseClassifier
{
public:
    /**
     * @brief Constructor for RandomForestClassifier.
     * @param pvec Pointer to a BaseVectorizer object for feature extraction.
     */
    RandomForestClassifier(BaseVectorizer* pvec);
    ~RandomForestClassifier();
    
    /**
     * @brief Set hyperparameters for the random forest classifier.
     */
    void setHyperparameters(std::string hyperparameters) override;

    /**
     * @brief Fit the classifier to the training data.
     * @param abs_filepath_to_features Absolute file path to the file containing features.
     * @param abs_filepath_to_labels Absolute file path to the file containing labels.
     */
    void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels) override;

    /**
     * @brief Predict labels for test data.
     * @param abs_filepath_to_features Absolute file path to the file containing features.
     * @param abs_filepath_to_labels Absolute file path to the file to save predicted labels.
     */
    void predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels, bool preprocess = true) override;

    /**
     * @brief Predict label for a single input sentence.
     * @param sentence Input sentence to predict label for.
     * @return Prediction object containing predicted label.
     */
    Prediction predict(std::string sentence, bool preprocess = true) override;

    /**
     * @brief Save the model to a file.
     * @param filename Name of the file to save the model.
     */
    void save(const std::string& filename) const override;

    /**
     * @brief Load the model from a file.
     * @param filename Name of the file to load the model from.
     */
    void load(const std::string& filename) override;

private:
    int num_trees; /**< Number of decision trees in the random forest. */
    int max_depth; /**< Maximum depth of each decision tree. */
    std::vector<std::shared_ptr<DecisionTree>> trees; /**< Vector of decision trees in the random forest. */
};

#endif // RANDOMFORESTCLASSIFIER_H__
