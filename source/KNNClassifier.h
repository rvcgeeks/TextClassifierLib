
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#ifndef KNNCLASSIFIER_H__
#define KNNCLASSIFIER_H__

#include <vector>
#include <string>
#include "BaseClassifier.h"
#include "KDTree.h"

/**
 * @file KNNClassifier.h
 * @brief Declaration of KNNClassifier class.
 */

/**
 * @brief k-Nearest Neighbors classifier implementation.
 *
 * KNNClassifier is a machine learning classifier that classifies
 * data points based on the majority class among their k nearest neighbors.
 */
class KNNClassifier : public BaseClassifier
{
public:
    /**
     * @brief Constructor for KNNClassifier.
     * @param pvec Pointer to a BaseVectorizer object for feature extraction.
     */
    KNNClassifier(BaseVectorizer* pvec);
    ~KNNClassifier();

    /**
     * @brief Set hyperparameters for the k-Nearest Neighbors classifier.
     * @param hyperparameters A string containing hyperparameters settings.
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
    int k; /**< Number of nearest neighbors to consider. */
    std::vector<std::vector<double>> training_features; /**< Training features. */
    std::vector<int> training_labels; /**< Training labels. */
    KDTree kd_tree; /**< KDTree for efficient nearest neighbor search. */

    /**
     * @brief Get the label for a given set of features using k-Nearest Neighbors.
     * @param features Input features for prediction.
     * @return Predicted label.
     */
    int getLabel(const std::vector<double>& features) const;

    /**
     * @brief Calculate the Euclidean distance between two feature vectors.
     * @param a First feature vector.
     * @param b Second feature vector.
     * @return Euclidean distance between the feature vectors.
     */
    double calculateDistance(const std::vector<double>& a, const std::vector<double>& b) const;
};

#endif // KNNCLASSIFIER_H__
