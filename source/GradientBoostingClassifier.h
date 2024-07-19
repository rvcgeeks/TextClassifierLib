
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#ifndef GRADIENTBOOSTINGCLASSIFIER_H__
#define GRADIENTBOOSTINGCLASSIFIER_H__

#include <vector>
#include <memory>
#include <string>

#include "BaseClassifier.h"
#include "DecisionTree.h"

/**
 * @file GradientBoostingClassifier.h
 * @brief Declaration of GradientBoostingClassifier class.
 */

/**
 * @brief Gradient boosting classifier implementation.
 *
 * GradientBoostingClassifier is a machine learning classifier that utilizes
 * an ensemble of decision trees for classification tasks. It builds a strong
 * learner by sequentially adding weak learners (decision trees) and fitting
 * them to the residual errors of the previous predictions.
 */
class GradientBoostingClassifier : public BaseClassifier
{
public:
    /**
     * @brief Constructor for GradientBoostingClassifier.
     * @param pvec Pointer to a BaseVectorizer object for feature extraction.
     */
    GradientBoostingClassifier(BaseVectorizer* pvec);
    ~GradientBoostingClassifier();
    
    /**
     * @brief Set hyperparameters for the gradient boosting classifier.
     * @param hyperparameters A string containing hyperparameters settings.
     */
    void setHyperparameters(std::string hyperparameters);

    /**
     * @brief Fit the classifier to the training data.
     * @param abs_filepath_to_features Absolute file path to the file containing features.
     * @param abs_filepath_to_labels Absolute file path to the file containing labels.
     */
    void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels);

    /**
     * @brief Predict labels for test data.
     * @param abs_filepath_to_features Absolute file path to the file containing features.
     * @param abs_filepath_to_labels Absolute file path to the file to save predicted labels.
     */
    void predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels, bool preprocess = true);

    /**
     * @brief Predict label for a single input sentence.
     * @param sentence Input sentence to predict label for.
     * @return Prediction object containing predicted label.
     */
    Prediction predict(std::string sentence, bool preprocess = true);

    /**
     * @brief Save the model to a file.
     * @param filename Name of the file to save the model.
     */
    void save(const std::string& filename) const;

    /**
     * @brief Load the model from a file.
     * @param filename Name of the file to load the model from.
     */
    void load(const std::string& filename);

private:
    std::vector<std::tr1::shared_ptr<DecisionTree> > trees; /**< Vector of decision trees. */
    int n_trees; /**< Number of trees in the ensemble. */
    int max_depth; /**< Maximum depth of each decision tree. */
    double learning_rate; /**< Learning rate for gradient boosting. */

    /**
     * @brief Predict using a single decision tree.
     * @param tree Decision tree to make predictions.
     * @param features Input features for prediction.
     * @return Predicted value.
     */
    double predict_tree(const DecisionTree& tree, const std::vector<double>& features) const;

    /**
     * @brief Predict class probabilities for input features.
     * @param features Input features for prediction.
     * @return Predicted class probabilities.
     */
    double predict_proba(const std::vector<double>& features) const;
};

#endif // GRADIENTBOOSTINGCLASSIFIER_H__
