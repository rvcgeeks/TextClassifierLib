
/**
 * @file LogisticRegressionClassifier.h
 * @brief Declaration of LogisticRegressionClassifier class.
 */

/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#ifndef LOGISTICREGRESSIONCLASSIFIER_H__
#define LOGISTICREGRESSIONCLASSIFIER_H__

#include <vector>
#include <cmath>
#include "BaseClassifier.h"

/**
 * @brief Logistic regression classifier implementation.
 *
 * LogisticRegressionClassifier is a machine learning classifier that models
 * the probability of a binary outcome using the logistic function:
 * \f[ P(y=1|x) = \frac{1}{1 + e^{-(w \cdot x + b)}} \f]
 * where \f$ w \f$ are the weights, \f$ b \f$ is the bias term, and \f$ x \f$ is the input feature vector.
 */
class LogisticRegressionClassifier: public BaseClassifier
{
public:
    /**
     * @brief Constructor for LogisticRegressionClassifier.
     * @param pvec Pointer to a BaseVectorizer object for feature extraction.
     */
    LogisticRegressionClassifier(BaseVectorizer* pvec);
    ~LogisticRegressionClassifier();

    /**
     * @brief Set hyperparameters for the logistic regression classifier.
     * @param hyperparameters A string containing hyperparameters settings.
     *                         Format: "epochs:learning_rate:l1_reg:l2_reg"
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
    std::vector<double> weights; /**< Coefficients for features. */
    double bias; /**< Bias term. */
    int epochs; /**< Number of training epochs. */
    double learning_rate; /**< Learning rate for gradient descent. */
    double l1_regularization_param; /**< L1 regularization parameter. */
    double l2_regularization_param; /**< L2 regularization parameter. */

    /**
     * @brief Predict class probability for input features using logistic function.
     * @param features Input features for prediction.
     * @return Predicted class probability.
     */
    double predict_proba(const std::vector<double>& features) const;
};

#endif // LOGISTICREGRESSIONCLASSIFIER_H__
