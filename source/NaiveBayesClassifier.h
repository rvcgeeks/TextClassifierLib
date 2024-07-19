/**
 * @file NaiveBayesClassifier.h
 * @brief Declaration of NaiveBayesClassifier class.
 */

/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#ifndef NAIVEBAYESCLASSIFIER_H__
#define NAIVEBAYESCLASSIFIER_H__

#include <cmath>
#include <vector>
#include <unordered_map>
#include "BaseClassifier.h"

/**
 * @brief Naive Bayes classifier implementation.
 *
 * NaiveBayesClassifier is a probabilistic classifier based on Bayes' theorem.
 * It assumes independence between features given the class label. The probability
 * of a class given the features is calculated using the formula:
 * \f[ P(y|x) \propto P(y) \prod_{i=1}^{n} P(x_i|y) \f]
 * where \f$ P(y) \f$ is the prior probability of class \f$ y \f$,
 * and \f$ P(x_i|y) \f$ is the conditional probability of feature \f$ x_i \f$ given class \f$ y \f$.
 */
class NaiveBayesClassifier : public BaseClassifier
{
public:
    /**
     * @brief Constructor for NaiveBayesClassifier.
     * @param pvec Pointer to a BaseVectorizer object for feature extraction.
     */
    NaiveBayesClassifier(BaseVectorizer* pvec);
    ~NaiveBayesClassifier();

    /**
     * @brief Set hyperparameters for the naive Bayes classifier.
     * @param hyperparameters A string containing hyperparameters settings.
     *                         Format: "alpha_value"
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
    double smoothing_param_m; /**< Laplace smoothing parameter. */
    double smoothing_param_p; /**< Laplace smoothing parameter. */
    std::tr1::unordered_map<int, double> log_prob_pos; /**< Log probabilities for positive class. */
    std::tr1::unordered_map<int, double> log_prob_neg; /**< Log probabilities for negative class. */
    double log_prior_pos; /**< Log prior probability for positive class. */
    double log_prior_neg; /**< Log prior probability for negative class. */

    /**
     * @brief Calculate the log probability of features given the class label.
     * @param features Input features for prediction.
     * @param is_positive Whether the class label is positive.
     * @return Log probability.
     */
    double calculate_log_probability(const std::vector<double>& features, bool is_positive) const;
};

#endif // NAIVEBAYESCLASSIFIER_H__
