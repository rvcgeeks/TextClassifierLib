
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#ifndef LINEARSVCCLASSIFIER_H__
#define LINEARSVCCLASSIFIER_H__

#include <vector>
#include <random> 
#include "BaseClassifier.h"

/**
 * @file LinearSVCClassifier.h
 * @brief Declaration of the Linear Support Vector Classifier (SVC) class.
 */

/**
 * @class SVCClassifier
 * @brief Implements a linear Support Vector Classifier (SVC).
 *
 * This class inherits from BaseClassifier and provides functionality for training
 * and using a linear SVC for text classification tasks.
 */
class SVCClassifier: public BaseClassifier
{
public:
    /**
     * @brief Constructor for SVCClassifier.
     * @param pvec Pointer to the BaseVectorizer object to be used for vectorization.
     */
    SVCClassifier(BaseVectorizer* pvec);
    
    /**
     * @brief Destructor for SVCClassifier.
     */
    ~SVCClassifier();
    
    /**
     * @brief Set hyperparameters for the SVC model.
     * @param hyperparameters String containing hyperparameters.
     */
    void setHyperparameters(std::string hyperparameters) override;
    
    /**
     * @brief Train the SVC model.
     * @param abs_filepath_to_features Absolute file path to the file containing features.
     * @param abs_filepath_to_labels Absolute file path to the file containing labels.
     */
    void fit(std::string abs_filepath_to_features, std::string abs_filepath_to_labels) override;
    
    /**
     * @brief Predict labels for features in a file.
     * @param abs_filepath_to_features Absolute file path to the file containing features.
     * @param abs_filepath_to_labels Absolute file path to the file to store predicted labels.
     */
    void predict(std::string abs_filepath_to_features, std::string abs_filepath_to_labels, bool preprocess = true) override;
    
    /**
     * @brief Predict label for a single sentence.
     * @param sentence Input sentence to predict label for.
     * @return Prediction object containing the predicted label.
     */
    Prediction predict(std::string sentence, bool preprocess = true) override;
    
    /**
     * @brief Save the trained model to a file.
     * @param filename Name of the file to save the model to.
     */
    void save(const std::string& filename) const override;
    
    /**
     * @brief Load a trained model from a file.
     * @param filename Name of the file to load the model from.
     */
    void load(const std::string& filename) override;

private:
    std::vector<double> weights; /**< Model weights. */
    double bias; /**< Model bias. */
    int epochs; /**< Number of epochs for training. */
    double learning_rate; /**< Learning rate for training. */
    double l1_regularization_param; /**< L1 regularization parameter. */
    double l2_regularization_param; /**< L2 regularization parameter. */

    /**
     * @brief Compute the margin for prediction.
     * @param features Vector of features for prediction.
     * @return Margin value for prediction.
     */
    double predict_margin(const std::vector<double>& features) const;
};

#endif // LINEARSVCCLASSIFIER_H__
