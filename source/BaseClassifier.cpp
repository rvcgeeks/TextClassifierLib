/**
 * @file BaseClassifier.cpp
 * @brief Implementation of the BaseClassifier class.
 */

/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#include "BaseClassifier.h"

/**
 * @brief Constructor for BaseClassifier.
 */
BaseClassifier::BaseClassifier()
{
}

/**
 * @brief Destructor for BaseClassifier.
 */
BaseClassifier::~BaseClassifier()
{
}

/**
 * @brief Display the shape of the dataset.
 */
void BaseClassifier::shape()
{
    pVec->shape();
}

/**
 * @brief Display the head of the dataset.
 */
void BaseClassifier::head()
{
    pVec->head();
}

/**
 * @brief Set Model Version.
 */
void BaseClassifier::setVersionInfo(char* vers_info_in)
{
	pVec->setVersionInfo(vers_info_in);
}
