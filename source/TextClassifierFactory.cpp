
/*++

Revision History:
	Date:	Jun 28, 2024.
	Author:	Rajas Chavadekar.
	Desc:	Created.

--*/

#include "TextClassifierFactory.h"

TextClassifierFactory::TextClassifierFactory()
{
}

TextClassifierFactory::~TextClassifierFactory()
{
}

TextClassifierFactory::Product TextClassifierFactory::getTextClassifier(int vectorizer_id, int classifier_id)
{
	shared_ptr<BaseClassifier> pclsfr = nullptr;
	BaseVectorizer *pVec = NULL;

	switch (vectorizer_id)
    {
        case ID_VECTORIZER_COUNT:
            pVec = new CountVectorizer();
            break;

        case ID_VECTORIZER_TFIDF:
            pVec = new TfidfVectorizer();
            break;

        default:
            return nullptr;
    }

    pVec->setBinary(false);
    pVec->setCaseSensitive(false);
    pVec->setIncludeStopWords(false);

	switch (classifier_id) 
	{
		case ID_CLASSIFIER_NAIVEBAYESCLASSIFIER:
			pclsfr = make_shared<NaiveBayesClassifier>(pVec);
			break;

		case ID_CLASSIFIER_LOGISTICREGRESSIONCLASSIFIER:
			pclsfr = make_shared<LogisticRegressionClassifier>(pVec);
			break;

		case ID_CLASSIFIER_SVCCLASSIFIER:
			pclsfr = make_shared<SVCClassifier>(pVec);
			break;

		case ID_CLASSIFIER_KNNCLASSIFIER:
			pclsfr = make_shared<KNNClassifier>(pVec);
			break;

		case ID_CLASSIFIER_RANDOMFORESTCLASSIFIER:
			pclsfr = make_shared<RandomForestClassifier>(pVec);
			break;

		case ID_CLASSIFIER_GRADIENTBOOSTINGCLASSIFIER:
			pclsfr = make_shared<GradientBoostingClassifier>(pVec);
			break;

		default:
			delete pVec;
			return nullptr;
	}

	return pclsfr;
}
