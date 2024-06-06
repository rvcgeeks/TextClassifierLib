
#include <iostream>

#include "../source/NaiveBayesClassifier.h"
#include "../source/LogisticRegressionClassifier.h"
#include "../source/SVCClassifier.h"
#include "../source/KNNClassifier.h"
#include "../source/RandomForestClassifier.h"
#include "../source/GradientBoostingClassifier.h"

using namespace std;

int main(int argc, char **argv)
{
	int vectorizerid;

	BaseClassifier* pclsfr;

	if (argc < 6)
	{
		cout << "Usage: " << endl
			 << "  " << argv[0] << " f (vectorizer id) (classifier id) my_model.bin features.txt labels.txt" << endl
			 << "  " << argv[0] << " p (vectorizer id) (classifier id) my_model.bin features.txt labels_pred.txt" << endl
		     << "  " << argv[0] << " 1 (vectorizer id) (classifier id) my_model.bin \"This is string to classify\" " << endl
			 << "\nwhere vectorizer id = " << endl
			 << "  1 CounterVectorizer\n  2 TfidfVectorizer" << endl
		     << "\nwhere classifier id = " << endl
			 << "  1 NaiveBayesClassifier\n  2 LogisticRegressionClassifier\n  3 SVCClassifier\n  4 KNNClassifier\n  5 RandomForestClassifier\n  6 GradientBoostingClassifier" << endl;
		return 1;
	}
	
	switch (atoi(argv[2]))
	{
		case ID_VECTORIZER_COUNT:
			vectorizerid = ID_VECTORIZER_COUNT;
			break;

		case ID_VECTORIZER_TFIDF:
			vectorizerid = ID_VECTORIZER_TFIDF;
			break;

		default:
			cerr << "Invalid Vectorizer!" << endl;
			return 1;
	}

	switch (atoi(argv[3])) 
	{
		case ID_CLASSIFIER_NAIVEBAYESCLASSIFIER:
			pclsfr = new NaiveBayesClassifier(vectorizerid);
			break;

		case ID_CLASSIFIER_LOGISTICREGRESSIONCLASSIFIER:
			pclsfr = new LogisticRegressionClassifier(vectorizerid);
			break;

		case ID_CLASSIFIER_SVCCLASSIFIER:
			pclsfr = new SVCClassifier(vectorizerid);
			break;

		case ID_CLASSIFIER_KNNCLASSIFIER:
			pclsfr = new KNNClassifier(vectorizerid);
			break;

		case ID_CLASSIFIER_RANDOMFORESTCLASSIFIER:
			pclsfr = new RandomForestClassifier(vectorizerid);
			break;

		case ID_CLASSIFIER_GRADIENTBOOSTINGCLASSIFIER:
			pclsfr = new GradientBoostingClassifier(vectorizerid);
			break;

		default:
			cerr << "Invalid Classifier!" << endl;
			return 1;
	}
	
	// txtclsfr f 2 my_model.bin features.txt labels.txt
	if(argv[1][0] == 'f') {
		cout << "Training\n";
		pclsfr->fit(argv[5], argv[6]);
		pclsfr->shape();
		pclsfr->save(argv[4]);
		cout << "Model Saved" << endl;

	// txtclsfr p 2 my_model.bin features.txt labels_pred.txt
	} else if(argv[1][0] == 'p') {
		cout << "Predicting\n";
		pclsfr->load(argv[4]);
		cout << "Model Loaded" << endl;
		pclsfr->predict(argv[5], argv[6]);

	// txtclsfr 1 2 my_model.bin "This is string to classify" 
	} else if(argv[1][0] == '1') {
		Prediction result;
		pclsfr->load(argv[4]);
		cout << "Model Loaded" << endl;
		result = pclsfr->predict(argv[5]);
		cout << result.label << "    "<< result.probability << endl;
	}

	delete pclsfr;

	cout << "Done\n";
    return 0;
}
