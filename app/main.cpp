
#include <iostream>

#include "../source/TextClassifierFactory.h"

using namespace std;

int main(int argc, char **argv)
{
	int vectorizer_id, classifier_id;

	TextClassifierFactory clsfrFactoryObj;
	TextClassifierFactory::Product pclsfr;

	if (argc < 6)
	{
		cout << "Usage: " << endl
			 << "  " << argv[0] << " f (vectorizer id) (classifier id) my_model.bin features.txt labels.txt \"hyperparam1=val1,hyperparam2=val2,...\"" << endl
			 << "  " << argv[0] << " p (vectorizer id) (classifier id) my_model.bin features.txt labels_pred.txt" << endl
		     << "  " << argv[0] << " 1 (vectorizer id) (classifier id) my_model.bin \"This is string to classify\" " << endl
			 << "\nwhere vectorizer id = " << endl
			 << "  1 CounterVectorizer\n  2 TfidfVectorizer" << endl
		     << "\nwhere classifier id = " << endl
			 << "  1 NaiveBayesClassifier\n  2 LogisticRegressionClassifier\n  3 SVCClassifier\n  4 KNNClassifier\n  5 RandomForestClassifier\n  6 GradientBoostingClassifier" << endl;
		return 1;
	}

	vectorizer_id = atoi(argv[2]);
	classifier_id = atoi(argv[3]);
	
	pclsfr = clsfrFactoryObj.getTextClassifier(vectorizer_id, classifier_id);
	if(nullptr == pclsfr)
	{
		cerr << "Invalid vectorizer id or classifier id!" << endl;
		return 1;
	}
	
	// txtclsfr f 2 my_model.bin features.txt labels.txt
	if(argv[1][0] == 'f') {
		cout << "Training\n";
		if (argc == 8) {
			pclsfr->setHyperparameters(string(argv[7]));
		}
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

	cout << "Done\n";
    return 0;
}
