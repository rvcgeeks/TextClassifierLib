#include "CountVectorizer.h"

using namespace std;

// ======================CONSTRUCTORS======================= |


CountVectorizer::CountVectorizer(bool case_sensitive_){
    case_sensitive = case_sensitive_;
}

CountVectorizer::~CountVectorizer() {
}

// ======================ANALYSIS FUNCTIONS======================= |

int CountVectorizer::analyze(string sentence) {
    MyGlobalVars vars;
    vector<string> processed_input;
    float weight = 0;
    processed_input = buildSentenceVector(sentence);
    weight = getWeight(processed_input);
    if (weight == -1) {
        cout << "Sorry, not enough data for this input." <<endl;
        return vars.UNK;
    }
    else if (weight < .5) {
        cout << "This sentence has a negative sentiment." << endl;
        return vars.NEG;
    }
    else if (weight > .5) {
        cout << "This sentence has a positive sentiment." << endl;
        return vars.POS;
    }
    else {
        cout << "This sentence has a neutral sentiment." << endl;
        return vars.NEU;
    }
}


// ======================ANALYSIS UTILITIES======================= |


void CountVectorizer::train() {
    ifstream in;
    
    in.open("/app/trainingdata.txt");

    if(!in) {
    cout << "Cannot open input file.\n";
    return;
    }

    string str;
    while (getline(in, str)) {
        std::cout << str << std::endl;
    }
    in.close();
}


float CountVectorizer::getWeight(vector<string> sentence_) {
    float count = 0;
    float num = 0;
    float sum = 0;
    vector<float> word_weights;
    unsigned int word_array_length = word_array.size();
    for (string queryWord:sentence_) {
        for (unsigned int i = 0; i < word_array_length; i++) {
            if (queryWord == getWord(i)) {
                for (auto sentence:sentences) {
                    if (is_wordInSentence(*sentence, i)) {
                        count += 1;
                        num += sentence->label;
                    }
                }
                float word_weight = num / count;
                word_weights.push_back(word_weight);
                count = 0;
                num = 0;
            }
        }
    }
    float foundOccurances = (float) word_weights.size();
    if (!foundOccurances) {
        return -1.0;
        }
    else {
        for (float wordWeight:word_weights) {
            sum += wordWeight;
        }
    }
    return sum / foundOccurances;
}

int CountVectorizer::is_wordInSentence(Sentence sentence_, unsigned int idx) {
    return (sentence_.sentence_array.size() > idx && sentence_.sentence_array[idx]);
}


// ======================TRAINING FUNCTIONS======================= |


void CountVectorizer::pushSentenceToWordArray(vector<string> new_sentence_vector) {
   for (string word:new_sentence_vector) {
       if (!CountVectorizerContainsWord(word)) {
           word_array.push_back(word);
       }
   }
}

shared_ptr<Sentence> CountVectorizer::createSentenceObject(vector<string> new_sentence_vector, bool label_) {
    shared_ptr<Sentence> new_sentence (new Sentence);
    unsigned int word_array_size = getSize();
    for (unsigned int i = 0; i < word_array_size; i++) {
        if (std::find(new_sentence_vector.begin(),
         new_sentence_vector.end(), word_array[i]) != new_sentence_vector.end()) {
             new_sentence->sentence_array.push_back(1);
        }
        else {
            new_sentence->sentence_array.push_back(0);
        }
    }
    new_sentence->label = label_;
    return new_sentence;
}

void CountVectorizer::addSentence(string new_sentence, bool label_) {
    vector<string> processedString;
    processedString = buildSentenceVector(new_sentence);
    pushSentenceToWordArray(processedString);
    shared_ptr<Sentence> sentObj = createSentenceObject(processedString, label_);
    sentences.push_back(sentObj);
}


// ======================TRAINING UTILITIES========================= |


bool CountVectorizer::CountVectorizerContainsWord(string word_to_check) {
    for (string word:word_array) {
        if (word == word_to_check) {
            return true;
        }
    }
    return false;
}

vector<string> CountVectorizer::buildSentenceVector(string sentence_) {
   string new_word = "";
   vector<string> ret;
   set<char> punctuation = {
       '.', '!', '?', ',', '\'', '/', ';'
       };
//    set<string> stopWords = {
//        "The", "the", "a", "A", "an", "An",
//         "This", "this", "That", "that", "is",
//          "Is", "my", "My"
//        };
   for (char x : sentence_) {
       if (isupper(x) && !case_sensitive) {
           x = tolower(x);
       }
       if (x == ' ') {
        //    if (stopWords.count(new_word)) {
        //        new_word = "";   // Don't push back a stop word
        //    }
        //    else {
               ret.push_back(new_word); 
               new_word = ""; 
        //    }
       }
       else if (punctuation.count(x)) {
           ret.push_back(new_word);
           string s(1, x);
           ret.push_back(s);  // Ugly
           new_word = "";
       }
       else { 
           new_word = new_word + x; 
       }
   }
   if (new_word != "") {ret.push_back(new_word);}
   return ret;
}
