#ifndef GLOBALDATA_H__
#define GLOBALDATA_H__

#include <string>
#include <set>
#include <vector>

using namespace std;

class GlobalData
{
public:
   int POS;
   int NEG;
   int NEU;
   int UNK;
   set<char> punctuation;
   set<string> stopWords;

   GlobalData()
   {
      POS = 1;
      NEG = 0;
      NEU = -1;
      UNK = -2;

      punctuation = {
          '!', '?', '/'
      };

      stopWords = {
          "The", "the", "a", "A", "an", "An",
          "This", "this", "That", "that", "is",
          "Is", "my", "My", ".", ":", ",", ";", "\'", ")", "(",
          "..."
      };

   }
};

#endif // GLOBALDATA_H__