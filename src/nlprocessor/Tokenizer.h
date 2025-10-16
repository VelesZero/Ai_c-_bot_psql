#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <set>

class Tokenizer {
public:
    Tokenizer();
    
    std::vector<std::string> tokenize(const std::string& text);
    std::vector<std::string> removeStopWords(const std::vector<std::string>& tokens);
    std::string normalize(const std::string& token);

private:
    std::set<std::string> stopWords_;
    
    void initializeStopWords();
    bool isStopWord(const std::string& word) const;
};

#endif // TOKENIZER_H