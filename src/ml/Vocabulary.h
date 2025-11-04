#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

class Vocabulary {
public:
    static constexpr int PAD_TOKEN = 0;
    static constexpr int SOS_TOKEN = 1;  // Start of sequence
    static constexpr int EOS_TOKEN = 2;  // End of sequence
    static constexpr int UNK_TOKEN = 3;  // Unknown token
    
    Vocabulary();
    
    void addWord(const std::string& word);
    // Add all tokens from a raw sentence into the vocabulary
    void addSentence(const std::string& text);
    int getIndex(const std::string& word) const;
    std::string getWord(int index) const;
    
    std::vector<int> encode(const std::string& sentence) const;
    std::string decode(const std::vector<int>& indices) const;
    
    int size() const { return word2idx_.size(); }
    
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    
private:
    std::unordered_map<std::string, int> word2idx_;
    std::unordered_map<int, std::string> idx2word_;
    int nextIdx_;
    
    std::vector<std::string> tokenize(const std::string& text) const;
};

#endif
