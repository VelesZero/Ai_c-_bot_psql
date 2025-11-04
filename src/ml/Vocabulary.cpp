#include "Vocabulary.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>

Vocabulary::Vocabulary() : nextIdx_(4) {
    word2idx_["<PAD>"] = PAD_TOKEN;
    word2idx_["<SOS>"] = SOS_TOKEN;
    word2idx_["<EOS>"] = EOS_TOKEN;
    word2idx_["<UNK>"] = UNK_TOKEN;
    
    idx2word_[PAD_TOKEN] = "<PAD>";
    idx2word_[SOS_TOKEN] = "<SOS>";
    idx2word_[EOS_TOKEN] = "<EOS>";
    idx2word_[UNK_TOKEN] = "<UNK>";
}

void Vocabulary::addWord(const std::string& word) {
    if (word2idx_.find(word) == word2idx_.end()) {
        word2idx_[word] = nextIdx_;
        idx2word_[nextIdx_] = word;
        nextIdx_++;
    }
}

void Vocabulary::addSentence(const std::string& text) {
    auto tokens = tokenize(text);
    for (const auto& t : tokens) {
        addWord(t);
    }
}

int Vocabulary::getIndex(const std::string& word) const {
    auto it = word2idx_.find(word);
    return (it != word2idx_.end()) ? it->second : UNK_TOKEN;
}

std::string Vocabulary::getWord(int index) const {
    auto it = idx2word_.find(index);
    return (it != idx2word_.end()) ? it->second : "<UNK>";
}

std::vector<std::string> Vocabulary::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string token;
    
    for (char c : text) {
        if (std::isspace(c) || c == ',' || c == '(' || c == ')' || 
            c == ';' || c == '=' || c == '<' || c == '>') {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
            if (!std::isspace(c)) {
                tokens.push_back(std::string(1, c));
            }
        } else {
            token += std::tolower(c);
        }
    }
    
    if (!token.empty()) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::vector<int> Vocabulary::encode(const std::string& sentence) const {
    auto tokens = tokenize(sentence);
    std::vector<int> indices;
    indices.push_back(SOS_TOKEN);
    
    for (const auto& token : tokens) {
        indices.push_back(getIndex(token));
    }
    
    indices.push_back(EOS_TOKEN);
    return indices;
}

std::string Vocabulary::decode(const std::vector<int>& indices) const {
    std::string result;
    
    for (int idx : indices) {
        if (idx == PAD_TOKEN || idx == SOS_TOKEN || idx == EOS_TOKEN) {
            continue;
        }
        
        std::string word = getWord(idx);
        if (!result.empty() && word != "," && word != "(" && word != ")" && 
            word != ";" && word != "=" && word != "<" && word != ">") {
            result += " ";
        }
        result += word;
    }
    
    return result;
}

bool Vocabulary::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) return false;
    
    file << nextIdx_ << "\n";
    for (const auto& [word, idx] : word2idx_) {
        file << word << "\t" << idx << "\n";
    }
    
    return true;
}

bool Vocabulary::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    
    word2idx_.clear();
    idx2word_.clear();
    
    file >> nextIdx_;
    file.ignore();
    
    std::string line;
    while (std::getline(file, line)) {
        size_t tab = line.find('\t');
        if (tab != std::string::npos) {
            std::string word = line.substr(0, tab);
            int idx = std::stoi(line.substr(tab + 1));
            word2idx_[word] = idx;
            idx2word_[idx] = word;
        }
    }
    
    return true;
}
