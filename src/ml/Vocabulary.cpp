#include "Vocabulary.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <vector>
#include <iostream>

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
    freq_[word] += 1;
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

void Vocabulary::limitSize(int max_size) {
    if (max_size <= 0) {
        limited_built_ = false;
        return;
    }
    rebuildFromFrequencies(max_size);
    limited_built_ = true;
}

void Vocabulary::rebuildFromFrequencies(int max_size) {
    // Preserve special tokens indices.
    word2idx_.clear();
    idx2word_.clear();
    nextIdx_ = 4;

    word2idx_["<PAD>"] = PAD_TOKEN;
    word2idx_["<SOS>"] = SOS_TOKEN;
    word2idx_["<EOS>"] = EOS_TOKEN;
    word2idx_["<UNK>"] = UNK_TOKEN;

    idx2word_[PAD_TOKEN] = "<PAD>";
    idx2word_[SOS_TOKEN] = "<SOS>";
    idx2word_[EOS_TOKEN] = "<EOS>";
    idx2word_[UNK_TOKEN] = "<UNK>";

    int budget = max_size - 4;
    if (budget <= 0) {
        return;
    }

    std::vector<std::pair<std::string, int>> items;
    items.reserve(freq_.size());
    for (const auto& kv : freq_) {
        // Never include special tokens into the counted vocabulary.
        if (kv.first == "<PAD>" || kv.first == "<SOS>" || kv.first == "<EOS>" || kv.first == "<UNK>") {
            continue;
        }
        items.push_back(kv);
    }

    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second; // higher freq first
        return a.first < b.first; // tie-breaker for determinism
    });

    if (static_cast<int>(items.size()) > budget) {
        items.resize(static_cast<size_t>(budget));
    }

    for (const auto& kv : items) {
        const auto& w = kv.first;
        if (word2idx_.find(w) == word2idx_.end()) {
            word2idx_[w] = nextIdx_;
            idx2word_[nextIdx_] = w;
            nextIdx_++;
        }
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
    freq_.clear();
    limited_built_ = true;
    
    std::string line;
    if (std::getline(file, line)) {
        try {
            nextIdx_ = std::stoi(line);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error parsing nextIdx_ from vocabulary file: " << e.what() << std::endl;
            return false;
        } catch (const std::out_of_range& e) {
            std::cerr << "nextIdx_ value out of range in vocabulary file: " << e.what() << std::endl;
            return false;
        }
    } else {
        std::cerr << "Error: Could not read nextIdx_ from vocabulary file" << std::endl;
        return false;
    }
    
    while (std::getline(file, line)) {
        size_t tab = line.find('\t');
        if (tab != std::string::npos) {
            std::string word = line.substr(0, tab);
            try {
                int idx = std::stoi(line.substr(tab + 1));
                word2idx_[word] = idx;
                idx2word_[idx] = word;
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error parsing index for word '" << word << "': " << e.what() << std::endl;
                return false;
            } catch (const std::out_of_range& e) {
                std::cerr << "Index value out of range for word '" << word << "': " << e.what() << std::endl;
                return false;
            }
        }
    }
    
    return true;
}
