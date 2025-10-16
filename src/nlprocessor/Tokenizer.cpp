#include "src/nlprocessor/Tokenizer.h"
#include "src/utils/Utilities.h"
#include <algorithm>
#include <sstream>
#include <cctype>

Tokenizer::Tokenizer() {
    initializeStopWords();
}

void Tokenizer::initializeStopWords() {
    stopWords_ = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "should", "could", "may", "might", "must", "can",
        "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "in", "out", "on", "off", "over",
        "under", "again", "further", "then", "once"
    };
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string token;
    
    for (char ch : text) {
        if (std::isalnum(ch) || ch == '_') {
            token += ch;
        } else {
            if (!token.empty()) {
                tokens.push_back(normalize(token));
                token.clear();
            }
            // Сохраняем некоторые специальные символы как отдельные токены
            if (ch == '=' || ch == '>' || ch == '<' || ch == '!' || 
                ch == ',' || ch == '.' || ch == '?') {
                tokens.push_back(std::string(1, ch));
            }
        }
    }
    
    if (!token.empty()) {
        tokens.push_back(normalize(token));
    }
    
    return tokens;
}

std::vector<std::string> Tokenizer::removeStopWords(const std::vector<std::string>& tokens) {
    std::vector<std::string> filtered;
    
    for (const auto& token : tokens) {
        if (!isStopWord(token)) {
            filtered.push_back(token);
        }
    }
    
    return filtered;
}

std::string Tokenizer::normalize(const std::string& token) {
    return Utils::toLower(token);
}

bool Tokenizer::isStopWord(const std::string& word) const {
    return stopWords_.find(Utils::toLower(word)) != stopWords_.end();
}