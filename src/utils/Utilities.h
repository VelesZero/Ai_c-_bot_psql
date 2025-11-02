#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>

namespace Utils {
    // Обрезка пробелов
    inline std::string trim(const std::string& str) {
        auto start = std::find_if_not(str.begin(), str.end(), 
                                     [](unsigned char ch) { return std::isspace(ch); });
        auto end = std::find_if_not(str.rbegin(), str.rend(), 
                                   [](unsigned char ch) { return std::isspace(ch); }).base();
        return (start < end) ? std::string(start, end) : std::string();
    }
    
    // Разделение строки
    inline std::vector<std::string> split(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(trim(token));
        }
        return tokens;
    }
    
    // Приведение к нижнему регистру
    inline std::string toLower(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(),
                      [](unsigned char c) { return std::tolower(c); });
        return result;
    }
    
    // Проверка на пустоту
    inline bool isEmpty(const std::string& str) {
        return trim(str).empty();
    }
}

#endif // UTILITIES_H