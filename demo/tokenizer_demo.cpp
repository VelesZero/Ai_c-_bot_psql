#include <iostream>
#include <string>
#include <vector>

#include <cctype>

static std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == '(' || c == ')' ||
            c == ';' || c == '=' || c == '<' || c == '>') {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
            if (!std::isspace(static_cast<unsigned char>(c))) {
                tokens.push_back(std::string(1, c));
            }
        } else {
            token += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

int main() {
    std::cout << "Enter query (Ctrl+D to exit):\n";
    std::string line;
    while (std::getline(std::cin, line)) {
        auto tokens = tokenize(line);
        std::cout << "Tokens:\n";
        for (const auto& t : tokens) {
            std::cout << t << "\n";
        }
        std::cout << "----\n";
    }
    return 0;
}
