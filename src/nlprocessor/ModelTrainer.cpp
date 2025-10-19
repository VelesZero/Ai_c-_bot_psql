#include "src/nlprocessor/ModelTrainer.h"
#include "src/nlprocessor/Tokenizer.h"
#include "src/utils/Logger.h"
#include "src/utils/Utilities.h"
#include <fstream>
#include <algorithm>

ModelTrainer::ModelTrainer() {}

bool ModelTrainer::loadTrainingData(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            Logger::getInstance().error("Cannot open training data file: " + filename);
            return false;
        }
        
        json data;
        file >> data;
        
        trainingData_.clear();
        
        for (const auto& item : data["examples"]) {
            TrainingExample example;
            example.naturalLanguage = item["natural_language"];
            example.sqlQuery = item["sql_query"];
            example.queryType = extractQueryType(example.sqlQuery);
            example.keywords = extractKeywords(example.naturalLanguage);
            
            trainingData_.push_back(example);
        }
        
        Logger::getInstance().info("Loaded " + std::to_string(trainingData_.size()) + 
                                  " training examples");
        return true;
        
    } catch (const std::exception& e) {
        Logger::getInstance().error("Error loading training data: " + std::string(e.what()));
        return false;
    }
}

bool ModelTrainer::train() {
    if (trainingData_.empty()) {
        Logger::getInstance().error("No training data available");
        return false;
    }
    
    Logger::getInstance().info("Starting model training...");
    
    patterns_.clear();
    keywordFrequency_.clear();
    
    extractPatterns();
    
    Logger::getInstance().info("Training completed. Patterns: " + 
                              std::to_string(patterns_.size()));
    return true;
}

void ModelTrainer::extractPatterns() {
    for (const auto& example : trainingData_) {
        for (const auto& keyword : example.keywords) {
            patterns_[keyword].push_back(example.sqlQuery);
            keywordFrequency_[keyword]++;
        }
        
        // Добавляем паттерн по типу запроса
        patterns_[example.queryType].push_back(example.sqlQuery);
    }
}

std::string ModelTrainer::extractQueryType(const std::string& sqlQuery) const {
    std::string normalized = Utils::toLower(Utils::trim(sqlQuery));
    
    if (normalized.find("select") == 0) return "SELECT";
    if (normalized.find("insert") == 0) return "INSERT";
    if (normalized.find("update") == 0) return "UPDATE";
    if (normalized.find("delete") == 0) return "DELETE";
    
    return "UNKNOWN";
}

std::vector<std::string> ModelTrainer::extractKeywords(const std::string& text) const {
    Tokenizer tokenizer;
    auto tokens = tokenizer.tokenize(text);
    return tokenizer.removeStopWords(tokens);
}

bool ModelTrainer::saveModel(const std::string& filename) {
    try {
        json model;
        
        // Сохранение паттернов
        for (const auto& [keyword, queries] : patterns_) {
            model["patterns"][keyword] = queries;
        }
        
        // Сохранение частот
        for (const auto& [keyword, freq] : keywordFrequency_) {
            model["frequencies"][keyword] = freq;
        }
        
        std::ofstream file(filename);
        file << model.dump(2);
        
        Logger::getInstance().info("Model saved to: " + filename);
        return true;
        
    } catch (const std::exception& e) {
        Logger::getInstance().error("Error saving model: " + std::string(e.what()));
        return false;
    }
}

bool ModelTrainer::loadModel(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            Logger::getInstance().warning("Model file not found: " + filename);
            return false;
        }
        
        json model;
        file >> model;
        
        patterns_.clear();
        keywordFrequency_.clear();
        
        // Загрузка паттернов
        if (model.contains("patterns")) {
            for (auto& [key, value] : model["patterns"].items()) {
                patterns_[key] = value.get<std::vector<std::string>>();
            }
        }
        
        // Загрузка частот
        if (model.contains("frequencies")) {
            for (auto& [key, value] : model["frequencies"].items()) {
                keywordFrequency_[key] = value.get<int>();
            }
        }
        
        Logger::getInstance().info("Model loaded from: " + filename);
        return true;
        
    } catch (const std::exception& e) {
        Logger::getInstance().error("Error loading model: " + std::string(e.what()));
        return false;
    }
}

std::string ModelTrainer::findSimilarQuery(const std::string& input) const {
    if (trainingData_.empty()) {
        return "";
    }

    auto inputKeywords = extractKeywords(input);
    std::string bestMatch = "";
    int maxScore = 0;

    for (const auto& example : trainingData_) {
        int score = 0;
        auto exampleKeywords = extractKeywords(example.naturalLanguage);

        // Подсчет совпадающих ключевых слов
        for (const auto& inputKeyword : inputKeywords) {
            for (const auto& exampleKeyword : exampleKeywords) {
                if (Utils::toLower(inputKeyword) == Utils::toLower(exampleKeyword)) {
                    score++;
                }
            }
        }

        // Если нашли точное совпадение по ключевым словам, возвращаем сразу
        if (score == static_cast<int>(inputKeywords.size()) && score > 0) {
            return example.sqlQuery;
        }

        // Иначе сохраняем лучший результат
        if (score > maxScore) {
            maxScore = score;
            bestMatch = example.sqlQuery;
        }
    }

    return bestMatch;
}