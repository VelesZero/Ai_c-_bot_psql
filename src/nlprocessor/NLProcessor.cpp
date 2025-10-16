#include "src/nlprocessor/NLProcessor.h"
#include "src/utils/Logger.h"
#include "src/utils/Utilities.h"
#include <algorithm>
#include <sstream>

NLProcessor::NLProcessor() {
    tokenizer_ = std::make_unique<Tokenizer>();
    trainer_ = std::make_unique<ModelTrainer>();
}

bool NLProcessor::initialize(const std::string& modelPath) {
    Logger::getInstance().info("Initializing NL Processor...");
    
    if (!trainer_->loadModel(modelPath)) {
        Logger::getInstance().warning("Could not load model, using default patterns");
        return false;
    }
    
    return true;
}

bool NLProcessor::trainModel(const std::string& trainingDataPath, 
                            const std::string& modelPath) {
    Logger::getInstance().info("Training model...");
    
    if (!trainer_->loadTrainingData(trainingDataPath)) {
        return false;
    }
    
    if (!trainer_->train()) {
        return false;
    }
    
    return trainer_->saveModel(modelPath);
}

std::string NLProcessor::processQuery(const std::string& naturalLanguageQuery) {
    auto result = processQueryDetailed(naturalLanguageQuery);
    return result.sqlQuery;
}

NLProcessor::ProcessingResult NLProcessor::processQueryDetailed(
    const std::string& naturalLanguageQuery) {
    
    ProcessingResult result;
    result.success = false;
    result.confidence = 0.0;
    
    try {
        // Токенизация
        auto tokens = tokenizer_->tokenize(naturalLanguageQuery);
        auto keywords = tokenizer_->removeStopWords(tokens);
        
        result.detectedKeywords = keywords;
        
        if (keywords.empty()) {
            result.errorMessage = "No meaningful keywords found in query";
            return result;
        }
        
        // Получение оценок паттернов
        auto scores = trainer_->getPatternScores(naturalLanguageQuery);
        
        // Определение типа запроса
        result.queryType = detectQueryType(keywords);
        
        // Генерация SQL
        result.sqlQuery = generateSQLFromPatterns(keywords, scores);
        
        if (result.sqlQuery.empty()) {
            result.errorMessage = "Could not generate SQL query";
            return result;
        }
        
        // Расчет уверенности
        double totalScore = 0.0;
        for (const auto& [_, score] : scores) {
            totalScore += score;
        }
        result.confidence = std::min(1.0, totalScore / (keywords.size() * 10.0));
        
        result.success = true;
        
        Logger::getInstance().info("Generated SQL: " + result.sqlQuery + 
                                  " (confidence: " + std::to_string(result.confidence) + ")");
        
    } catch (const std::exception& e) {
        result.errorMessage = "Processing error: " + std::string(e.what());
        Logger::getInstance().error(result.errorMessage);
    }
    
    return result;
}

std::string NLProcessor::detectQueryType(const std::vector<std::string>& keywords) {
    // Простая эвристика для определения типа запроса
    for (const auto& keyword : keywords) {
        std::string lower = Utils::toLower(keyword);
        
        if (lower == "show" || lower == "get" || lower == "find" || 
            lower == "list" || lower == "display" || lower == "select") {
            return "SELECT";
        }
        if (lower == "add" || lower == "insert" || lower == "create") {
            return "INSERT";
        }
        if (lower == "update" || lower == "modify" || lower == "change") {
            return "UPDATE";
        }
        if (lower == "delete" || lower == "remove") {
            return "DELETE";
        }
    }
    
    return "SELECT"; // По умолчанию
}

std::string NLProcessor::extractTableName(const std::vector<std::string>& keywords) {
    // Простая эвристика: ищем существительные после ключевых слов
    std::vector<std::string> tableIndicators = {"from", "in", "table"};
    
    for (size_t i = 0; i < keywords.size(); ++i) {
        std::string lower = Utils::toLower(keywords[i]);
        
        for (const auto& indicator : tableIndicators) {
            if (lower == indicator && i + 1 < keywords.size()) {
                return keywords[i + 1];
            }
        }
    }
    
    // Если не найдено, возвращаем последнее существительное
    if (!keywords.empty()) {
        return keywords.back();
    }
    
    return "table_name";
}

std::vector<std::string> NLProcessor::extractColumns(const std::vector<std::string>& keywords) {
    std::vector<std::string> columns;
    
    // Ищем колонки между определенными ключевыми словами
    bool inColumnSection = false;
    
    for (const auto& keyword : keywords) {
        std::string lower = Utils::toLower(keyword);
        
        if (lower == "select" || lower == "show" || lower == "get") {
            inColumnSection = true;
            continue;
        }
        
        if (lower == "from" || lower == "where") {
            inColumnSection = false;
            continue;
        }
        
        if (inColumnSection) {
            columns.push_back(keyword);
        }
    }
    
    return columns;
}

std::string NLProcessor::extractCondition(const std::string& query,
                                         const std::vector<std::string>& keywords) {
    // Ищем условия после "where", "with", "having"
    std::string lower = Utils::toLower(query);
    
    size_t wherePos = lower.find("where");
    if (wherePos == std::string::npos) {
        wherePos = lower.find("with");
    }
    
    if (wherePos != std::string::npos) {
        return Utils::trim(query.substr(wherePos + 5));
    }
    
    return "";
}

std::string NLProcessor::generateSQLFromPatterns(
    const std::vector<std::string>& keywords,
    const std::map<std::string, double>& scores) {
    
    std::string queryType = detectQueryType(keywords);
    std::string tableName = extractTableName(keywords);
    
    std::ostringstream sql;
    
    if (queryType == "SELECT") {
        auto columns = extractColumns(keywords);
        
        sql << "SELECT ";
        
        if (columns.empty()) {
            sql << "*";
        } else {
            for (size_t i = 0; i < columns.size(); ++i) {
                if (i > 0) sql << ", ";
                sql << columns[i];
            }
        }
        
        sql << " FROM " << tableName;
        
        // Добавление условий (упрощенная версия)
        bool hasCondition = false;
        for (size_t i = 0; i < keywords.size(); ++i) {
            std::string lower = Utils::toLower(keywords[i]);
            if (lower == "where" && i + 2 < keywords.size()) {
                sql << " WHERE " << keywords[i + 1] << " = '" << keywords[i + 2] << "'";
                hasCondition = true;
                break;
            }
        }
        
    } else if (queryType == "INSERT") {
        sql << "INSERT INTO " << tableName << " VALUES (...)";
        
    } else if (queryType == "UPDATE") {
        sql << "UPDATE " << tableName << " SET ... WHERE ...";
        
    } else if (queryType == "DELETE") {
        sql << "DELETE FROM " << tableName << " WHERE ...";
    }
    
    return sql.str();
}