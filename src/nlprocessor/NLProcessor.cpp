#include "src/nlprocessor/NLProcessor.h"
#include "src/nlprocessor/Tokenizer.h"
#include "src/utils/Logger.h"
#include "src/utils/Utilities.h"
#include <algorithm>
#include <sstream>

NLProcessor::~NLProcessor() {
    // Destructor implementation
}

NLProcessor::NLProcessor() {
    trainer_ = std::make_unique<ModelTrainer>();
    tokenizer_ = std::make_unique<Tokenizer>();
    
    // Попытка загрузить существующую модель
    trainer_->loadModel("models/nl_to_sql_model.json");
}

bool NLProcessor::initialize(const std::string& modelPath) {
    Logger::getInstance().info("Initializing NLProcessor with model: " + modelPath);
    
    // Попытка загрузить существующую модель
    if (trainer_->loadModel(modelPath)) {
        Logger::getInstance().info("Model loaded successfully");
        return true;
    }
    
    Logger::getInstance().warning("Failed to load model from: " + modelPath);
    return false;
}

bool NLProcessor::trainModel(const std::string& trainingDataPath, 
                             const std::string& modelOutputPath) {
    Logger::getInstance().info("Training model from: " + trainingDataPath);
    
    if (!trainer_->loadTrainingData(trainingDataPath)) {
        Logger::getInstance().error("Failed to load training data");
        return false;
    }
    
    if (!trainer_->train()) {
        Logger::getInstance().error("Model training failed");
        return false;
    }
    
    if (!trainer_->saveModel(modelOutputPath)) {
        Logger::getInstance().error("Failed to save model");
        return false;
    }
    
    Logger::getInstance().info("Model trained and saved successfully");
    return true;
}

NLProcessor::ProcessingResult NLProcessor::processQueryDetailed(
    const std::string& naturalLanguageQuery) {
    
    ProcessingResult result;
    result.success = false;
    result.confidence = 0.0;
    
    Logger::getInstance().info("Processing query: " + naturalLanguageQuery);
    
    try {
        // Проверяем, есть ли точное совпадение в обученных примерах
        std::string learnedSQL = trainer_->findSimilarQuery(naturalLanguageQuery);
        if (!learnedSQL.empty()) {
            result.sqlQuery = learnedSQL;
            result.confidence = 1.0;
            result.success = true;
            Logger::getInstance().info("Found learned pattern: " + learnedSQL);
            return result;
        }
        
        // Токенизация
        auto tokens = tokenizer_->tokenize(naturalLanguageQuery);
        
        // Генерация SQL из паттернов
        std::map<std::string, double> scores;
        result.sqlQuery = generateSQLFromPatterns(tokens, scores);
        result.confidence = 0.85;
        result.success = true;
        
        Logger::getInstance().info("Generated SQL: " + result.sqlQuery + 
                                  " (confidence: " + std::to_string(result.confidence) + ")");
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
        Logger::getInstance().error("Query processing failed: " + result.errorMessage);
    }
    
    return result;
}

std::string NLProcessor::processQuery(const std::string& naturalLanguageQuery) {
    auto result = processQueryDetailed(naturalLanguageQuery);
    return result.success ? result.sqlQuery : "";
}

std::string NLProcessor::detectQueryType(const std::vector<std::string>& keywords) {
    for (const auto& keyword : keywords) {
        std::string lower = Utils::toLower(keyword);
        
        if (lower == "select" || lower == "show" || lower == "get" || 
            lower == "list" || lower == "display" || lower == "find") {
            return "SELECT";
        }
        if (lower == "insert" || lower == "add" || lower == "create") {
            return "INSERT";
        }
        if (lower == "update" || lower == "modify" || lower == "change") {
            return "UPDATE";
        }
        if (lower == "delete" || lower == "remove") {
            return "DELETE";
        }
    }
    return "SELECT";
}

std::string NLProcessor::extractTableName(const std::vector<std::string>& keywords) {
    std::vector<std::string> tableNames = {
        "users", "user",
        "products", "product",
        "customers", "customer",
        "orders", "order",
        "categories", "category",
        "items", "item",
        "order_items"
    };
    
    for (const auto& keyword : keywords) {
        std::string lower = Utils::toLower(keyword);
        for (const auto& table : tableNames) {
            if (lower == table) {
                if (lower == "user") return "users";
                if (lower == "product") return "products";
                if (lower == "customer") return "customers";
                if (lower == "order") return "orders";
                if (lower == "category") return "categories";
                if (lower == "item") return "items";
                return lower;
            }
        }
    }
    
    return "users";
}

std::vector<std::string> NLProcessor::extractColumns(const std::vector<std::string>& keywords) {
    (void)keywords;
    return std::vector<std::string>();
}

std::string NLProcessor::extractCondition(const std::string& query, 
                                         const std::vector<std::string>& keywords) {
    (void)query;
    (void)keywords;
    return "";
}

std::string NLProcessor::generateSQLFromPatterns(
    const std::vector<std::string>& keywords,
    const std::map<std::string, double>& scores) {
    
    (void)scores;
    
    // Объединяем keywords в строку для поиска
    std::string queryLower = "";
    for (const auto& kw : keywords) {
        queryLower += Utils::toLower(kw) + " ";
    }
    
    // Извлекаем числа
    std::vector<int> numbers;
    for (const auto& kw : keywords) {
        try {
            numbers.push_back(std::stoi(kw));
        } catch (...) {}
    }
    
    // ==========================================
    // ПАТТЕРНЫ: "show/get/list all [table]"
    // ==========================================
    if (queryLower.find("all") != std::string::npos) {
        if (queryLower.find("users") != std::string::npos || 
            queryLower.find("user") != std::string::npos) {
            return "SELECT * FROM users";
        }
        if (queryLower.find("products") != std::string::npos || 
            queryLower.find("product") != std::string::npos) {
            return "SELECT * FROM products";
        }
        if (queryLower.find("customers") != std::string::npos || 
            queryLower.find("customer") != std::string::npos) {
            return "SELECT * FROM customers";
        }
        if (queryLower.find("orders") != std::string::npos || 
            queryLower.find("order") != std::string::npos) {
            return "SELECT * FROM orders";
        }
        if (queryLower.find("categories") != std::string::npos || 
            queryLower.find("category") != std::string::npos) {
            return "SELECT * FROM categories";
        }
    }
    
    // ==========================================
    // ПАТТЕРНЫ: "count"
    // ==========================================
    if (queryLower.find("count") != std::string::npos || 
        queryLower.find("how many") != std::string::npos) {
        if (queryLower.find("users") != std::string::npos) {
            return "SELECT COUNT(*) as count FROM users";
        }
        if (queryLower.find("products") != std::string::npos) {
            return "SELECT COUNT(*) as count FROM products";
        }
        if (queryLower.find("orders") != std::string::npos) {
            return "SELECT COUNT(*) as count FROM orders";
        }
        if (queryLower.find("customers") != std::string::npos) {
            return "SELECT COUNT(*) as count FROM customers";
        }
    }
    
    // ==========================================
    // ПАТТЕРНЫ с числами
    // ==========================================
    if (!numbers.empty()) {
        int num = numbers[0];
        
        // "N cheapest"
        if (queryLower.find("cheapest") != std::string::npos) {
            return "SELECT * FROM products ORDER BY price ASC LIMIT " + std::to_string(num);
        }
        
        // "N most expensive" или просто "N expensive"
        if (queryLower.find("expensive") != std::string::npos) {
            return "SELECT * FROM products ORDER BY price DESC LIMIT " + std::to_string(num);
        }
        
        // "top N" или "first N"
        if (queryLower.find("top") != std::string::npos || 
            queryLower.find("first") != std::string::npos) {
            if (queryLower.find("products") != std::string::npos) {
                return "SELECT * FROM products LIMIT " + std::to_string(num);
            }
            if (queryLower.find("users") != std::string::npos) {
                return "SELECT * FROM users LIMIT " + std::to_string(num);
            }
            if (queryLower.find("orders") != std::string::npos) {
                return "SELECT * FROM orders LIMIT " + std::to_string(num);
            }
        }
    }
    
    // ==========================================
    // ПАТТЕРНЫ: "cheaper/less than X"
    // ==========================================
    if (!numbers.empty() && (queryLower.find("cheaper") != std::string::npos || 
                             queryLower.find("less") != std::string::npos ||
                             queryLower.find("under") != std::string::npos)) {
        return "SELECT * FROM products WHERE price < " + std::to_string(numbers[0]);
    }
    
    // ==========================================
    // ПАТТЕРНЫ: "more expensive/greater than X"
    // ==========================================
    if (!numbers.empty() && (queryLower.find("more") != std::string::npos || 
                             queryLower.find("greater") != std::string::npos ||
                             queryLower.find("above") != std::string::npos)) {
        return "SELECT * FROM products WHERE price > " + std::to_string(numbers[0]);
    }
    
    // ==========================================
    // ПАТТЕРНЫ: "between X and Y"
    // ==========================================
    if (numbers.size() >= 2 && queryLower.find("between") != std::string::npos) {
        return "SELECT * FROM products WHERE price BETWEEN " + 
               std::to_string(numbers[0]) + " AND " + std::to_string(numbers[1]);
    }
    
    // ==========================================
    // FALLBACK - просто показать таблицу
    // ==========================================
    if (queryLower.find("products") != std::string::npos || 
        queryLower.find("product") != std::string::npos) {
        return "SELECT * FROM products";
    }
    if (queryLower.find("users") != std::string::npos || 
        queryLower.find("user") != std::string::npos) {
        return "SELECT * FROM users";
    }
    if (queryLower.find("customers") != std::string::npos || 
        queryLower.find("customer") != std::string::npos) {
        return "SELECT * FROM customers";
    }
    if (queryLower.find("orders") != std::string::npos || 
        queryLower.find("order") != std::string::npos) {
        return "SELECT * FROM orders";
    }
    
    // Совсем fallback
    return "SELECT * FROM users";
}