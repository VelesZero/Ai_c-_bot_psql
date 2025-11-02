#include "src/core/Agent.h"
#include "src/utils/Logger.h"

Agent::Agent() 
    : outputFormat_(ResponseParser::OutputFormat::TABLE),
      initialized_(false) {
    
    dbConnector_ = std::make_unique<DatabaseConnector>();
    queryBuilder_ = std::make_unique<QueryBuilder>();
    responseParser_ = std::make_unique<ResponseParser>();
    nlProcessor_ = std::make_unique<NLProcessor>();
}

Agent::~Agent() {
    if (dbConnector_) {
        dbConnector_->disconnect();
    }
}

bool Agent::initialize(const std::string& configFile) {
    Logger::getInstance().info("Initializing AI Query Agent...");
    
    Config& config = Config::getInstance();
    
    if (!configFile.empty()) {
        if (!config.loadFromFile(configFile)) {
            Logger::getInstance().warning("Using default configuration");
        }
    }
    
    // Настройка логирования
    Logger::getInstance().setLogFile(config.get("log_file", "agent.log"));
    
    // Инициализация NL процессора
    std::string modelPath = config.getModelPath();
    nlProcessor_->initialize(modelPath);
    
    initialized_ = true;
    Logger::getInstance().info("Agent initialized successfully");
    
    return true;
}

bool Agent::connectToDatabase(const std::string& host, int port,
                              const std::string& dbname,
                              const std::string& user,
                              const std::string& password) {
    
    Logger::getInstance().info("Connecting to database: " + dbname);
    
    return dbConnector_->connect(host, port, dbname, user, password);
}

std::string Agent::processNaturalLanguageQuery(const std::string& query) {
    auto response = processQueryDetailed(query);
    return response.result;
}

Agent::QueryResponse Agent::processQueryDetailed(const std::string& naturalLanguageQuery) {
    QueryResponse response;
    response.success = false;
    response.confidence = 0.0;
    
    if (!initialized_) {
        response.errorMessage = "Agent not initialized";
        Logger::getInstance().error(response.errorMessage);
        return response;
    }

    const bool dbReady = dbConnector_->isConnected();
    
    Logger::getInstance().info("Processing query: " + naturalLanguageQuery);
    
    // Обработка естественного языка
    auto nlResult = nlProcessor_->processQueryDetailed(naturalLanguageQuery);
    
    if (!nlResult.success) {
        response.errorMessage = nlResult.errorMessage;
        return response;
    }
    
    response.sqlQuery = nlResult.sqlQuery;
    response.confidence = nlResult.confidence;
    
    // Валидация SQL
    if (!queryBuilder_->validateSQL(response.sqlQuery)) {
        response.errorMessage = "Generated SQL failed validation";
        Logger::getInstance().error(response.errorMessage);
        return response;
    }
    
    // Если БД недоступна, но оффлайн-режим разрешён — вернуть только сгенерированный SQL
    if (!dbReady) {
        if (allowOfflineSQL_) {
            response.result = ""; // нет результата без выполнения
            response.success = true;
            Logger::getInstance().warning("Offline SQL generation: database is not connected, returning SQL only");
            return response;
        } else {
            response.errorMessage = "Not connected to database";
            Logger::getInstance().error(response.errorMessage);
            return response;
        }
    }

    // Выполнение запроса (когда БД подключена)
    {
        auto dbResult = dbConnector_->executeQuery(response.sqlQuery);
        if (!dbResult.success) {
            response.errorMessage = dbResult.errorMessage;
            return response;
        }
        response.result = responseParser_->formatResponse(dbResult, outputFormat_);
        response.success = true;
    }
    
    return response;
}

std::string Agent::executeSQL(const std::string& sqlQuery) {
    if (!dbConnector_->isConnected()) {
        return "Error: Not connected to database";
    }
    
    if (!queryBuilder_->validateSQL(sqlQuery)) {
        return "Error: SQL validation failed";
    }
    
    auto result = dbConnector_->executeQuery(sqlQuery);
    return responseParser_->formatResponse(result, outputFormat_);
}

bool Agent::trainModel(const std::string& trainingDataPath) {
    Logger::getInstance().info("Training model with data from: " + trainingDataPath);
    
    Config& config = Config::getInstance();
    std::string modelPath = config.getModelPath();
    
    return nlProcessor_->trainModel(trainingDataPath, modelPath);
}

void Agent::setOutputFormat(ResponseParser::OutputFormat format) {
    outputFormat_ = format;
}