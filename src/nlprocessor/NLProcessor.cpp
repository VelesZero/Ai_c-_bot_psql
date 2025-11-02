#include "src/nlprocessor/NLProcessor.h"
#include "src/utils/Logger.h"
#include "src/ml/ModelTrainer.h"
#include <sstream>

NLProcessor::~NLProcessor() {}

NLProcessor::NLProcessor() {
    trainer_ = std::make_unique<MLModelTrainer>();
    modelLoaded_ = false;
}

bool NLProcessor::initialize(const std::string& modelPath) {
    Logger::getInstance().info("Initializing NLProcessor with model: " + modelPath);
    modelPath_ = modelPath;
    modelLoaded_ = trainer_->load(modelPath);
    if (modelLoaded_) {
        Logger::getInstance().info("Model loaded successfully");
        return true;
    }
    Logger::getInstance().warning("Failed to load model from: " + modelPath);
    return false;
}

bool NLProcessor::trainModel(const std::string& trainingDataPath, const std::string& modelOutputPath) {
    Logger::getInstance().info("Training model from: " + trainingDataPath);
    if (!trainer_->loadDataset(trainingDataPath)) {
        Logger::getInstance().error("Failed to load training data");
        return false;
    }
    if (!trainer_->train()) {
        Logger::getInstance().error("Model training failed");
        return false;
    }
    if (!trainer_->save(modelOutputPath)) {
        Logger::getInstance().error("Failed to save model");
        return false;
    }
    modelPath_ = modelOutputPath;
    modelLoaded_ = true;
    Logger::getInstance().info("Model trained and saved successfully");
    return true;
}

NLProcessor::ProcessingResult NLProcessor::processQueryDetailed(const std::string& naturalLanguageQuery) {
    ProcessingResult result;
    result.success = false;
    result.confidence = 0.0;

    if (!modelLoaded_ && !modelPath_.empty()) {
        initialize(modelPath_);
    }

    Logger::getInstance().info("Processing query: " + naturalLanguageQuery);

    try {
        std::string sql = trainer_->predict(naturalLanguageQuery);
        if (!sql.empty()) {
            result.sqlQuery = sql;
            result.confidence = 0.85;
            result.success = true;
            Logger::getInstance().info("Predicted SQL: " + sql);
        } else {
            result.success = false;
            result.errorMessage = "Empty prediction";
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
        Logger::getInstance().error("Query processing failed: " + result.errorMessage);
    }

    return result;
}

std::string NLProcessor::processQuery(const std::string& naturalLanguageQuery) {
    auto r = processQueryDetailed(naturalLanguageQuery);
    return r.success ? r.sqlQuery : std::string();
}