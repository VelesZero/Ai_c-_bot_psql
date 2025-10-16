#include "src/config/Config.h"
#include "src/utils/Logger.h"
#include "src/utils/Utilities.h"
#include <fstream>
#include <sstream>

Config::Config() {
    setDefaults();
}

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

void Config::setDefaults() {
    config_["db_host"] = "localhost";
    config_["db_port"] = "5432";
    config_["db_name"] = "testdb";
    config_["db_user"] = "postgres";
    config_["db_password"] = "";
    config_["model_path"] = "models/nl_to_sql_model.json";
    config_["training_data_path"] = "training_data/queries.json";
    config_["log_file"] = "agent.log";
    config_["log_level"] = "INFO";
}

bool Config::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        Logger::getInstance().error("Cannot open config file: " + filename);
        return false;
    }
    
    std::string line;
    int lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        line = Utils::trim(line);
        
        // Пропуск комментариев и пустых строк
        if (line.empty() || line[0] == '#') continue;
        
        size_t pos = line.find('=');
        if (pos == std::string::npos) {
            Logger::getInstance().warning("Invalid config line " + 
                                        std::to_string(lineNumber) + ": " + line);
            continue;
        }
        
        std::string key = Utils::trim(line.substr(0, pos));
        std::string value = Utils::trim(line.substr(pos + 1));
        
        config_[key] = value;
    }
    
    Logger::getInstance().info("Configuration loaded from: " + filename);
    return true;
}

std::string Config::get(const std::string& key, const std::string& defaultValue) const {
    auto it = config_.find(key);
    return (it != config_.end()) ? it->second : defaultValue;
}

int Config::getInt(const std::string& key, int defaultValue) const {
    auto it = config_.find(key);
    if (it != config_.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

bool Config::getBool(const std::string& key, bool defaultValue) const {
    auto it = config_.find(key);
    if (it != config_.end()) {
        std::string value = Utils::toLower(it->second);
        return (value == "true" || value == "1" || value == "yes");
    }
    return defaultValue;
}

void Config::set(const std::string& key, const std::string& value) {
    config_[key] = value;
}

std::string Config::getDbHost() const { return get("db_host"); }
int Config::getDbPort() const { return getInt("db_port"); }
std::string Config::getDbName() const { return get("db_name"); }
std::string Config::getDbUser() const { return get("db_user"); }
std::string Config::getDbPassword() const { return get("db_password"); }
std::string Config::getModelPath() const { return get("model_path"); }
std::string Config::getTrainingDataPath() const { return get("training_data_path"); }