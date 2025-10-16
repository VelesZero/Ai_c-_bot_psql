#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <map>
#include <memory>

class Config {
public:
    static Config& getInstance();
    
    bool loadFromFile(const std::string& filename);
    
    std::string get(const std::string& key, const std::string& defaultValue = "") const;
    int getInt(const std::string& key, int defaultValue = 0) const;
    bool getBool(const std::string& key, bool defaultValue = false) const;
    
    void set(const std::string& key, const std::string& value);
    
    // Database configuration
    std::string getDbHost() const;
    int getDbPort() const;
    std::string getDbName() const;
    std::string getDbUser() const;
    std::string getDbPassword() const;
    
    // Model configuration
    std::string getModelPath() const;
    std::string getTrainingDataPath() const;
    
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

private:
    Config();
    
    std::map<std::string, std::string> config_;
    void setDefaults();
};

#endif // CONFIG_H