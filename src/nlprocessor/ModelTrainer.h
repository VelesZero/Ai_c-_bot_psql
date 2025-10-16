#ifndef MODEL_TRAINER_H
#define MODEL_TRAINER_H

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ModelTrainer {
public:
    struct TrainingExample {
        std::string naturalLanguage;
        std::string sqlQuery;
        std::vector<std::string> keywords;
        std::string queryType; // SELECT, INSERT, UPDATE, DELETE
    };
    
    ModelTrainer();
    
    bool loadTrainingData(const std::string& filename);
    bool train();
    bool saveModel(const std::string& filename);
    bool loadModel(const std::string& filename);
    
    std::map<std::string, double> getPatternScores(const std::string& input) const;
    
private:
    std::vector<TrainingExample> trainingData_;
    std::map<std::string, std::vector<std::string>> patterns_; // keyword -> SQL templates
    std::map<std::string, int> keywordFrequency_;
    
    void extractPatterns();
    std::string extractQueryType(const std::string& sqlQuery) const;
    std::vector<std::string> extractKeywords(const std::string& text) const;
};

#endif // MODEL_TRAINER_H