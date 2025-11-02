#ifndef NL_PROCESSOR_H
#define NL_PROCESSOR_H

#include <string>
#include <vector>
#include <memory>

class MLModelTrainer; // forward declaration

class NLProcessor {
public:
    NLProcessor();
    virtual ~NLProcessor();
    
    bool initialize(const std::string& modelPath);
    bool trainModel(const std::string& trainingDataPath, const std::string& modelPath);
    
    std::string processQuery(const std::string& naturalLanguageQuery);
    
    struct ProcessingResult {
        std::string sqlQuery;
        double confidence;
        std::vector<std::string> detectedKeywords;
        std::string queryType;
        bool success;
        std::string errorMessage;
    };
    
    ProcessingResult processQueryDetailed(const std::string& naturalLanguageQuery);

private:
    std::unique_ptr<MLModelTrainer> trainer_;
    bool modelLoaded_;
    std::string modelPath_;
};

#endif // NL_PROCESSOR_H