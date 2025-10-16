#ifndef NL_PROCESSOR_H
#define NL_PROCESSOR_H

#include "src/nlprocessor/Tokenizer.h"
#include "src/nlprocessor/ModelTrainer.h"
#include <string>
#include <vector>
#include <memory>

class NLProcessor {
public:
    NLProcessor();
    
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
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<ModelTrainer> trainer_;
    
    std::string generateSQLFromPatterns(const std::vector<std::string>& keywords,
                                       const std::map<std::string, double>& scores);
    
    std::string detectQueryType(const std::vector<std::string>& keywords);
    std::string extractTableName(const std::vector<std::string>& keywords);
    std::vector<std::string> extractColumns(const std::vector<std::string>& keywords);
    std::string extractCondition(const std::string& query, 
                                 const std::vector<std::string>& keywords);
};

#endif // NL_PROCESSOR_H