#ifndef AGENT_H
#define AGENT_H

#include "src/core/DatabaseConnector.h"
#include "src/core/QueryBuilder.h"
#include "src/core/ResponseParser.h"
#include "src/nlprocessor/NLProcessor.h"
#include "src/config/Config.h"
#include <memory>
#include <string>

class Agent {
public:
    Agent();
    ~Agent();
    
    bool initialize(const std::string& configFile = "");
    bool connectToDatabase(const std::string& host, int port,
                          const std::string& dbname,
                          const std::string& user,
                          const std::string& password);
    
    std::string processNaturalLanguageQuery(const std::string& query);
    std::string executeSQL(const std::string& sqlQuery);
    
    bool trainModel(const std::string& trainingDataPath);
    
    void setOutputFormat(ResponseParser::OutputFormat format);

    // Разрешить оффлайн-генерацию SQL без подключения к БД.
    // Если включено, метод processQueryDetailed() будет генерировать SQL и возвращать
    // успех даже при отсутствии соединения с БД (поле result будет пустым).
    void setAllowOfflineSQL(bool allow) { allowOfflineSQL_ = allow; }
    
    struct QueryResponse {
        bool success;
        std::string result;
        std::string sqlQuery;
        double confidence;
        std::string errorMessage;
    };
    
    QueryResponse processQueryDetailed(const std::string& naturalLanguageQuery);

private:
    std::unique_ptr<DatabaseConnector> dbConnector_;
    std::unique_ptr<QueryBuilder> queryBuilder_;
    std::unique_ptr<ResponseParser> responseParser_;
    std::unique_ptr<NLProcessor> nlProcessor_;
    
    ResponseParser::OutputFormat outputFormat_;
    
    bool initialized_;
    bool allowOfflineSQL_ = false;
};

#endif // AGENT_H