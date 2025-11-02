#ifndef RESPONSE_PARSER_H
#define RESPONSE_PARSER_H

#include "src/core/DatabaseConnector.h"
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ResponseParser {
public:
    enum class OutputFormat {
        JSON,
        TABLE,
        CSV,
        PLAIN
    };
    
    ResponseParser();
    
    std::string formatResponse(const DatabaseConnector::QueryResult& result,
                               OutputFormat format = OutputFormat::TABLE);
    
    json toJSON(const DatabaseConnector::QueryResult& result);
    std::string toTable(const DatabaseConnector::QueryResult& result);
    std::string toCSV(const DatabaseConnector::QueryResult& result);
    std::string toPlainText(const DatabaseConnector::QueryResult& result);

private:
    std::vector<size_t> calculateColumnWidths(const DatabaseConnector::QueryResult& result);
    std::string padString(const std::string& str, size_t width);
};

#endif // RESPONSE_PARSER_H