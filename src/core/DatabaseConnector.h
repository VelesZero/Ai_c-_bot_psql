#ifndef DATABASE_CONNECTOR_H
#define DATABASE_CONNECTOR_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <pqxx/pqxx>

class DatabaseConnector {
public:
    DatabaseConnector();
    ~DatabaseConnector();
    
    bool connect(const std::string& host, int port, 
                const std::string& dbname,
                const std::string& user, 
                const std::string& password);
    
    void disconnect();
    bool isConnected() const;
    
    struct QueryResult {
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
        int rowCount;
        bool success;
        std::string errorMessage;
    };
    
    QueryResult executeQuery(const std::string& query);
    std::vector<std::string> getTableNames();
    std::map<std::string, std::vector<std::string>> getTableSchema(const std::string& tableName);

private:
    std::unique_ptr<pqxx::connection> connection_;
    std::string connectionString_;
    
    std::string buildConnectionString(const std::string& host, int port,
                                     const std::string& dbname,
                                     const std::string& user,
                                     const std::string& password);
};

#endif // DATABASE_CONNECTOR_H