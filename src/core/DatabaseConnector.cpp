#include "src/core/DatabaseConnector.h"
#include "src/utils/Logger.h"
#include <sstream>

DatabaseConnector::DatabaseConnector() : connection_(nullptr) {}

DatabaseConnector::~DatabaseConnector() {
    disconnect();
}

std::string DatabaseConnector::buildConnectionString(
    const std::string& host, int port,
    const std::string& dbname,
    const std::string& user,
    const std::string& password) {
    
    std::ostringstream oss;
    oss << "host=" << host 
        << " port=" << port
        << " dbname=" << dbname
        << " user=" << user;
    
    if (!password.empty()) {
        oss << " password=" << password;
    }
    
    return oss.str();
}

bool DatabaseConnector::connect(const std::string& host, int port,
                               const std::string& dbname,
                               const std::string& user,
                               const std::string& password) {
    try {
        connectionString_ = buildConnectionString(host, port, dbname, user, password);
        connection_ = std::make_unique<pqxx::connection>(connectionString_);
        
        if (connection_->is_open()) {
            Logger::getInstance().info("Successfully connected to database: " + dbname);
            return true;
        }
    } catch (const std::exception& e) {
        Logger::getInstance().error("Database connection failed: " + std::string(e.what()));
        connection_.reset();
    }
    
    return false;
}

void DatabaseConnector::disconnect() {
    if (connection_ && connection_->is_open()) {
        connection_->close();
        Logger::getInstance().info("Database connection closed");
    }
    connection_.reset();
}

bool DatabaseConnector::isConnected() const {
    return connection_ && connection_->is_open();
}

DatabaseConnector::QueryResult DatabaseConnector::executeQuery(const std::string& query) {
    QueryResult result;
    result.success = false;
    result.rowCount = 0;
    
    if (!isConnected()) {
        result.errorMessage = "Not connected to database";
        Logger::getInstance().error(result.errorMessage);
        return result;
    }
    
    try {
        pqxx::work txn(*connection_);
        pqxx::result res = txn.exec(query);
        
        // Получение имен колонок
        for (size_t i = 0; i < res.columns(); ++i) {
            result.columns.push_back(res.column_name(i));
        }
        
        // Получение данных
        for (const auto& row : res) {
            std::vector<std::string> rowData;
            for (size_t i = 0; i < row.size(); ++i) {
                rowData.push_back(row[i].is_null() ? "NULL" : row[i].c_str());
            }
            result.rows.push_back(rowData);
        }
        
        result.rowCount = res.size();
        result.success = true;
        txn.commit();
        
        Logger::getInstance().info("Query executed successfully. Rows: " + 
                                  std::to_string(result.rowCount));
        
    } catch (const std::exception& e) {
        result.errorMessage = e.what();
        Logger::getInstance().error("Query execution failed: " + result.errorMessage);
    }
    
    return result;
}

std::vector<std::string> DatabaseConnector::getTableNames() {
    std::vector<std::string> tables;
    
    std::string query = 
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY table_name";
    
    auto result = executeQuery(query);
    
    if (result.success) {
        for (const auto& row : result.rows) {
            if (!row.empty()) {
                tables.push_back(row[0]);
            }
        }
    }
    
    return tables;
}

std::map<std::string, std::vector<std::string>> 
DatabaseConnector::getTableSchema(const std::string& tableName) {
    std::map<std::string, std::vector<std::string>> schema;
    
    std::string query = 
        "SELECT column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_name = '" + tableName + "' "
        "ORDER BY ordinal_position";
    
    auto result = executeQuery(query);
    
    if (result.success) {
        for (const auto& row : result.rows) {
            if (row.size() >= 3) {
                schema[row[0]] = {row[1], row[2]};
            }
        }
    }
    
    return schema;
}