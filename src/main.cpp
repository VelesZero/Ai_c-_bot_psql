#include "src/core/Agent.h"
#include "src/config/Config.h"
#include "src/utils/Logger.h"
#include <iostream>
#include <string>

void printHelp() {
    std::cout << "\n=== AI SQL Query Agent ===\n\n";
    std::cout << "Commands:\n";
    std::cout << "  query <text>  - Process natural language query\n";
    std::cout << "  sql <query>   - Execute SQL directly\n";
    std::cout << "  train <file>  - Train model with JSON file\n";
    std::cout << "  offline <on|off> - Toggle offline SQL generation (no DB required)\n";
    std::cout << "  format <type> - Set output format (table/json/csv/plain)\n";
    std::cout << "  tables        - Show all tables\n";
    std::cout << "  help          - Show this help\n";
    std::cout << "  exit          - Exit program\n\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== AI SQL Query Agent ===\n";
    std::cout << "Version 1.0.0\n\n";
    
    // Инициализация агента
    Agent agent;
    
    std::string configFile = (argc > 1) ? argv[1] : "src/config/default.conf";
    
    if (!agent.initialize(configFile)) {
        std::cerr << "Failed to initialize agent\n";
        return 1;
    }
    
    // Подключение к БД
    Config& config = Config::getInstance();
    
    if (!agent.connectToDatabase(
        config.getDbHost(),
        config.getDbPort(),
        config.getDbName(),
        config.getDbUser(),
        config.getDbPassword())) {
        std::cerr << "Failed to connect to database. Enabling OFFLINE SQL mode.\n";
        agent.setAllowOfflineSQL(true);
    } else {
        std::cout << "Connected to database successfully!\n";
    }
    printHelp();
    
    // Основной цикл
    std::string line;
    
    while (true) {
        std::cout << "\n> ";
        std::getline(std::cin, line);
        
        if (line.empty()) continue;
        
        if (line == "exit" || line == "quit") {
            break;
        }
        
        if (line == "help") {
            printHelp();
            continue;
        }
        
        if (line == "tables") {
            std::string result = agent.executeSQL(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'");
            std::cout << result << "\n";
            continue;
        }
        
        if (line.substr(0, 6) == "query ") {
            std::string query = line.substr(6);
            auto response = agent.processQueryDetailed(query);
            
            if (response.success) {
                std::cout << "\nGenerated SQL: " << response.sqlQuery << "\n";
                std::cout << "Confidence: " << (response.confidence * 100) << "%\n\n";
                std::cout << response.result << "\n";
            } else {
                std::cout << "Error: " << response.errorMessage << "\n";
            }
            continue;
        }
        
        if (line.substr(0, 4) == "sql ") {
            std::string sql = line.substr(4);
            std::string result = agent.executeSQL(sql);
            std::cout << result << "\n";
            continue;
        }
        
        if (line.substr(0, 6) == "train ") {
            std::string file = line.substr(6);
            if (agent.trainModel(file)) {
                std::cout << "Model trained successfully!\n";
            } else {
                std::cout << "Training failed!\n";
            }
            continue;
        }
        
        if (line.substr(0, 8) == "offline ") {
            std::string opt = line.substr(8);
            if (opt == "on") {
                agent.setAllowOfflineSQL(true);
                std::cout << "Offline SQL generation ENABLED.\n";
            } else if (opt == "off") {
                agent.setAllowOfflineSQL(false);
                std::cout << "Offline SQL generation DISABLED.\n";
            } else {
                std::cout << "Usage: offline <on|off>\n";
            }
            continue;
        }
        
        if (line.substr(0, 7) == "format ") {
            std::string format = line.substr(7);
            if (format == "table") {
                agent.setOutputFormat(ResponseParser::OutputFormat::TABLE);
                std::cout << "Output format set to TABLE\n";
            } else if (format == "json") {
                agent.setOutputFormat(ResponseParser::OutputFormat::JSON);
                std::cout << "Output format set to JSON\n";
            } else if (format == "csv") {
                agent.setOutputFormat(ResponseParser::OutputFormat::CSV);
                std::cout << "Output format set to CSV\n";
            } else if (format == "plain") {
                agent.setOutputFormat(ResponseParser::OutputFormat::PLAIN);
                std::cout << "Output format set to PLAIN\n";
            } else {
                std::cout << "Unknown format. Use: table, json, csv, or plain\n";
            }
            continue;
        }
        
        std::cout << "Unknown command. Type 'help' for available commands.\n";
    }
    
    std::cout << "\nGoodbye!\n";
    return 0;
}