#include "../src/core/DatabaseConnector.h"
#include "../src/utils/Logger.h"
#include <iostream>
#include <cassert>

void testDatabaseConnection() {
    Logger::getInstance().info("Testing database connection...");
    
    DatabaseConnector db;
    
    // Test connection (will fail without actual database, but tests the interface)
    bool connected = db.connect("test_connection_string");
    assert(!connected); // Should fail with dummy connection string
    
    std::cout << "Database connection test completed" << std::endl;
}

void testDatabaseQuery() {
    Logger::getInstance().info("Testing database query execution...");
    
    DatabaseConnector db;
    
    // Test query execution (will fail without actual database)
    DatabaseResult result = db.executeQuery("SELECT * FROM test_table");
    
    // Since no real database is connected, we expect empty results
    assert(result.columns.empty());
    assert(result.data.empty());
    
    std::cout << "Database query test completed" << std::endl;
}

int main() {
    Logger::getInstance().setLogLevel(LogLevel::DEBUG);
    
    try {
        testDatabaseConnection();
        testDatabaseQuery();
        
        std::cout << "All database tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
