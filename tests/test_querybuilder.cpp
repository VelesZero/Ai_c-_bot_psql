#include "../src/core/QueryBuilder.h"
#include "../src/utils/Logger.h"
#include <iostream>
#include <cassert>
#include <string>

void testSelectQuery() {
    Logger::getInstance().info("Testing SELECT query building...");
    
    QueryBuilder qb;
    
    // Test basic SELECT query
    std::string query = qb.buildSelectQuery("users");
    assert(query.find("SELECT * FROM users") != std::string::npos);
    
    // Test SELECT with specific columns
    query = qb.buildSelectQuery("users", {"id", "name", "email"});
    assert(query.find("SELECT id, name, email FROM users") != std::string::npos);
    
    // Test SELECT with WHERE clause
    query = qb.buildSelectQuery("users", {}, "age > 18");
    assert(query.find("SELECT * FROM users WHERE age > 18") != std::string::npos);
    
    // Test SELECT with ORDER BY
    query = qb.buildSelectQuery("users", {}, "", "name ASC");
    assert(query.find("SELECT * FROM users ORDER BY name ASC") != std::string::npos);
    
    // Test SELECT with LIMIT
    query = qb.buildSelectQuery("users", {}, "", "", 10);
    assert(query.find("SELECT * FROM users LIMIT 10") != std::string::npos);
    
    std::cout << "SELECT query test completed" << std::endl;
}

void testInsertQuery() {
    Logger::getInstance().info("Testing INSERT query building...");
    
    QueryBuilder qb;
    
    // Test INSERT query
    std::map<std::string, std::string> values = {
        {"name", "John Doe"},
        {"email", "john@example.com"},
        {"age", "25"}
    };
    
    std::string query = qb.buildInsertQuery("users", values);
    assert(query.find("INSERT INTO users") != std::string::npos);
    assert(query.find("name") != std::string::npos);
    assert(query.find("email") != std::string::npos);
    assert(query.find("age") != std::string::npos);
    
    std::cout << "INSERT query test completed" << std::endl;
}

void testUpdateQuery() {
    Logger::getInstance().info("Testing UPDATE query building...");
    
    QueryBuilder qb;
    
    std::map<std::string, std::string> values = {
        {"name", "Jane Doe"},
        {"email", "jane@example.com"}
    };
    
    std::string query = qb.buildUpdateQuery("users", values, "id = 1");
    assert(query.find("UPDATE users SET") != std::string::npos);
    assert(query.find("name = 'Jane Doe'") != std::string::npos);
    assert(query.find("WHERE id = 1") != std::string::npos);
    
    std::cout << "UPDATE query test completed" << std::endl;
}

void testDeleteQuery() {
    Logger::getInstance().info("Testing DELETE query building...");
    
    QueryBuilder qb;
    
    std::string query = qb.buildDeleteQuery("users", "id = 1");
    assert(query.find("DELETE FROM users WHERE id = 1") != std::string::npos);
    
    std::cout << "DELETE query test completed" << std::endl;
}

int main() {
    Logger::getInstance().setLogLevel(LogLevel::DEBUG);
    
    try {
        testSelectQuery();
        testInsertQuery();
        testUpdateQuery();
        testDeleteQuery();
        
        std::cout << "All query builder tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
