#include "../src/core/Agent.h"
#include "../src/nlprocessor/NLProcessor.h"
#include "../src/utils/Logger.h"
#include <iostream>
#include <cassert>
#include <string>

void testNLProcessor() {
    Logger::getInstance().info("Testing NL Processor...");
    
    NLProcessor processor;
    
    // Test initialization
    assert(processor.initialize());
    
    // Test text processing
    NLProcessingResult result = processor.processNaturalLanguage("Show me all users");
    assert(result.success);
    assert(result.intent == "select");
    
    // Test tokenization
    std::vector<std::string> tokens = processor.tokenize("Show me all users");
    assert(!tokens.empty());
    assert(std::find(tokens.begin(), tokens.end(), "users") != tokens.end());
    
    // Test normalization
    std::string normalized = processor.normalizeText("SHOW ME ALL USERS");
    assert(normalized == "show me all users");
    
    std::cout << "NL Processor test completed" << std::endl;
}

void testAgent() {
    Logger::getInstance().info("Testing Agent...");
    
    Agent agent;
    
    // Test initialization
    assert(agent.initialize());
    
    // Test query processing (will be basic since no real database)
    agent.processQuery("Show me all users");
    std::string response = agent.generateResponse();
    assert(!response.empty());
    
    std::cout << "Agent test completed" << std::endl;
}

void testIntegration() {
    Logger::getInstance().info("Testing full integration...");
    
    // Test that all components work together
    NLProcessor processor;
    assert(processor.initialize());
    
    Agent agent;
    assert(agent.initialize());
    
    // Process a natural language query through the full pipeline
    NLProcessingResult nlResult = processor.processNaturalLanguage("Show me all users");
    assert(nlResult.success);
    
    agent.processQuery("Show me all users");
    std::string response = agent.generateResponse();
    assert(!response.empty());
    
    std::cout << "Integration test completed" << std::endl;
}

int main() {
    Logger::getInstance().setLogLevel(LogLevel::DEBUG);
    
    try {
        testNLProcessor();
        testAgent();
        testIntegration();
        
        std::cout << "All integration tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Integration test failed: " << e.what() << std::endl;
        return 1;
    }
}
