#include "src/ml/ModelTrainer.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Training NL to SQL Model (CPU-only) ===" << std::endl;
    
    MLModelTrainer trainer;
    
    std::cout << "Loading dataset..." << std::endl;
    if (!trainer.loadDataset("training_data/nl_to_sql_train.json")) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }
    
    std::cout << "Starting training..." << std::endl;
    if (!trainer.train(50, 0.001f)) {
        std::cerr << "Training failed" << std::endl;
        return 1;
    }
    
    std::cout << "Saving model..." << std::endl;
    if (!trainer.save("models/seq2seq_model")) {
        std::cerr << "Failed to save model" << std::endl;
        return 1;
    }
    
    std::cout << "Training completed successfully!" << std::endl;
    
    // Test predictions
    std::cout << "\n=== Testing Predictions ===" << std::endl;
    std::vector<std::string> test_queries = {
        "Show all users",
        "Get 5 cheapest products",
        "Count all orders"
    };
    
    for (const auto& query : test_queries) {
        std::string sql = trainer.predict(query);
        std::cout << "NL: " << query << std::endl;
        std::cout << "SQL: " << sql << std::endl << std::endl;
    }
    
    return 0;
}