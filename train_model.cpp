#include "core/Agent.h"
#include "config/Config.h"
#include "utils/Logger.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <training_data.json>\n";
        return 1;
    }
    
    std::string trainingFile = argv[1];
    
    // Настройка логирования
    Logger::getInstance().setLogFile("training.log");
    Logger::getInstance().setLogLevel(LogLevel::INFO);
    
    // Создание агента
    Agent agent;
    
    // Инициализация
    if (!agent.initialize("src/config/default.conf")) {
        std::cerr << "Failed to initialize agent\n";
        return 1;
    }
    
    // Обучение
    std::cout << "Starting model training...\n";
    std::cout << "Training data: " << trainingFile << "\n\n";
    
    if (agent.trainModel(trainingFile)) {
        std::cout << "\n✓ Model trained successfully!\n";
        std::cout << "Model saved to: " << Config::getInstance().getModelPath() << "\n";
        return 0;
    } else {
        std::cerr << "\n✗ Training failed!\n";
        return 1;
    }
}