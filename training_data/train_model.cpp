#include "src/ml/ModelTrainer.h"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
	std::cout << "=== Training NL to SQL Model (from training_data) ===" << std::endl;
	
	MLModelTrainer trainer;

	// CLI: train_model_data [epochs] [lr] [--resume] [--cpu|--cuda|--auto]
	int epochs = 50;
	float lr = 0.001f;
	bool resume = false;
	enum class DeviceMode { Auto, CPU, CUDA } device_mode = DeviceMode::Auto;
	if (argc >= 2) {
		epochs = std::stoi(argv[1]);
	}
	if (argc >= 3) {
		lr = std::stof(argv[2]);
	}
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "--resume") {
			resume = true;
		}
		if (std::string(argv[i]) == "--cpu") {
			device_mode = DeviceMode::CPU;
		}
		if (std::string(argv[i]) == "--cuda") {
			device_mode = DeviceMode::CUDA;
		}
		if (std::string(argv[i]) == "--auto") {
			device_mode = DeviceMode::Auto;
		}
	}

	if (device_mode == DeviceMode::CPU) {
		trainer.setDevice(torch::kCPU);
	} else if (device_mode == DeviceMode::CUDA) {
		trainer.setDevice(torch::kCUDA);
	}

	std::cout << "Training device: "
			  << (trainer.device().is_cuda() ? "CUDA" : "CPU") << std::endl;

	std::cout << "Loading dataset..." << std::endl;
	if (!trainer.loadDataset("training_data/nl_to_sql_train.json")) {
		std::cerr << "Failed to load dataset" << std::endl;
		return 1;
	}

	if (resume) {
		std::cout << "Resuming from checkpoint: models/seq2seq_model*" << std::endl;
		if (!trainer.load("models/seq2seq_model")) {
			std::cerr << "Failed to load existing checkpoint, starting fresh" << std::endl;
		}
	}

	std::cout << "Starting training for " << epochs << " epochs, lr=" << lr << std::endl;
	if (!trainer.train(epochs, lr)) {
		std::cerr << "Training failed" << std::endl;
		return 1;
	}

	std::cout << "Saving model..." << std::endl;
	if (!trainer.save("models/seq2seq_model")) {
		std::cerr << "Failed to save model" << std::endl;
		return 1;
	}

	std::cout << "Training completed successfully!" << std::endl;

	// Quick demo predictions
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

