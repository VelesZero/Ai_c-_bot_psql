#include "src/ml/ModelTrainer.h"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    std::cout << "=== Training NL to SQL Model ===" << std::endl;
    
    MLModelTrainer trainer;
    
    // CLI: train_model [epochs] [lr] [--resume] [--cpu|--cuda|--auto] [--data PATH] [--out PREFIX]
    //      [--tiny500k] [--emb N] [--hid N] [--batch N] [--nl-vocab N] [--sql-vocab N]
    //      [--dropout F] [--beam N] [--tf-start F] [--tf-end F]
    int epochs = 50;
    float lr = 0.001f;
    bool resume = false;
    enum class DeviceMode { Auto, CPU, CUDA } device_mode = DeviceMode::Auto;
    std::string data_path = "training_data/nl_to_sql_train.json";
    std::string out_prefix = "models/seq2seq_model";
    bool tiny500k = false;
    int emb_dim = -1;
    int hid_dim = -1;
    int batch_size = 64;
    int nl_vocab_max = 0;
    int sql_vocab_max = 0;
    float dropout = 0.1f;
    int beam_width = 3;
    float tf_start = 1.0f;
    float tf_end = 0.5f;

    auto is_number = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        size_t i = 0;
        if (s[0] == '+' || s[0] == '-') i = 1;
        bool any_digit = false;
        bool any_dot = false;
        for (; i < s.size(); ++i) {
            const char c = s[i];
            if (c >= '0' && c <= '9') {
                any_digit = true;
                continue;
            }
            if (c == '.' && !any_dot) {
                any_dot = true;
                continue;
            }
            return false;
        }
        return any_digit;
    };

    // Backward-compatible positional args: train_model <epochs> <lr>
    if (argc >= 2 && is_number(argv[1])) {
        epochs = std::stoi(argv[1]);
    }
    if (argc >= 3 && is_number(argv[2])) {
        lr = std::stof(argv[2]);
    }

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--resume") {
            resume = true;
        }
        if (std::string(argv[i]) == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
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
        if (std::string(argv[i]) == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        }
        if (std::string(argv[i]) == "--out" && i + 1 < argc) {
            out_prefix = argv[++i];
        }
        if (std::string(argv[i]) == "--tiny500k") {
            tiny500k = true;
        }
        if (std::string(argv[i]) == "--emb" && i + 1 < argc) {
            emb_dim = std::stoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--hid" && i + 1 < argc) {
            hid_dim = std::stoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--nl-vocab" && i + 1 < argc) {
            nl_vocab_max = std::stoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--sql-vocab" && i + 1 < argc) {
            sql_vocab_max = std::stoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--dropout" && i + 1 < argc) {
            dropout = std::stof(argv[++i]);
        }
        if (std::string(argv[i]) == "--beam" && i + 1 < argc) {
            beam_width = std::stoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--tf-start" && i + 1 < argc) {
            tf_start = std::stof(argv[++i]);
        }
        if (std::string(argv[i]) == "--tf-end" && i + 1 < argc) {
            tf_end = std::stof(argv[++i]);
        }
    }

    if (device_mode == DeviceMode::CPU) {
        trainer.setDevice(torch::kCPU);
    } else if (device_mode == DeviceMode::CUDA) {
        trainer.setDevice(torch::kCUDA);
    }

    std::cout << "Training device: "
              << (trainer.device().is_cuda() ? "CUDA" : "CPU") << std::endl;

    if (tiny500k) {
        trainer.setModelDims(64, 128);
    }
    if (emb_dim > 0 && hid_dim > 0) {
        trainer.setModelDims(emb_dim, hid_dim);
    }

    trainer.setBatchSize(batch_size);
    trainer.setVocabLimits(nl_vocab_max, sql_vocab_max);
    trainer.setDropout(dropout);
    trainer.setBeamWidth(beam_width);
    trainer.setTeacherForcingRatio(tf_start, tf_end);
    
    std::cout << "Loading dataset..." << std::endl;
    if (!trainer.loadDataset(data_path)) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }

    if (resume) {
        std::cout << "Resuming from checkpoint: " << out_prefix << "*" << std::endl;
        if (!trainer.load(out_prefix)) {
            std::cerr << "Failed to load existing checkpoint, starting fresh" << std::endl;
        }
    }

    std::cout << "Starting training for " << epochs << " epochs, lr=" << lr << std::endl;
    if (!trainer.train(epochs, lr)) {
        std::cerr << "Training failed" << std::endl;
        return 1;
    }
    
    std::cout << "Saving model..." << std::endl;
    if (!trainer.save(out_prefix)) {
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