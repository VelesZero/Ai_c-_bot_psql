#include "ModelTrainer.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

MLModelTrainer::MLModelTrainer() {}

bool MLModelTrainer::loadDataset(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open dataset: " << path << std::endl;
        return false;
    }
    
    json data;
    file >> data;
    
    dataset_.clear();
    for (const auto& example : data["examples"]) {
        TrainingExample ex;
        ex.nl_query = example["nl"];
        ex.sql_query = example["sql"];
        dataset_.push_back(ex);
    }
    
    std::cout << "Loaded " << dataset_.size() << " examples" << std::endl;
    
    buildVocabularies();
    
    model_ = std::make_unique<Seq2SeqModel>(
        nl_vocab_.size(), 
        sql_vocab_.size(), 
        256,  // embedding_dim
        512   // hidden_dim
    );
    
    return true;
}

void MLModelTrainer::buildVocabularies() {
    for (const auto& example : dataset_) {
        // Learn vocabulary from raw sentences
        nl_vocab_.addSentence(example.nl_query);
        sql_vocab_.addSentence(example.sql_query);
    }
    
    std::cout << "NL Vocabulary size: " << nl_vocab_.size() << std::endl;
    std::cout << "SQL Vocabulary size: " << sql_vocab_.size() << std::endl;
}

std::tuple<torch::Tensor, torch::Tensor> MLModelTrainer::prepareData(
    const TrainingExample& example) {
    
    auto nl_indices = nl_vocab_.encode(example.nl_query);
    auto sql_indices = sql_vocab_.encode(example.sql_query);
    
    auto src = torch::tensor(nl_indices).unsqueeze(1);
    auto trg = torch::tensor(sql_indices).unsqueeze(1);
    
    return std::make_tuple(src, trg);
}

bool MLModelTrainer::train(int epochs, float learning_rate) {
    if (!model_) {
        std::cerr << "Model not initialized" << std::endl;
        return false;
    }
    
    model_->train();
    
    torch::optim::Adam optimizer(
        std::vector<torch::Tensor>{},
        torch::optim::AdamOptions(learning_rate)
    );
    
    // Add encoder and decoder parameters
    for (const auto& p : model_->encoder_->parameters()) {
        optimizer.param_groups()[0].params().push_back(p);
    }
    for (const auto& p : model_->decoder_->parameters()) {
        optimizer.param_groups()[0].params().push_back(p);
    }
    
    auto criterion = torch::nn::CrossEntropyLoss(
        torch::nn::CrossEntropyLossOptions().ignore_index(Vocabulary::PAD_TOKEN)
    );
    
    const int total_examples = dataset_.size();
    const int log_interval = std::max(1, total_examples / 50);  // Show ~50 progress steps per epoch
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        int example_count = 0;
        auto epoch_start = std::chrono::steady_clock::now();
        
        // Progress bar header
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs << " [";
        int bar_width = 50;
        
        for (int i = 0; i < dataset_.size(); ++i) {
            const auto& example = dataset_[i];
            auto [src, trg] = prepareData(example);
            
            optimizer.zero_grad();
            
            auto outputs = model_->forward(src, trg);
            auto outputs_slice = outputs.slice(0, 1);
            auto trg_slice = trg.slice(0, 1);
            
            auto logits = outputs_slice.reshape({-1, outputs.size(2)});
            auto targets = trg_slice.reshape({-1});
            
            auto loss = criterion(logits, targets);
            loss.backward();
            optimizer.step();
            
            total_loss += loss.item<float>();
            example_count++;
            
            // Update progress bar
            if (i % log_interval == 0 || i == total_examples - 1) {
                float progress = static_cast<float>(i + 1) / total_examples;
                int pos = bar_width * progress;
                std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs << " [";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress * 100.0) << "%" << std::flush;
            }
        }
        
        // Epoch summary
        auto epoch_end = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        float avg_loss = total_loss / example_count;
        
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs << " [";
        for (int i = 0; i < bar_width; ++i) std::cout << "=";
        std::cout << "] " << "100% " << "(" << epoch_duration.count() << "s)"
                  << " Loss: " << avg_loss << std::endl;
        
        // Show example predictions every 5 epochs
        if ((epoch + 1) % 5 == 0 && !dataset_.empty()) {
            std::cout << "\nExample predictions after epoch " << (epoch + 1) << ":\n";
            for (int i = 0; i < std::min(2, static_cast<int>(dataset_.size())); i++) {
                const auto& ex = dataset_[i];
                auto pred = predict(ex.nl_query);
                std::cout << "  NL: " << ex.nl_query << "\n";
                std::cout << "  Pred SQL: " << pred << "\n";
                std::cout << "  True SQL: " << ex.sql_query << "\n\n";
            }
        }
    }
    
    return true;
}

bool MLModelTrainer::save(const std::string& model_path) {
    if (!model_->save(model_path)) {
        return false;
    }
    
    nl_vocab_.save(model_path + "_nl_vocab.txt");
    sql_vocab_.save(model_path + "_sql_vocab.txt");
    
    return true;
}

bool MLModelTrainer::load(const std::string& model_path) {
    if (!nl_vocab_.load(model_path + "_nl_vocab.txt") ||
        !sql_vocab_.load(model_path + "_sql_vocab.txt")) {
        return false;
    }
    
    model_ = std::make_unique<Seq2SeqModel>(
        nl_vocab_.size(), 
        sql_vocab_.size()
    );
    
    return model_->load(model_path);
}

std::string MLModelTrainer::predict(const std::string& nl_query) {
    if (!model_) {
        return "";
    }
    
    model_->eval();
    
    auto nl_indices = nl_vocab_.encode(nl_query);
    auto sql_indices = model_->predict(nl_indices);
    
    return sql_vocab_.decode(sql_indices);
}
