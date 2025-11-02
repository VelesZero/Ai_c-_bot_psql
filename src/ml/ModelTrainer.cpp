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
        auto nl_tokens = nl_vocab_.encode(example.nl_query);
        auto sql_tokens = sql_vocab_.encode(example.sql_query);
        
        for (const auto& token : nl_tokens) {
            nl_vocab_.addWord(nl_vocab_.getWord(token));
        }
        
        for (const auto& token : sql_tokens) {
            sql_vocab_.addWord(sql_vocab_.getWord(token));
        }
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
    
    auto criterion = torch::nn::CrossEntropyLoss();
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        
        for (const auto& example : dataset_) {
            auto [src, trg] = prepareData(example);
            
            optimizer.zero_grad();
            
            auto output = model_->forward(src, trg);
            
            // Reshape for loss calculation
            output = output[1].view({-1, output.size(2)});
            trg = trg[1].view({-1});
            
            auto loss = criterion(output, trg);
            loss.backward();
            optimizer.step();
            
            total_loss += loss.item<float>();
        }
        
        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                      << " - Loss: " << (total_loss / dataset_.size()) << std::endl;
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
