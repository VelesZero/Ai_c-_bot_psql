#include "ModelTrainer.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

using json = nlohmann::json;

MLModelTrainer::MLModelTrainer()
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      embedding_dim_(256),
      hidden_dim_(512) {}

void MLModelTrainer::setBatchSize(int batch_size) {
    if (batch_size <= 0) {
        std::cerr << "Invalid batch size: " << batch_size << std::endl;
        return;
    }
    batch_size_ = batch_size;
}

void MLModelTrainer::setVocabLimits(int nl_vocab_max, int sql_vocab_max) {
    nl_vocab_max_ = nl_vocab_max;
    sql_vocab_max_ = sql_vocab_max;
}

void MLModelTrainer::setModelDims(int embedding_dim, int hidden_dim) {
    if (embedding_dim <= 0 || hidden_dim <= 0) {
        std::cerr << "Invalid model dims: embedding_dim=" << embedding_dim
                  << ", hidden_dim=" << hidden_dim << std::endl;
        return;
    }
    embedding_dim_ = embedding_dim;
    hidden_dim_ = hidden_dim;
}

void MLModelTrainer::setDevice(torch::Device device) {
    if (device.is_cuda() && !torch::cuda::is_available()) {
        std::cerr << "CUDA requested but not available. Falling back to CPU." << std::endl;
        device_ = torch::kCPU;
    } else {
        device_ = device;
    }

    if (model_) {
        model_->to(device_);
    }
}

torch::Device MLModelTrainer::device() const {
    return device_;
}

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
        embedding_dim_,
        hidden_dim_,
        device_
    );
    
    return true;
}

void MLModelTrainer::buildVocabularies() {
    for (const auto& example : dataset_) {
        // Learn vocabulary from raw sentences
        nl_vocab_.addSentence(example.nl_query);
        sql_vocab_.addSentence(example.sql_query);
    }

    if (nl_vocab_max_ > 0) {
        nl_vocab_.limitSize(nl_vocab_max_);
    }
    if (sql_vocab_max_ > 0) {
        sql_vocab_.limitSize(sql_vocab_max_);
    }
    
    std::cout << "NL Vocabulary size: " << nl_vocab_.size() << std::endl;
    std::cout << "SQL Vocabulary size: " << sql_vocab_.size() << std::endl;
}

std::tuple<torch::Tensor, torch::Tensor> MLModelTrainer::prepareData(
    const TrainingExample& example) {
    
    auto nl_indices = nl_vocab_.encode(example.nl_query);
    auto sql_indices = sql_vocab_.encode(example.sql_query);
    
    auto src = torch::tensor(nl_indices, torch::kLong).unsqueeze(1);
    auto trg = torch::tensor(sql_indices, torch::kLong).unsqueeze(1);
    
    return std::make_tuple(src, trg);
}

std::tuple<torch::Tensor, torch::Tensor> MLModelTrainer::prepareBatch(const std::vector<size_t>& indices) {
    // Build variable-length sequences for a batch, then pad to max length.
    std::vector<std::vector<int>> src_seqs;
    std::vector<std::vector<int>> trg_seqs;
    src_seqs.reserve(indices.size());
    trg_seqs.reserve(indices.size());

    int64_t max_src_len = 0;
    int64_t max_trg_len = 0;

    for (size_t idx : indices) {
        const auto& ex = dataset_[idx];
        src_seqs.push_back(nl_vocab_.encode(ex.nl_query));
        trg_seqs.push_back(sql_vocab_.encode(ex.sql_query));

        max_src_len = std::max<int64_t>(max_src_len, static_cast<int64_t>(src_seqs.back().size()));
        max_trg_len = std::max<int64_t>(max_trg_len, static_cast<int64_t>(trg_seqs.back().size()));
    }

    const int64_t batch = static_cast<int64_t>(indices.size());

    auto src = torch::full({max_src_len, batch}, Vocabulary::PAD_TOKEN, torch::kLong);
    auto trg = torch::full({max_trg_len, batch}, Vocabulary::PAD_TOKEN, torch::kLong);

    for (int64_t b = 0; b < batch; ++b) {
        const auto& s = src_seqs[static_cast<size_t>(b)];
        const auto& t = trg_seqs[static_cast<size_t>(b)];
        for (int64_t i = 0; i < static_cast<int64_t>(s.size()); ++i) {
            src.index_put_({i, b}, s[static_cast<size_t>(i)]);
        }
        for (int64_t i = 0; i < static_cast<int64_t>(t.size()); ++i) {
            trg.index_put_({i, b}, t[static_cast<size_t>(i)]);
        }
    }

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
    // Ensure the loss computation happens on the same device as the model (CPU/GPU)
    criterion->to(model_->device_);
    
    const int64_t total_examples = static_cast<int64_t>(dataset_.size());
    const int64_t bs = std::max<int64_t>(1, static_cast<int64_t>(batch_size_));
    const int64_t total_batches = (total_examples + bs - 1) / bs;
    const int64_t log_interval = std::max<int64_t>(1, total_batches / 50); // ~50 updates per epoch
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int64_t batch_count = 0;
        auto epoch_start = std::chrono::steady_clock::now();
        
        // Progress bar header
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs << " [";
        int bar_width = 50;
        
        std::vector<size_t> order(static_cast<size_t>(total_examples));
        std::iota(order.begin(), order.end(), static_cast<size_t>(0));
        std::mt19937 rng(static_cast<std::mt19937::result_type>(epoch + 1337));
        std::shuffle(order.begin(), order.end(), rng);

        for (int64_t b = 0; b < total_batches; ++b) {
            const int64_t start = b * bs;
            const int64_t end = std::min<int64_t>(start + bs, total_examples);
            std::vector<size_t> batch_indices;
            batch_indices.reserve(static_cast<size_t>(end - start));
            for (int64_t i = start; i < end; ++i) {
                batch_indices.push_back(order[static_cast<size_t>(i)]);
            }

            auto [src, trg] = prepareBatch(batch_indices);

            optimizer.zero_grad();

            auto outputs = model_->forward(src, trg);
            auto outputs_slice = outputs.slice(0, 1);
            auto trg_slice = trg.slice(0, 1);

            auto logits = outputs_slice.reshape({-1, outputs.size(2)});
            auto targets = trg_slice.reshape({-1}).to(model_->device_);

            auto loss = criterion(logits, targets);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;

            if (b % log_interval == 0 || b == total_batches - 1) {
                float progress = static_cast<float>(b + 1) / static_cast<float>(total_batches);
                int pos = static_cast<int>(bar_width * progress);
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
        float avg_loss = (batch_count > 0) ? (total_loss / static_cast<float>(batch_count)) : 0.0f;
        
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs << " [";
        for (int i = 0; i < bar_width; ++i) std::cout << "=";
        std::cout << "] " << "100% " << "(" << epoch_duration.count() << "s)"
                  << " Loss: " << avg_loss << std::endl;
        
        // Show example predictions every 5 epochs
        if ((epoch + 1) % 5 == 0 && !dataset_.empty()) {
            std::cout << "\nExample predictions after epoch " << (epoch + 1) << ":\n";
            for (int i = 0; i < std::min(2, static_cast<int>(dataset_.size())); i++) {
                const auto& ex = dataset_[static_cast<size_t>(i)];
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
        sql_vocab_.size(),
        embedding_dim_,
        hidden_dim_,
        device_
    );
    
    return model_->load(model_path);
}

std::string MLModelTrainer::predict(const std::string& nl_query) {
    if (!model_) {
        return "";
    }

    const bool was_training = model_->encoder_->is_training() && model_->decoder_->is_training();

    torch::NoGradGuard no_grad;
    model_->eval();

    auto nl_indices = nl_vocab_.encode(nl_query);
    auto sql_indices = model_->predict(nl_indices);

    if (was_training) {
        model_->train();
    }

    return sql_vocab_.decode(sql_indices);
}
