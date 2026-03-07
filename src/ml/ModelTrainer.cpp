#include "ModelTrainer.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <cmath>
#include <iomanip>

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

void MLModelTrainer::setDropout(float dropout_p) {
    dropout_p_ = std::clamp(dropout_p, 0.0f, 0.9f);
}

void MLModelTrainer::setTeacherForcingRatio(float start, float end) {
    tf_start_ = std::clamp(start, 0.0f, 1.0f);
    tf_end_ = std::clamp(end, 0.0f, 1.0f);
}

void MLModelTrainer::setBeamWidth(int width) {
    beam_width_ = std::max(0, width);
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
    preEncodeDataset();

    model_ = std::make_unique<Seq2SeqModel>(
        nl_vocab_.size(),
        sql_vocab_.size(),
        embedding_dim_,
        hidden_dim_,
        dropout_p_,
        device_
    );

    return true;
}

void MLModelTrainer::buildVocabularies() {
    for (const auto& example : dataset_) {
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

void MLModelTrainer::preEncodeDataset() {
    const size_t n = dataset_.size();
    encoded_nl_.resize(n);
    encoded_sql_.resize(n);

    std::cout << "Pre-encoding " << n << " examples..." << std::flush;
    for (size_t i = 0; i < n; ++i) {
        encoded_nl_[i] = nl_vocab_.encode(dataset_[i].nl_query);
        encoded_sql_[i] = sql_vocab_.encode(dataset_[i].sql_query);
    }
    std::cout << " done." << std::endl;
}

std::tuple<torch::Tensor, torch::Tensor> MLModelTrainer::prepareData(
    const TrainingExample& example) {

    auto nl_indices = nl_vocab_.encode(example.nl_query);
    auto sql_indices = sql_vocab_.encode(example.sql_query);

    auto src = torch::tensor(nl_indices, torch::kLong).unsqueeze(1);
    auto trg = torch::tensor(sql_indices, torch::kLong).unsqueeze(1);

    return std::make_tuple(src, trg);
}

std::tuple<torch::Tensor, torch::Tensor> MLModelTrainer::prepareBatch(
    const std::vector<size_t>& indices) {

    int64_t max_src_len = 0;
    int64_t max_trg_len = 0;

    for (size_t idx : indices) {
        max_src_len = std::max<int64_t>(max_src_len, static_cast<int64_t>(encoded_nl_[idx].size()));
        max_trg_len = std::max<int64_t>(max_trg_len, static_cast<int64_t>(encoded_sql_[idx].size()));
    }

    const int64_t batch = static_cast<int64_t>(indices.size());

    // Pre-fill with PAD tokens
    auto src = torch::full({max_src_len, batch}, Vocabulary::PAD_TOKEN, torch::kLong);
    auto trg = torch::full({max_trg_len, batch}, Vocabulary::PAD_TOKEN, torch::kLong);

    // Fill using accessor for better performance than index_put_
    auto src_acc = src.accessor<int64_t, 2>();
    auto trg_acc = trg.accessor<int64_t, 2>();

    for (int64_t b = 0; b < batch; ++b) {
        const auto& s = encoded_nl_[indices[static_cast<size_t>(b)]];
        const auto& t = encoded_sql_[indices[static_cast<size_t>(b)]];
        for (int64_t i = 0; i < static_cast<int64_t>(s.size()); ++i) {
            src_acc[i][b] = s[static_cast<size_t>(i)];
        }
        for (int64_t i = 0; i < static_cast<int64_t>(t.size()); ++i) {
            trg_acc[i][b] = t[static_cast<size_t>(i)];
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

    // Collect all parameters properly before creating optimizer
    std::vector<torch::Tensor> all_params;
    for (const auto& p : model_->encoder_->parameters()) {
        all_params.push_back(p);
    }
    for (const auto& p : model_->decoder_->parameters()) {
        all_params.push_back(p);
    }

    torch::optim::Adam optimizer(all_params, torch::optim::AdamOptions(learning_rate));

    auto criterion = torch::nn::CrossEntropyLoss(
        torch::nn::CrossEntropyLossOptions().ignore_index(Vocabulary::PAD_TOKEN)
    );
    criterion->to(model_->device_);

    const int64_t total_examples = static_cast<int64_t>(dataset_.size());
    const int64_t bs = std::max<int64_t>(1, static_cast<int64_t>(batch_size_));
    const int64_t total_batches = (total_examples + bs - 1) / bs;
    const int64_t log_interval = std::max<int64_t>(1, total_batches / 200);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int64_t batch_count = 0;
        auto epoch_start = std::chrono::steady_clock::now();

        // Teacher forcing ratio: linear decay from tf_start_ to tf_end_
        float tf_ratio = tf_start_;
        if (epochs > 1) {
            tf_ratio = tf_start_ - (tf_start_ - tf_end_) *
                       static_cast<float>(epoch) / static_cast<float>(epochs - 1);
        }

        int bar_width = 50;
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs
                  << " (tf=" << tf_ratio << ") [";

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

            auto outputs = model_->forward(src, trg, tf_ratio);
            auto outputs_slice = outputs.slice(0, 1);
            auto trg_slice = trg.slice(0, 1);

            auto logits = outputs_slice.reshape({-1, outputs.size(2)});
            auto targets = trg_slice.reshape({-1}).to(model_->device_);

            auto loss = criterion(logits, targets);
            loss.backward();

            // Gradient clipping to prevent exploding gradients
            torch::nn::utils::clip_grad_norm_(all_params, 1.0);

            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;

            if (b % log_interval == 0 || b == total_batches - 1) {
                float progress = static_cast<float>(b + 1) / static_cast<float>(total_batches);
                int pos = static_cast<int>(bar_width * progress);

                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - epoch_start).count();
                int64_t eta = (progress > 0.001f)
                    ? static_cast<int64_t>(static_cast<float>(elapsed) / progress * (1.0f - progress))
                    : 0;
                float batches_per_sec = (elapsed > 0) ? static_cast<float>(b + 1) / static_cast<float>(elapsed) : 0;

                std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs
                          << " (tf=" << tf_ratio << ") [";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress * 100.0) << "%"
                          << " " << (b + 1) << "/" << total_batches
                          << " [" << elapsed << "s<" << eta << "s"
                          << ", " << std::fixed << std::setprecision(1) << batches_per_sec << " bat/s]"
                          << "   " << std::flush;
            }
        }

        auto epoch_end = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        float avg_loss = (batch_count > 0) ? (total_loss / static_cast<float>(batch_count)) : 0.0f;

        std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs
                  << " (tf=" << tf_ratio << ") [";
        for (int i = 0; i < bar_width; ++i) std::cout << "=";
        std::cout << "] " << "100% " << "(" << epoch_duration.count() << "s)"
                  << " Loss: " << avg_loss << std::endl;

        // Show example predictions every 5 epochs
        if ((epoch + 1) % 5 == 0 && !dataset_.empty()) {
            std::cout << "\nExample predictions after epoch " << (epoch + 1) << ":\n";
            for (int i = 0; i < std::min(2, static_cast<int>(dataset_.size())); i++) {
                const auto& ex = dataset_[static_cast<size_t>(i)];
                auto pred = predictWithConfidence(ex.nl_query);
                std::cout << "  NL: " << ex.nl_query << "\n";
                std::cout << "  Pred SQL: " << pred.sql
                          << " (confidence: " << (pred.confidence * 100) << "%)\n";
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
        dropout_p_,
        device_
    );

    return model_->load(model_path);
}

std::string MLModelTrainer::predict(const std::string& nl_query) {
    return predictWithConfidence(nl_query).sql;
}

PredictResult MLModelTrainer::predictWithConfidence(const std::string& nl_query) {
    PredictResult result;
    result.confidence = 0.0f;

    if (!model_) {
        return result;
    }

    torch::NoGradGuard no_grad;
    model_->eval();

    auto nl_indices = nl_vocab_.encode(nl_query);

    PredictionResult pred;
    if (beam_width_ > 1) {
        pred = model_->beam_search(nl_indices, beam_width_);
    } else {
        pred = model_->predict(nl_indices);
    }

    result.sql = sql_vocab_.decode(pred.tokens);
    result.confidence = pred.confidence;

    return result;
}
