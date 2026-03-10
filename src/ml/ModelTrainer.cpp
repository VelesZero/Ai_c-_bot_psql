#include "ModelTrainer.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
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

void MLModelTrainer::setGradAccumSteps(int steps) {
    grad_accum_steps_ = std::max(1, steps);
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

void MLModelTrainer::setValSplit(float split) {
    val_split_ = std::clamp(split, 0.0f, 0.5f);
}

void MLModelTrainer::setPatience(int patience) {
    patience_ = std::max(0, patience);
}

void MLModelTrainer::setWeightDecay(float wd) {
    weight_decay_ = std::max(0.0f, wd);
}

void MLModelTrainer::setLabelSmoothing(float ls) {
    label_smoothing_ = std::clamp(ls, 0.0f, 0.5f);
}

void MLModelTrainer::setNumLayers(int n) {
    num_layers_ = std::max(1, n);
}

void MLModelTrainer::setBeamWidth(int width) {
    beam_width_ = std::max(0, width);
}

void MLModelTrainer::setCheckpointEvery(int epochs, const std::string& prefix) {
    checkpoint_every_ = std::max(0, epochs);
    checkpoint_prefix_ = prefix;
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
        num_layers_,
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

    // --- Train / validation split ---
    const size_t n = dataset_.size();
    std::vector<size_t> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    {
        std::mt19937 split_rng(42);
        std::shuffle(all_indices.begin(), all_indices.end(), split_rng);
    }
    const size_t val_size = static_cast<size_t>(n * val_split_);
    std::vector<size_t> val_indices(all_indices.begin(), all_indices.begin() + val_size);
    std::vector<size_t> train_indices(all_indices.begin() + val_size, all_indices.end());

    if (val_size > 0) {
        std::cout << "Train: " << train_indices.size()
                  << " | Val: " << val_size << " examples\n";
    }

    // --- Collect parameters ---
    std::vector<torch::Tensor> all_params;
    for (const auto& p : model_->encoder_->parameters()) all_params.push_back(p);
    for (const auto& p : model_->decoder_->parameters()) all_params.push_back(p);

    torch::optim::Adam optimizer(all_params,
        torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay_));

    auto criterion = torch::nn::CrossEntropyLoss(
        torch::nn::CrossEntropyLossOptions()
            .ignore_index(Vocabulary::PAD_TOKEN)
            .label_smoothing(label_smoothing_)
    );
    criterion->to(model_->device_);

    const int64_t total_train = static_cast<int64_t>(train_indices.size());
    const int64_t bs = std::max<int64_t>(1, static_cast<int64_t>(batch_size_));
    const int64_t total_batches = (total_train + bs - 1) / bs;
    const int64_t log_interval = std::max<int64_t>(1, total_batches / 200);
    const int bar_width = 50;

    // --- Early stopping & LR scheduling state ---
    float best_val_loss = std::numeric_limits<float>::infinity();
    int patience_counter = 0;
    int lr_patience_counter = 0;
    const int lr_patience = 2;
    const float lr_factor = 0.5f;
    float current_lr = learning_rate;

    // --- LR warmup: linear ramp over first 5% of batches in epoch 0 ---
    const int64_t warmup_steps = std::max<int64_t>(1, total_batches * 5 / 100);

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Teacher forcing ratio: linear decay from tf_start_ to tf_end_
        float tf_ratio = tf_start_;
        if (epochs > 1) {
            tf_ratio = tf_start_ - (tf_start_ - tf_end_) *
                       static_cast<float>(epoch) / static_cast<float>(epochs - 1);
        }

        // --- Training loop ---
        model_->train();
        torch::Tensor loss_accum = torch::zeros({}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
        int64_t batch_count = 0;
        auto epoch_start = std::chrono::steady_clock::now();

        std::vector<size_t> shuffled_train = train_indices;
        {
            std::mt19937 rng(static_cast<std::mt19937::result_type>(epoch + 1337));
            std::shuffle(shuffled_train.begin(), shuffled_train.end(), rng);
        }

        const int64_t accum_steps = std::max<int64_t>(1, static_cast<int64_t>(grad_accum_steps_));

        for (int64_t b = 0; b < total_batches; ++b) {
            const int64_t start = b * bs;
            const int64_t end = std::min<int64_t>(start + bs, total_train);
            std::vector<size_t> batch_indices;
            batch_indices.reserve(static_cast<size_t>(end - start));
            for (int64_t i = start; i < end; ++i) {
                batch_indices.push_back(shuffled_train[static_cast<size_t>(i)]);
            }

            auto [src, trg] = prepareBatch(batch_indices);

            // LR warmup during first epoch
            if (epoch == 0 && b < warmup_steps) {
                float warmup_lr = learning_rate * static_cast<float>(b + 1) / static_cast<float>(warmup_steps);
                for (auto& pg : optimizer.param_groups()) {
                    static_cast<torch::optim::AdamOptions&>(pg.options()).lr(warmup_lr);
                }
                if (b == warmup_steps - 1) {
                    current_lr = learning_rate;
                }
            }

            if (b % accum_steps == 0) optimizer.zero_grad();

            auto outputs = model_->forward(src, trg, tf_ratio);
            auto logits = outputs.slice(0, 1).reshape({-1, outputs.size(2)});
            auto targets = trg.slice(0, 1).reshape({-1}).to(model_->device_);

            auto loss = criterion(logits, targets) / static_cast<float>(accum_steps);
            loss.backward();

            const bool is_last_batch = (b == total_batches - 1);
            if (((b + 1) % accum_steps == 0) || is_last_batch) {
                torch::nn::utils::clip_grad_norm_(all_params, 1.0);
                optimizer.step();
            }

            loss_accum.add_((loss * static_cast<float>(accum_steps)).detach());
            batch_count++;

            if (b % log_interval == 0 || b == total_batches - 1) {
                float progress = static_cast<float>(b + 1) / static_cast<float>(total_batches);
                int pos = static_cast<int>(bar_width * progress);
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - epoch_start).count();
                int64_t eta = (progress > 0.001f)
                    ? static_cast<int64_t>(static_cast<float>(elapsed) / progress * (1.0f - progress)) : 0;
                float bps = (elapsed > 0) ? static_cast<float>(b + 1) / static_cast<float>(elapsed) : 0;

                std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs
                          << " (tf=" << std::fixed << std::setprecision(2) << tf_ratio << ") [";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress * 100.0) << "%"
                          << " " << (b + 1) << "/" << total_batches
                          << " [" << elapsed << "s<" << eta << "s"
                          << ", " << std::fixed << std::setprecision(1) << bps << " bat/s]   "
                          << std::flush;
            }
        }

        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - epoch_start);
        float train_loss = (batch_count > 0)
            ? (loss_accum.item<float>() / static_cast<float>(batch_count)) : 0.0f;

        // --- Validation loop ---
        float val_loss = 0.0f;
        if (!val_indices.empty()) {
            model_->eval();
            torch::NoGradGuard no_grad;
            const int64_t val_total = static_cast<int64_t>(val_indices.size());
            const int64_t val_batches = (val_total + bs - 1) / bs;
            torch::Tensor val_accum = torch::zeros({}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
            int64_t val_count = 0;

            for (int64_t b = 0; b < val_batches; ++b) {
                const int64_t start = b * bs;
                const int64_t end = std::min<int64_t>(start + bs, val_total);
                std::vector<size_t> batch_idx;
                batch_idx.reserve(static_cast<size_t>(end - start));
                for (int64_t i = start; i < end; ++i)
                    batch_idx.push_back(val_indices[static_cast<size_t>(i)]);

                auto [src, trg] = prepareBatch(batch_idx);
                auto outputs = model_->forward(src, trg, 1.0f);
                auto logits = outputs.slice(0, 1).reshape({-1, outputs.size(2)});
                auto targets = trg.slice(0, 1).reshape({-1}).to(model_->device_);
                val_accum.add_(criterion(logits, targets).detach());
                val_count++;
            }
            val_loss = (val_count > 0) ? (val_accum.item<float>() / static_cast<float>(val_count)) : 0.0f;
        }

        // --- Epoch summary ---
        std::cout << "\rEpoch " << (epoch + 1) << "/" << epochs
                  << " (tf=" << std::fixed << std::setprecision(2) << tf_ratio << ") [";
        for (int i = 0; i < bar_width; ++i) std::cout << "=";
        std::cout << "] 100% (" << epoch_duration.count() << "s)"
                  << " | train: " << std::fixed << std::setprecision(4) << train_loss;
        if (!val_indices.empty()) {
            std::cout << " | val: " << val_loss;
        }
        std::cout << "\n";

        // --- Early stopping & LR scheduling (based on val loss) ---
        if (!val_indices.empty()) {
            if (val_loss < best_val_loss - 1e-4f) {
                best_val_loss = val_loss;
                patience_counter = 0;
                lr_patience_counter = 0;
            } else {
                ++patience_counter;
                ++lr_patience_counter;
                if (lr_patience_counter >= lr_patience) {
                    current_lr *= lr_factor;
                    for (auto& pg : optimizer.param_groups()) {
                        static_cast<torch::optim::AdamOptions&>(pg.options()).lr(current_lr);
                    }
                    lr_patience_counter = 0;
                    std::cout << "  [LR reduced to " << current_lr << "]\n";
                }
                if (patience_ > 0 && patience_counter >= patience_) {
                    std::cout << "  [Early stopping at epoch " << (epoch + 1) << "]\n";
                    break;
                }
            }
        }

        // --- Periodic checkpoint ---
        if (checkpoint_every_ > 0 && !checkpoint_prefix_.empty() &&
            (epoch + 1) % checkpoint_every_ == 0) {
            std::string ckpt = checkpoint_prefix_ + "_epoch" + std::to_string(epoch + 1);
            if (save(ckpt)) std::cout << "  [checkpoint saved: " << ckpt << "]\n";
            else            std::cerr << "  [checkpoint save failed]\n";
        }

        // --- Example predictions every 5 epochs ---
        // BUG FIX: predictWithConfidence() calls model_->eval(), restore train mode after
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
            model_->train();  // restore train mode (predictWithConfidence sets eval)
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
        num_layers_,
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
