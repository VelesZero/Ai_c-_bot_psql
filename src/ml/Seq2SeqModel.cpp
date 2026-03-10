#include "Seq2SeqModel.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

// ───────────────── Attention ─────────────────

AttentionImpl::AttentionImpl(int hidden_dim) {
    attn_ = register_module("attn",
        torch::nn::Linear(hidden_dim * 2, hidden_dim));
    v_ = register_module("v",
        torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, 1).bias(false)));
}

std::tuple<torch::Tensor, torch::Tensor> AttentionImpl::forward(
    torch::Tensor hidden, torch::Tensor encoder_outputs) {
    // hidden:          (1, batch, hidden_dim)
    // encoder_outputs: (src_len, batch, hidden_dim)
    int64_t src_len = encoder_outputs.size(0);

    // Expand hidden to (src_len, batch, hidden_dim) - zero-copy view
    auto h = hidden.expand({src_len, -1, -1});

    // energy: (src_len, batch, hidden_dim)
    auto energy = torch::tanh(attn_->forward(torch::cat({h, encoder_outputs}, 2)));

    // attention weights: (src_len, batch)
    auto attention = v_->forward(energy).squeeze(2);
    attention = torch::softmax(attention, /*dim=*/0);

    // context: weighted sum -> (1, batch, hidden_dim)
    auto context = (attention.unsqueeze(2) * encoder_outputs).sum(/*dim=*/0, /*keepdim=*/true);

    return {context, attention};
}

// ───────────────── Encoder ─────────────────

EncoderImpl::EncoderImpl(int vocab_size, int embedding_dim, int hidden_dim,
                         int num_layers, bool bidirectional, float dropout_p)
    : hidden_dim_(hidden_dim), num_layers_(num_layers), bidirectional_(bidirectional) {

    embedding_ = register_module("embedding",
        torch::nn::Embedding(vocab_size, embedding_dim));

    auto lstm_opts = torch::nn::LSTMOptions(embedding_dim, hidden_dim)
        .num_layers(num_layers)
        .bidirectional(bidirectional)
        .dropout(num_layers > 1 ? dropout_p : 0.0);
    lstm_ = register_module("lstm", torch::nn::LSTM(lstm_opts));

    if (bidirectional) {
        // Project bidirectional encoder_outputs (hidden_dim*2) → hidden_dim
        fc_proj_ = register_module("fc_proj",
            torch::nn::Linear(hidden_dim * 2, hidden_dim));
        // Project concatenated fwd+bwd hidden/cell per layer → decoder size
        fc_hidden_ = register_module("fc_hidden",
            torch::nn::Linear(hidden_dim * 2, hidden_dim));
        fc_cell_ = register_module("fc_cell",
            torch::nn::Linear(hidden_dim * 2, hidden_dim));
    }

    dropout_ = register_module("dropout",
        torch::nn::Dropout(dropout_p));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EncoderImpl::forward(
    torch::Tensor input) {
    // input: (src_len, batch)
    auto embedded = dropout_->forward(embedding_->forward(input));
    // embedded: (src_len, batch, emb_dim)

    auto lstm_out = lstm_->forward(embedded);
    auto outputs = std::get<0>(lstm_out);
    // outputs: (src_len, batch, hidden_dim * num_directions)
    auto hidden_tuple = std::get<1>(lstm_out);
    auto hidden = std::get<0>(hidden_tuple);
    // hidden: (num_layers * num_directions, batch, hidden_dim)
    auto cell = std::get<1>(hidden_tuple);

    if (bidirectional_) {
        // Project encoder_outputs: (src_len, batch, hidden_dim*2) → (src_len, batch, hidden_dim)
        outputs = torch::tanh(fc_proj_->forward(outputs));

        // hidden/cell shape: (num_layers*2, batch, hidden_dim)
        // Reshape to (num_layers, 2, batch, hidden_dim), concat fwd+bwd → (num_layers, batch, hidden_dim*2)
        int64_t batch_size = hidden.size(1);
        auto h = hidden.view({num_layers_, 2, batch_size, hidden_dim_});
        auto c = cell.view({num_layers_, 2, batch_size, hidden_dim_});

        // Concatenate forward (idx 0) and backward (idx 1) along hidden dim
        auto h_cat = torch::cat({h.select(1, 0), h.select(1, 1)}, /*dim=*/2);
        // h_cat: (num_layers, batch, hidden_dim*2)
        auto c_cat = torch::cat({c.select(1, 0), c.select(1, 1)}, /*dim=*/2);

        hidden = torch::tanh(fc_hidden_->forward(h_cat));
        // hidden: (num_layers, batch, hidden_dim)
        cell = torch::tanh(fc_cell_->forward(c_cat));
    }

    return {outputs, hidden, cell};
}

// ───────────────── Decoder ─────────────────

DecoderImpl::DecoderImpl(int vocab_size, int embedding_dim, int hidden_dim,
                         int num_layers, float dropout_p)
    : hidden_dim_(hidden_dim), num_layers_(num_layers) {

    embedding_ = register_module("embedding",
        torch::nn::Embedding(vocab_size, embedding_dim));

    attention_ = register_module("attention",
        Attention(hidden_dim));

    // LSTM input: embedding + attention context
    auto lstm_opts = torch::nn::LSTMOptions(embedding_dim + hidden_dim, hidden_dim)
        .num_layers(num_layers)
        .dropout(num_layers > 1 ? dropout_p : 0.0);
    lstm_ = register_module("lstm", torch::nn::LSTM(lstm_opts));

    // Output: LSTM hidden + attention context → vocab
    fc_ = register_module("fc",
        torch::nn::Linear(hidden_dim * 2, vocab_size));

    dropout_ = register_module("dropout",
        torch::nn::Dropout(dropout_p));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
DecoderImpl::forward(torch::Tensor input, torch::Tensor hidden,
                     torch::Tensor cell, torch::Tensor encoder_outputs) {
    // input: (batch,)
    // hidden/cell: (num_layers, batch, hidden_dim)
    auto embedded = dropout_->forward(embedding_->forward(input.unsqueeze(0)));
    // embedded: (1, batch, emb_dim)

    // Attention context from top-layer hidden state
    auto top_hidden = hidden.narrow(0, num_layers_ - 1, 1);
    // top_hidden: (1, batch, hidden_dim)
    auto [context, attn_weights] = attention_->forward(top_hidden, encoder_outputs);
    // context: (1, batch, hidden_dim)

    // Concatenate embedding with context
    auto lstm_input = torch::cat({embedded, context}, /*dim=*/2);
    // lstm_input: (1, batch, emb_dim + hidden_dim)

    auto lstm_out = lstm_->forward(lstm_input, std::make_tuple(hidden, cell));
    auto output = std::get<0>(lstm_out);           // (1, batch, hidden_dim)
    auto hidden_tuple = std::get<1>(lstm_out);
    auto h = std::get<0>(hidden_tuple);            // (num_layers, batch, hidden_dim)
    auto c = std::get<1>(hidden_tuple);

    // Predict: concat LSTM output with context
    auto prediction = fc_->forward(
        torch::cat({output.squeeze(0), context.squeeze(0)}, /*dim=*/1));
    // prediction: (batch, vocab_size)

    return {prediction, h, c, attn_weights};
}

// ───────────────── Seq2SeqModel ─────────────────

Seq2SeqModel::Seq2SeqModel(int input_vocab_size, int output_vocab_size,
                           int embedding_dim, int hidden_dim,
                           int num_layers, float dropout_p,
                           torch::Device device)
    : encoder_(Encoder(input_vocab_size, embedding_dim, hidden_dim,
                       num_layers, /*bidirectional=*/true, dropout_p)),
      decoder_(Decoder(output_vocab_size, embedding_dim, hidden_dim,
                       num_layers, dropout_p)),
      device_(torch::kCPU) {

    to(device);
    std::cout << "Seq2SeqModel: layers=" << num_layers
              << ", bidir_encoder, emb=" << embedding_dim
              << ", hid=" << hidden_dim
              << ", device=" << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
}

void Seq2SeqModel::to(torch::Device device) {
    if (device.is_cuda() && !torch::cuda::is_available()) {
        std::cerr << "CUDA requested but not available. Falling back to CPU." << std::endl;
        device_ = torch::kCPU;
    } else {
        device_ = device;
    }
    encoder_->to(device_);
    decoder_->to(device_);
}

torch::Tensor Seq2SeqModel::forward(torch::Tensor src, torch::Tensor trg,
                                    float teacher_forcing_ratio) {
    src = src.to(device_);
    trg = trg.to(device_);

    int batch_size = static_cast<int>(trg.size(1));
    int trg_len = static_cast<int>(trg.size(0));
    int trg_vocab_size = decoder_->output_vocab_size();

    auto outputs = torch::zeros({trg_len, batch_size, trg_vocab_size}).to(device_);

    // Encoder
    auto [encoder_outputs, hidden, cell] = encoder_->forward(src);

    auto input = trg[0]; // SOS tokens

    for (int t = 1; t < trg_len; t++) {
        auto [output, hidden_new, cell_new, _attn] =
            decoder_->forward(input, hidden, cell, encoder_outputs);

        outputs[t] = output;
        hidden = hidden_new;
        cell = cell_new;

        // Teacher forcing: use ground truth or model prediction
        bool use_teacher = (static_cast<float>(rand()) / RAND_MAX) < teacher_forcing_ratio;
        input = use_teacher ? trg[t] : output.argmax(1);
    }

    return outputs;
}

PredictionResult Seq2SeqModel::predict(const std::vector<int>& input, int max_length) {
    const bool enc_was_training = encoder_->is_training();
    const bool dec_was_training = decoder_->is_training();
    encoder_->eval();
    decoder_->eval();

    torch::NoGradGuard no_grad;

    auto src = torch::tensor(input).unsqueeze(1).to(device_);
    auto [encoder_outputs, hidden, cell] = encoder_->forward(src);

    PredictionResult result;
    float total_log_prob = 0.0f;
    int current_token = 1; // SOS_TOKEN

    for (int i = 0; i < max_length; i++) {
        auto input_tensor = torch::tensor({current_token}).to(device_);
        auto [output, hidden_new, cell_new, _attn] =
            decoder_->forward(input_tensor, hidden, cell, encoder_outputs);

        hidden = hidden_new;
        cell = cell_new;

        // Compute probability of chosen token
        auto probs = torch::softmax(output, /*dim=*/1);
        auto max_result = probs.max(/*dim=*/1);
        float max_prob = std::get<0>(max_result).item<float>();
        current_token = std::get<1>(max_result).item<int>();

        if (current_token == 2) { // EOS_TOKEN
            break;
        }

        result.tokens.push_back(current_token);
        total_log_prob += std::log(max_prob + 1e-10f);
    }

    // Geometric mean of per-token probabilities
    if (!result.tokens.empty()) {
        result.confidence = std::exp(total_log_prob / static_cast<float>(result.tokens.size()));
    } else {
        result.confidence = 0.0f;
    }

    if (enc_was_training) encoder_->train();
    if (dec_was_training) decoder_->train();

    return result;
}

PredictionResult Seq2SeqModel::beam_search(const std::vector<int>& input,
                                           int beam_width, int max_length) {
    const bool enc_was_training = encoder_->is_training();
    const bool dec_was_training = decoder_->is_training();
    encoder_->eval();
    decoder_->eval();

    torch::NoGradGuard no_grad;

    auto src = torch::tensor(input).unsqueeze(1).to(device_);
    auto [encoder_outputs, hidden, cell] = encoder_->forward(src);

    struct BeamState {
        std::vector<int> tokens;
        float score;            // cumulative log-probability
        torch::Tensor hidden;
        torch::Tensor cell;
        bool finished;
    };

    std::vector<BeamState> beams;
    beams.push_back({{}, 0.0f, hidden, cell, false});

    for (int step = 0; step < max_length; step++) {
        std::vector<BeamState> candidates;

        for (auto& beam : beams) {
            if (beam.finished) {
                candidates.push_back(beam);
                continue;
            }

            int last_token = beam.tokens.empty() ? 1 : beam.tokens.back(); // SOS or last
            auto input_tensor = torch::tensor({last_token}).to(device_);

            auto [output, h_new, c_new, _attn] =
                decoder_->forward(input_tensor, beam.hidden, beam.cell, encoder_outputs);

            auto log_probs = torch::log_softmax(output, /*dim=*/1).squeeze(0);

            // Top-k candidates
            auto topk = log_probs.topk(beam_width);
            auto topk_values = std::get<0>(topk);
            auto topk_indices = std::get<1>(topk);

            for (int k = 0; k < beam_width; k++) {
                int token = topk_indices[k].item<int>();
                float token_score = topk_values[k].item<float>();

                BeamState new_beam;
                new_beam.tokens = beam.tokens;
                new_beam.score = beam.score + token_score;
                new_beam.hidden = h_new;
                new_beam.cell = c_new;

                if (token == 2) { // EOS
                    new_beam.finished = true;
                } else {
                    new_beam.tokens.push_back(token);
                    new_beam.finished = false;
                }

                candidates.push_back(std::move(new_beam));
            }
        }

        // Keep top beam_width candidates
        std::sort(candidates.begin(), candidates.end(),
                  [](const BeamState& a, const BeamState& b) {
                      // Length-normalized score
                      float sa = a.tokens.empty() ? a.score : a.score / static_cast<float>(a.tokens.size());
                      float sb = b.tokens.empty() ? b.score : b.score / static_cast<float>(b.tokens.size());
                      return sa > sb;
                  });

        if (static_cast<int>(candidates.size()) > beam_width) {
            candidates.resize(static_cast<size_t>(beam_width));
        }

        beams = std::move(candidates);

        // Check if all beams finished
        bool all_done = true;
        for (const auto& b : beams) {
            if (!b.finished) { all_done = false; break; }
        }
        if (all_done) break;
    }

    // Pick best beam
    PredictionResult result;
    if (!beams.empty()) {
        const auto& best = beams[0];
        result.tokens = best.tokens;
        if (!result.tokens.empty()) {
            result.confidence = std::exp(best.score / static_cast<float>(result.tokens.size()));
        } else {
            result.confidence = 0.0f;
        }
    }

    if (enc_was_training) encoder_->train();
    if (dec_was_training) decoder_->train();

    return result;
}

void Seq2SeqModel::train() {
    encoder_->train();
    decoder_->train();
}

void Seq2SeqModel::eval() {
    encoder_->eval();
    decoder_->eval();
}

bool Seq2SeqModel::save(const std::string& path) {
    try {
        torch::save(encoder_, path + "_encoder.pt");
        torch::save(decoder_, path + "_decoder.pt");
        return true;
    } catch (...) {
        return false;
    }
}

bool Seq2SeqModel::load(const std::string& path) {
    try {
        torch::load(encoder_, path + "_encoder.pt");
        torch::load(decoder_, path + "_decoder.pt");
        encoder_->to(device_);
        decoder_->to(device_);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Seq2SeqModel::load failed: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Seq2SeqModel::load failed: unknown error" << std::endl;
        return false;
    }
}
