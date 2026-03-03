#ifndef SEQ2SEQ_MODEL_H
#define SEQ2SEQ_MODEL_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <tuple>

// Bahdanau (additive) attention mechanism
class AttentionImpl : public torch::nn::Module {
public:
    AttentionImpl(int hidden_dim);

    // hidden:          (1, batch, hidden_dim)
    // encoder_outputs: (src_len, batch, hidden_dim)
    // returns: context (1, batch, hidden_dim), weights (src_len, batch)
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor hidden, torch::Tensor encoder_outputs);

private:
    torch::nn::Linear attn_{nullptr};
    torch::nn::Linear v_{nullptr};
};
TORCH_MODULE(Attention);

// Encoder: bidirectional-ready single-layer LSTM with dropout
class EncoderImpl : public torch::nn::Module {
public:
    EncoderImpl(int vocab_size, int embedding_dim, int hidden_dim,
                float dropout_p = 0.1);

    // Returns (encoder_outputs, hidden, cell)
    //   encoder_outputs: (src_len, batch, hidden_dim)
    //   hidden/cell:     (1, batch, hidden_dim)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor input);

private:
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    int hidden_dim_;
};
TORCH_MODULE(Encoder);

// Decoder: LSTM + attention + dropout
class DecoderImpl : public torch::nn::Module {
public:
    DecoderImpl(int vocab_size, int embedding_dim, int hidden_dim,
                float dropout_p = 0.1);

    // input:           (batch,)
    // hidden/cell:     (1, batch, hidden_dim)
    // encoder_outputs: (src_len, batch, hidden_dim)
    // Returns (prediction, hidden, cell, attention_weights)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor input, torch::Tensor hidden, torch::Tensor cell,
        torch::Tensor encoder_outputs);

    int output_vocab_size() const { return static_cast<int>(fc_->options.out_features()); }

private:
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::LSTM lstm_{nullptr};
    Attention attention_{nullptr};
    torch::nn::Linear fc_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    int hidden_dim_;
};
TORCH_MODULE(Decoder);

// Prediction result with confidence
struct PredictionResult {
    std::vector<int> tokens;
    float confidence;   // geometric mean of per-step probabilities
};

class Seq2SeqModel {
public:
    Seq2SeqModel(int input_vocab_size, int output_vocab_size,
                 int embedding_dim = 256, int hidden_dim = 512,
                 float dropout_p = 0.1,
                 torch::Device device = torch::kCPU);

    // Training forward with teacher forcing
    torch::Tensor forward(torch::Tensor src, torch::Tensor trg,
                          float teacher_forcing_ratio = 1.0f);

    // Greedy decode with confidence
    PredictionResult predict(const std::vector<int>& input, int max_length = 50);

    // Beam search decode
    PredictionResult beam_search(const std::vector<int>& input,
                                 int beam_width = 3, int max_length = 50);

    void train();
    void eval();

    bool save(const std::string& path);
    bool load(const std::string& path);

    void to(torch::Device device);

    Encoder encoder_;
    Decoder decoder_;
    torch::Device device_;
};

#endif
