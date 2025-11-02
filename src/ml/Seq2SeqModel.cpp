#include "Seq2SeqModel.h"

EncoderImpl::EncoderImpl(int vocab_size, int embedding_dim, int hidden_dim)
    : hidden_dim_(hidden_dim) {
    
    embedding_ = register_module("embedding", 
        torch::nn::Embedding(vocab_size, embedding_dim));
    
    lstm_ = register_module("lstm", 
        torch::nn::LSTM(torch::nn::LSTMOptions(embedding_dim, hidden_dim)));
}

std::tuple<torch::Tensor, torch::Tensor> EncoderImpl::forward(torch::Tensor input) {
    auto embedded = embedding_->forward(input);
    auto lstm_out = lstm_->forward(embedded);
    auto hidden_tuple = std::get<1>(lstm_out);
    return std::make_tuple(std::get<0>(hidden_tuple), std::get<1>(hidden_tuple));
}

DecoderImpl::DecoderImpl(int vocab_size, int embedding_dim, int hidden_dim)
    : hidden_dim_(hidden_dim) {
    
    embedding_ = register_module("embedding", 
        torch::nn::Embedding(vocab_size, embedding_dim));
    
    lstm_ = register_module("lstm", 
        torch::nn::LSTM(torch::nn::LSTMOptions(embedding_dim, hidden_dim)));
    
    fc_ = register_module("fc", 
        torch::nn::Linear(hidden_dim, vocab_size));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> DecoderImpl::forward(
    torch::Tensor input, torch::Tensor hidden, torch::Tensor cell) {
    
    auto embedded = embedding_->forward(input.unsqueeze(0));
    
    // hidden and cell are expected to be of shape (num_layers, batch, hidden_dim)
    // Encoder already returns (1, batch, hidden_dim), so we pass them directly
    auto lstm_out = lstm_->forward(embedded, std::make_tuple(hidden, cell));
    
    auto output = std::get<0>(lstm_out);
    auto prediction = fc_->forward(output.squeeze(0));
    
    auto hidden_tuple = std::get<1>(lstm_out);
    return std::make_tuple(prediction, 
                          std::get<0>(hidden_tuple).squeeze(0), 
                          std::get<1>(hidden_tuple).squeeze(0));
}

Seq2SeqModel::Seq2SeqModel(int input_vocab_size, int output_vocab_size, 
                           int embedding_dim, int hidden_dim)
    : encoder_(Encoder(input_vocab_size, embedding_dim, hidden_dim)),
      decoder_(Decoder(output_vocab_size, embedding_dim, hidden_dim)),
      device_(torch::kCPU) {
    
    if (torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        encoder_->to(device_);
        decoder_->to(device_);
    }
}

torch::Tensor Seq2SeqModel::forward(torch::Tensor src, torch::Tensor trg) {
    int batch_size = trg.size(1);
    int trg_len = trg.size(0);
    int trg_vocab_size = decoder_->parameters()[2].size(0);
    
    auto outputs = torch::zeros({trg_len, batch_size, trg_vocab_size}).to(device_);
    
    auto [hidden, cell] = encoder_->forward(src);
    
    auto input = trg[0];
    
    for (int t = 1; t < trg_len; t++) {
        auto [output, hidden_new, cell_new] = decoder_->forward(input, hidden, cell);
        outputs[t] = output;
        hidden = hidden_new;
        cell = cell_new;
        input = trg[t];
    }
    
    return outputs;
}

std::vector<int> Seq2SeqModel::predict(const std::vector<int>& input, int max_length) {
    encoder_->eval();
    decoder_->eval();
    
    torch::NoGradGuard no_grad;
    
    auto src = torch::tensor(input).unsqueeze(1).to(device_);
    auto [hidden, cell] = encoder_->forward(src);
    
    std::vector<int> result;
    int current_token = 1; // SOS_TOKEN
    
    for (int i = 0; i < max_length; i++) {
        auto input_tensor = torch::tensor({current_token}).to(device_);
        auto [output, hidden_new, cell_new] = decoder_->forward(input_tensor, hidden, cell);
        
        hidden = hidden_new;
        cell = cell_new;
        
        current_token = output.argmax(1).item<int>();
        
        if (current_token == 2) { // EOS_TOKEN
            break;
        }
        
        result.push_back(current_token);
    }
    
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
        return true;
    } catch (...) {
        return false;
    }
}
