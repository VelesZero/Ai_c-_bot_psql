#ifndef SEQ2SEQ_MODEL_H
#define SEQ2SEQ_MODEL_H

#include <torch/torch.h>
#include <string>
#include <vector>

class EncoderImpl : public torch::nn::Module {
public:
    EncoderImpl(int vocab_size, int embedding_dim, int hidden_dim);
    
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input);
    
private:
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::LSTM lstm_{nullptr};
    int hidden_dim_;
};
TORCH_MODULE(Encoder);

class DecoderImpl : public torch::nn::Module {
public:
    DecoderImpl(int vocab_size, int embedding_dim, int hidden_dim);
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor input, 
        torch::Tensor hidden, 
        torch::Tensor cell
    );
    
private:
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Linear fc_{nullptr};
    int hidden_dim_;
};
TORCH_MODULE(Decoder);

class Seq2SeqModel {
public:
    Seq2SeqModel(int input_vocab_size, int output_vocab_size, 
                 int embedding_dim = 256, int hidden_dim = 512);
    
    torch::Tensor forward(torch::Tensor src, torch::Tensor trg);
    std::vector<int> predict(const std::vector<int>& input, int max_length = 50);
    
    void train();
    void eval();
    
    bool save(const std::string& path);
    bool load(const std::string& path);
    
    Encoder encoder_;
    Decoder decoder_;
    torch::Device device_;
};

#endif
