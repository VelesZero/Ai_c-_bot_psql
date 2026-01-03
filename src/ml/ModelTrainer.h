#ifndef ML_MODEL_TRAINER_H
#define ML_MODEL_TRAINER_H

#include "Seq2SeqModel.h"
#include "Vocabulary.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <tuple>

struct TrainingExample {
    std::string nl_query;
    std::string sql_query;
};

class MLModelTrainer {
public:
    MLModelTrainer();

    void setDevice(torch::Device device);
    torch::Device device() const;
    
    void setModelDims(int embedding_dim, int hidden_dim);

    bool loadDataset(const std::string& path);
    bool train(int epochs = 100, float learning_rate = 0.001);
    bool save(const std::string& model_path);
    bool load(const std::string& model_path);
    
    std::string predict(const std::string& nl_query);
    
private:
    std::vector<TrainingExample> dataset_;
    Vocabulary nl_vocab_;
    Vocabulary sql_vocab_;
    std::unique_ptr<Seq2SeqModel> model_;
    torch::Device device_;

    int embedding_dim_;
    int hidden_dim_;
    
    void buildVocabularies();
    std::tuple<torch::Tensor, torch::Tensor> prepareData(const TrainingExample& example);
};

#endif
