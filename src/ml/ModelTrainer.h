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

struct PredictResult {
    std::string sql;
    float confidence;
};

class MLModelTrainer {
public:
    MLModelTrainer();

    void setDevice(torch::Device device);
    torch::Device device() const;

    void setModelDims(int embedding_dim, int hidden_dim);
    void setDropout(float dropout_p);
    void setTeacherForcingRatio(float start, float end);
    void setBeamWidth(int width);

    void setBatchSize(int batch_size);
    void setVocabLimits(int nl_vocab_max, int sql_vocab_max);

    bool loadDataset(const std::string& path);
    bool train(int epochs = 100, float learning_rate = 0.001);
    bool save(const std::string& model_path);
    bool load(const std::string& model_path);

    std::string predict(const std::string& nl_query);
    PredictResult predictWithConfidence(const std::string& nl_query);

private:
    std::vector<TrainingExample> dataset_;
    Vocabulary nl_vocab_;
    Vocabulary sql_vocab_;
    std::unique_ptr<Seq2SeqModel> model_;
    torch::Device device_;

    int embedding_dim_;
    int hidden_dim_;
    float dropout_p_ = 0.1f;
    int batch_size_ = 32;
    int nl_vocab_max_ = 0;
    int sql_vocab_max_ = 0;

    // Teacher forcing: linearly decays from tf_start_ to tf_end_ over epochs
    float tf_start_ = 1.0f;
    float tf_end_ = 0.5f;

    // Beam search width (0 = greedy)
    int beam_width_ = 3;

    void buildVocabularies();
    void preEncodeDataset();
    std::tuple<torch::Tensor, torch::Tensor> prepareData(const TrainingExample& example);
    std::tuple<torch::Tensor, torch::Tensor> prepareBatch(const std::vector<size_t>& indices);

    // Pre-encoded sequences (filled once, reused every epoch)
    std::vector<std::vector<int>> encoded_nl_;
    std::vector<std::vector<int>> encoded_sql_;
};

#endif
