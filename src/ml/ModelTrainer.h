#ifndef ML_MODEL_TRAINER_H
#define ML_MODEL_TRAINER_H

#include "Seq2SeqModel.h"
#include "Vocabulary.h"
#include <torch/torch.h>
#include <ATen/autocast_mode.h>
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
    void setValSplit(float split);
    void setPatience(int patience);
    void setWeightDecay(float wd);
    void setLabelSmoothing(float ls);
    void setNumLayers(int n);

    void setBatchSize(int batch_size);
    void setGradAccumSteps(int steps);
    void setVocabLimits(int nl_vocab_max, int sql_vocab_max);
    void setCheckpointEvery(int epochs, const std::string& prefix);
    void setAMP(bool enabled);

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
    int grad_accum_steps_ = 1;
    int nl_vocab_max_ = 0;
    int sql_vocab_max_ = 0;

    // Teacher forcing: linearly decays from tf_start_ to tf_end_ over epochs
    float tf_start_ = 1.0f;
    float tf_end_ = 0.0f;

    // Beam search width (0 = greedy)
    int beam_width_ = 3;

    // Validation split and early stopping
    float val_split_ = 0.1f;
    int patience_ = 5;          // 0 = disabled

    // Model depth
    int num_layers_ = 2;

    // Regularization
    float weight_decay_ = 1e-5f;
    float label_smoothing_ = 0.1f;

    // Checkpoint: save every N epochs (0 = disabled)
    int checkpoint_every_ = 0;
    std::string checkpoint_prefix_;

    // Mixed precision (AMP)
    bool use_amp_ = false;

    struct GradScaler {
        float scale = 65536.0f;
        float growth_factor = 2.0f;
        float backoff_factor = 0.5f;
        int growth_interval = 2000;
        int step_count = 0;
    };
    GradScaler grad_scaler_;

    void buildVocabularies();
    void preEncodeDataset();
    std::tuple<torch::Tensor, torch::Tensor> prepareData(const TrainingExample& example);
    std::tuple<torch::Tensor, torch::Tensor> prepareBatch(const std::vector<size_t>& indices);

    // Pre-encoded sequences (filled once, reused every epoch)
    std::vector<std::vector<int>> encoded_nl_;
    std::vector<std::vector<int>> encoded_sql_;
};

#endif
