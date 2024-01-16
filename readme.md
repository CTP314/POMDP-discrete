## Installation

We use python 3.7+ and list the basic requirements in [`requirements.txt`](https://github.com/twni2016/Memory-RL/blob/main/requirements.txt).

## Reproducing the Results

Below are example commands to reproduce the *main* results shown in Figure 3 and 6.
For the ablation results, please adjust the corresponding hyperparameters.

To run Passive T-Maze with a memory length of 50 with LSTM-based agent:

```bash
python main.py \
    --config_env configs/envs/tmaze_passive.py \
    --config_env.env_name 50 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 20000 \
    --config_seq configs/seq_models/lstm_default.py \
    --config_seq.sampled_seq_len -1 \
```

To run Passive T-Maze with a memory length of 1500 with Transformer-based agent:

```bash
python main.py \
    --config_env configs/envs/tmaze_passive.py \
    --config_env.env_name 1500 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 6700 \
    --config_seq configs/seq_models/gpt_default.py \
    --config_seq.sampled_seq_len -1 \
```

To run Active T-Maze with a memory length of 20 with Transformer-based agent:

```bash
python main.py \
    --config_env configs/envs/tmaze_active.py \
    --config_env.env_name 20 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 40000 \
    --config_seq configs/seq_models/gpt_default.py \
    --config_seq.sampled_seq_len -1 \
    --config_seq.model.seq_model_config.n_layer 2 \
    --config_seq.model.seq_model_config.n_head 2 \
```

To run Passive Visual Match with a memory length of 60 with Transformer-based agent:

```bash
python main.py \
    --config_env configs/envs/visual_match.py \
    --config_env.env_name 60 \
    --config_rl configs/rl/sacd_default.py \
    --shared_encoder --freeze_critic \
    --train_episodes 40000 \
    --config_seq configs/seq_models/gpt_cnn.py \
    --config_seq.sampled_seq_len -1 \
```

To run Key-to-Door with a memory length of 120 with LSTM-based agent:

```bash
python main.py \
    --config_env configs/envs/keytodoor.py \
    --config_env.env_name 120 \
    --config_rl configs/rl/sacd_default.py \
    --shared_encoder --freeze_critic \
    --train_episodes 40000 \
    --config_seq configs/seq_models/lstm_cnn.py \
    --config_seq.sampled_seq_len -1 \
    --config_seq.model.seq_model_config.n_layer 2 \
```

To run Key-to-Door with a memory length of 250 with Transformer-based agent:

```bash
python main.py \
    --config_env configs/envs/visual_match.py \
    --config_env.env_name 250 \
    --config_rl configs/rl/sacd_default.py \
    --shared_encoder --freeze_critic \
    --train_episodes 30000 \
    --config_seq configs/seq_models/gpt_cnn.py \
    --config_seq.sampled_seq_len -1 \
    --config_seq.model.seq_model_config.n_layer 2 \
    --config_seq.model.seq_model_config.n_head 2 \
```

The `train_episodes` of each task is specified in [`budget.py`](https://github.com/twni2016/Memory-RL/blob/main/budget.py).

By default, the logging data will be stored in `logs/` folder with csv format. If you use `--debug` flag, it will be stored in `debug/` folder.

To run Regular with a memory length of 50 with LRU-based agent:

```
python main.py \
        --config_env configs/envs/parity_pomdp.py \
        --config_env.env_name 50 \
        --config_rl configs/rl/dqn_default.py \
        --train_episodes 20000 \
        --config_seq configs/seq_models/lru_default.py \
        --config_seq.sampled_seq_len -1 \
        --config_seq.model.action_embedder.hidden_size=0 \
        --config_seq.model.observ_embedder.hidden_size=128 \
        --config_rl.config_critic.hidden_dims="()"\
```

## Logging and Plotting

After the logging data is stored, you can plot the learning curves and aggregation plots (e.g., Figure 3 and 6) using [`vis.ipynb`](https://github.com/twni2016/Memory-RL/blob/main/vis.ipynb) jupyter notebook.

We also provide our logging data used in the paper shared in [google drive](https://drive.google.com/file/d/1bX8lRtm6IYihCmATzgVU7Enq4xuSFAVq/view?usp=sharing) (< 400 MB).

## 添加正规语言对应的 POMDP

在 envs/regular 里从 regular base 中按照 DFA 的方式构造正则语言，之后可以用 Regula rPOMDP 直接转化为 gym 的 env。
