from .rnn_vanilla import RNN, LSTM, GRU
from .gpt2_vanilla import GPT2
from .lru_vanilla import LRU
from .mlp_vanilla import MLP


SEQ_MODELS = {
    RNN.name: RNN, 
    LSTM.name: LSTM, 
    GRU.name: GRU, 
    GPT2.name: GPT2,
    LRU.name: LRU,
    MLP.name: MLP,
}
