from experiment_utils.baselines.lstm import OneStepPredictionDataset
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch import optim
import math
import seaborn as sns
import matplotlib.pyplot as plt
from einops import rearrange
from rspnlib.evaluation import (
    compute_metrics,
    compute_regression_metrics,
    get_empty_metrics,
    add_metrics,
    print_aggregate_metrics,
    evaluate_rspn_one_step_ahead_prediction,
    evaluate_rspn_observation_regression  
)


class TransformerPredictor(nn.Module):
    def __init__(self, input_size, d_model=32, hidden_size=64, output_size=12, dropout=0.1, max_len=6):
        super(TransformerPredictor, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.embedder = nn.Linear(input_size, d_model, dropout)
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=hidden_size, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        #self.decoder = nn.Linear(d_model, 20, dropout)
        self.output_layer = nn.Linear(d_model, output_size, dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        src = self.relu(self.embedder(inputs)) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.encoder(src)
        #output = self.decoder(output[0])
        #output = self.relu(output)
        output = self.output_layer(output[0])  
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len=6):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x) 

def train_eval_transformer(
    train_dataset,
    test_dataset,
    num_inputs,
    num_observations,
    seed,
    transformer_hidden_size=64,
    d_model=32,
    num_epochs=50,
    continuous=False, 
    action_prediction=False
):
    """
    Trains a Transformer on the train dataset and evaluates it on the test dataset

    :param train_dataset: The train dataset
    :param test_dataset: The test dataset
    :param num_inputs: The number of inputs
    :param num_observations: The number of observations if discrete or their dimension if continuous
    :param seed: The seed to use for reproducibility
    :param num_epochs: The number of epochs to train for
    :param transformer_hidden_size: The hidden size of the LSTM
    :param continuous: Whether the observations are continuous
    """
    print(
        f"——————————[ Running Transformer {'Classification' if not continuous else 'Regression'} evaluation for seed {seed} ]——————————"
    )
    print("Training Transformer...")
    torch.manual_seed(seed)
    inpr_train = train_dataset.inputs_raw
    obsr_train = (train_dataset.observations_raw_discrete) if (not continuous) else (train_dataset.observations_raw_continuous)
    inpr_test = test_dataset.inputs_raw
    obsr_test = (test_dataset.observations_raw_discrete) if (not continuous) else test_dataset.observations_raw_continuous
    
    obsr_train = obsr_train.reshape((-1, obsr_train.shape[1]*obsr_train.shape[2]))
    obsr_test = obsr_test.reshape((-1, obsr_test.shape[1]*obsr_test.shape[2]))

    train_set = OneStepPredictionDataset(inpr_train, obsr_train, num_inputs, num_observations, continuous=continuous, action_prediction=action_prediction)
    test_set = OneStepPredictionDataset(inpr_test, obsr_test, num_inputs, num_observations, continuous=continuous, action_prediction=action_prediction)

    train_data_loader = data.DataLoader(train_set, batch_size=128, num_workers=0, shuffle=True)
    test_data_loader = data.DataLoader(test_set, batch_size=128, num_workers=0, shuffle=True)
    model = TransformerPredictor(next(iter(train_data_loader))[0].shape[2], d_model=d_model, hidden_size=transformer_hidden_size, 
                                 output_size = 2 if action_prediction else num_observations, dropout=0.1, max_len=next(iter(train_data_loader))[0].shape[0])
    model = model.double()

    loss_fn = nn.CrossEntropyLoss() if not continuous else nn.MSELoss()

    lr = 0.001
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 1.0, gamma=0.99)
    losses = []
    DEVICE = 'cpu'
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_data_loader):
            labels_cpu = labels
            inputs = rearrange(inputs, 'b s n -> s b n').to(DEVICE)
            predictions = model(inputs)
            
            if not continuous:
                labels = labels.long().to(DEVICE) 
                labels = labels.flatten()
            else:
                labels = labels.to(DEVICE)
            
            loss = loss_fn(predictions, labels)
            epoch_loss += loss.to("cpu").item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(epoch_loss)

    sns.lineplot(x=range(0, num_epochs), y=losses)
    plt.show()
    print("Training complete.")
    
    print("Evaluating...")
    model.eval()
    all_labels = []
    all_predictions = []
    if not continuous:
        all_argmax_predictions = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data_loader):
            inputs = rearrange(inputs, 'b s n -> s b n').to(DEVICE)

            predictions = model(inputs)
            if not continuous:
                argmax_predictions = torch.argmax(predictions, dim=-1)
                predictions = torch.nn.functional.softmax(predictions, dim=-1).to("cpu")

                all_labels.extend(labels.long().tolist())
            else:
                all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())
            if not continuous:
                all_argmax_predictions.extend(argmax_predictions.tolist())

    if not continuous:
        metrics = compute_metrics(
            all_labels,
            all_argmax_predictions,
            all_predictions,
            is_binary=action_prediction,
        )
        return all_labels, all_predictions, all_argmax_predictions, metrics
    else:
        metrics = compute_regression_metrics(all_labels, all_predictions)
        return all_labels, all_predictions, metrics
