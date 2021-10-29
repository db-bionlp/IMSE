import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
import torch.nn.functional as F
import numpy as np
import math

PRETRAINED_MODEL_MAP = {
    'biobert': BertModel,
    'scibert': BertModel,
    'roberta': RobertaModel,
    'albert': AlbertModel
}

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, args, N_fingerprints, dim, layer_hidden, layer_output, mode, activation):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.args = args
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        if layer_output != 0:
            self.W_output = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                           for _ in range(layer_output)])
            self.W_output_ = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_output)])
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.mode = mode
        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(),
                       'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':GELU()}
        self.activation = activations[activation]

    def pad(self, matrices):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)

        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            pad_matrices = zeros
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n

        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = self.activation(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fingerprints = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:,0,]]
        adjacencies = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:,1,]]
        molecular_sizes = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:,2,]]
        masks = [torch.tensor(x, dtype=torch.float).to(device) for x in inputs[:,3,]]
        masks = torch.cat(masks).unsqueeze(-1)


        #fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies)

        """GNN layer (update the fingerprint_file vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint_file vectors."""
        if self.mode == 'sum':
            molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        elif self.mode == 'mean':
            molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        if self.layer_output != 0:
            for l in self.W_output_:
                molecular_vectors = self.activation(l(molecular_vectors))

        """Mask invalid SMILES vectors"""
        molecular_vectors *= masks

        return molecular_vectors

    def mlp(self, vectors1, vectors2):
        vectors = torch.cat((vectors1, vectors2), 1)
        if self.layer_output != 0:
            for l in self.W_output:
                vectors = torch.relu(l(vectors))
        return vectors

class ddi_Bert(BertPreTrainedModel):
    def __init__(self, config, gnn_config, args):
        super(ddi_Bert, self).__init__(config)

        self.args = args
        self.num_labels = config.num_labels
        self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)

        # only use the BERT model
        if self.args.model == "only_bert":
            self.label_classifier = FCLayer(config.hidden_size, config.num_labels, args.dropout_rate, use_activation=False)

        # use  BERT + Interaction attention vector
        if self.args.model == "bert_int":
            self.label_classifier = FCLayer(config.hidden_size * 2, config.num_labels, args.dropout_rate, use_activation=False)

        # use  BERT + molecular
        if self.args.model == "bert_mol":
            self.gnn = MolecularGraphNeuralNetwork(args, gnn_config.N_fingerprints, gnn_config.dim,
                                                   gnn_config.layer_hidden,
                                                   gnn_config.layer_output, gnn_config.mode, gnn_config.activation)
            self.label_classifier = FCLayer(config.hidden_size + 2 * gnn_config.dim, config.num_labels,
                                            args.dropout_rate,
                                            use_activation=False)

        # use  BERT + Interaction attention vector + molecular
        if self.args.model == "bert_int_mol":
            self.gnn = MolecularGraphNeuralNetwork(args, gnn_config.N_fingerprints, gnn_config.dim, gnn_config.layer_hidden,
                                                   gnn_config.layer_output, gnn_config.mode, gnn_config.activation)
            self.label_classifier = FCLayer(config.hidden_size*2 + 2*gnn_config.dim, config.num_labels, args.dropout_rate,
                                            use_activation=False)

        # use  BERT + Interaction attention + Entities attention  + molecular
        if self.args.model == "bert_int_ent_mol":
            self.gnn = MolecularGraphNeuralNetwork(args, gnn_config.N_fingerprints, gnn_config.dim,
                                                       gnn_config.layer_hidden,
                                                       gnn_config.layer_output, gnn_config.mode, gnn_config.activation)
            self.label_classifier = FCLayer(config.hidden_size * 3 + 2 * gnn_config.dim, config.num_labels,
                                                args.dropout_rate,
                                                use_activation=False)

    @staticmethod
    # Averaging treatment
    def average(hidden_output, list):
        list_unsqueeze = list.unsqueeze(1)
        length_tensor = (list != 0).sum(dim=1).unsqueeze(1)
        sum_vector = torch.bmm(list_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids,
                labels,
                int_list,
                ent_list,
                fingerprint_index,
                fingerprint_data,
                ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.fc_layer(pooled_output)

        # only BERT model used
        if self.args.model == "only_bert":
            logits = self.label_classifier(pooled_output)
            outputs = (logits,) + outputs[2:]

        # use BERT model and Interaction vector
        if self.args.model == 'bert_int':
            int = self.average(sequence_output, int_list)
            int = self.fc_layer(int)
            concat = torch.cat([int, pooled_output, ], dim=-1)
            logits = self.label_classifier(concat)
            outputs = (logits,) + outputs[2:]

        # use  BERT + Molecualr
        if self.args.model == "bert_mol":
            fingerprint = fingerprint_data[fingerprint_index.cpu()]
            if fingerprint.ndim == 3:
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            else:
                fingerprint = np.expand_dims(fingerprint, 0)
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            gnn_output1 = self.gnn.gnn(fingerprint1)
            gnn_output2 = self.gnn.gnn(fingerprint2)
            gnn_output = torch.cat((gnn_output1, gnn_output2), -1)

            concat = torch.cat((pooled_output, gnn_output), -1)

            logits = self.label_classifier(concat)
            outputs = (logits,) + outputs[2:]

        # use  BERT + Intearction vector + Molecular
        if self.args.model == "bert_int_mol":
                fingerprint = fingerprint_data[fingerprint_index.cpu()]
                if fingerprint.ndim == 3:
                    fingerprint1 = fingerprint[:, 0, ]
                    fingerprint2 = fingerprint[:, 1, ]
                else:
                    fingerprint = np.expand_dims(fingerprint, 0)
                    fingerprint1 = fingerprint[:, 0, ]
                    fingerprint2 = fingerprint[:, 1, ]
                gnn_output1 = self.gnn.gnn(fingerprint1)
                gnn_output2 = self.gnn.gnn(fingerprint2)
                gnn_output = torch.cat((gnn_output1, gnn_output2), -1)

                int = self.average(sequence_output, int_list)
                int = self.fc_layer(int)

                concat = torch.cat((pooled_output, gnn_output, int), -1)

                logits = self.label_classifier(concat)

                outputs = (logits,) + outputs[2:]

        # use  BERT + Intearction attention + Entities attention +  Molecular
        if self.args.model == "bert_int_ent_mol":
            fingerprint = fingerprint_data[fingerprint_index.cpu()]
            if fingerprint.ndim == 3:
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            else:
                fingerprint = np.expand_dims(fingerprint, 0)
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            gnn_output1 = self.gnn.gnn(fingerprint1)
            gnn_output2 = self.gnn.gnn(fingerprint2)
            gnn_output = torch.cat((gnn_output1, gnn_output2), -1)

            int = self.average(sequence_output, int_list)
            int = self.fc_layer(int)

            ent = self.average(sequence_output, ent_list)
            ent = self.fc_layer(ent)

            concat = torch.cat((pooled_output, int, ent, gnn_output), -1)

            logits = self.label_classifier(concat)

            outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)






