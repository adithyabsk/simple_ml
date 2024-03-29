# noqa: E402
# TODO: fix this noqa
import operator
from functools import reduce

import numpy as np

import simple_ml as sm
import simple_ml.nn as nn

np.random.seed(0)


class ConvBN(sm.nn.Module):
    conv_layer = nn.Conv

    def __init__(self, in_channels, out_channels, kernel_size, stride, device):
        super().__init__()
        self.conv = self.conv_layer(
            in_channels, out_channels, kernel_size, stride, device=device
        )
        self.bn = nn.BatchNorm(out_channels, device=device)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBN4(ConvBN):
    conv_layer = nn.Conv4


class ResNet9(sm.nn.Module):
    convbn_layer = ConvBN

    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.first_set = nn.Sequential(
            self.convbn_layer(3, 16, 7, 4, device), ConvBN(16, 32, 3, 2, device)
        )
        self.second_set = nn.Sequential(
            self.convbn_layer(32, 32, 3, 1, device), ConvBN(32, 32, 3, 1, device)
        )
        # res_1 = nn.Residual(self.first_set)
        self.third_set = nn.Sequential(
            self.convbn_layer(32, 64, 3, 2, device), ConvBN(64, 128, 3, 2, device)
        )
        self.fourth_set = nn.Sequential(
            self.convbn_layer(128, 128, 3, 1, device), ConvBN(128, 128, 3, 1, device)
        )
        # res_2 = nn.Residual(self.third_set)
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 10, device=device),
        )

    def forward(self, x):
        fs = self.first_set(x)
        ss = self.second_set(fs)
        r1 = ss + fs.broadcast_to(ss.shape)
        ts = self.third_set(r1)
        fos = self.fourth_set(ts)
        r2 = fos + ts.broadcast_to(fos.shape)
        return self.output(r2)


class ResNet94(ResNet9):
    convbn_layer = ConvBN4


class LangFlatten(nn.Module):
    """
    Flattens the dimensions of a Tensor after the first into one dimension.

    Input shape: (bs, s_1, ..., s_n)
    Output shape: (bs*s_1*...*s_n-1, s_n)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: "sm.Tensor") -> "sm.Tensor":

        x_shape = x.shape
        flat_dims = reduce(operator.mul, x_shape[:-1])
        new_shape = (flat_dims, x_shape[-1])
        return x.reshape(new_shape)


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        output_size,
        hidden_size,
        num_layers=1,
        seq_model="rnn",
        device=None,
        dtype="float32",
    ):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.

        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()

        self.output_size = output_size
        self.embedding = nn.Embedding(
            output_size, embedding_size, device=device, dtype=dtype
        )
        self.seq_model = seq_model
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model = (
            nn.RNN(
                embedding_size,
                hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
            if seq_model == "rnn"
            else nn.LSTM(
                embedding_size,
                hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
        )
        self.flat = LangFlatten()
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).

        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)

        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """

        emb = self.embedding(x)
        seq_out, h = self.seq_model(emb, h)
        flat_out = self.flat(seq_out)
        lin_out = self.linear(flat_out)
        return lin_out, h


# if __name__ == "__main__":
#     model = ResNet9()
#     x = sm.ops.randu((1, 32, 32, 3), requires_grad=True)
#     model(x)
#     cifar10_train_dataset = sm.data.CIFAR10Dataset(
#         "data/cifar-10-batches-py", train=True
#     )
#     train_loader = sm.data.DataLoader(
#         cifar10_train_dataset, 128, sm.cpu(), dtype="float32"
#     )
#     print(dataset[1][0].shape)
