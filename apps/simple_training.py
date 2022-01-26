import sys

sys.path.append("../python")

import time

import needle as ndl
import needle.nn as nn
import numpy as np
from models import *  # noqa: F403
from tqdm import tqdm

# TODO: Make these concrete imports instead of a star import

### CIFAR-10 training ###


def epoch_general_cifar10(dataloader, model, loss_fn=None, opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    if loss_fn is None:
        loss_fn = nn.SoftmaxLoss()
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    accs = []
    losses = []
    if not opt:
        model.eval()
    else:
        model.train()
    for X, y in tqdm(dataloader, position=0, leave=True):
        if opt:
            opt.reset_grad()
        out = model(X)
        loss = loss_fn(out, y)
        if opt:
            loss.backward()
            opt.step()
        out_cat = np.argmax(out.numpy(), axis=1)
        acc = np.mean(out_cat == y.numpy())
        accs.append(acc)
        losses.append(loss.numpy())
    model.train()
    return np.mean(accs), np.mean(losses)
    ### END YOUR SOLUTION


def train_cifar10(
    model,
    train_dataloader,
    val_dataloader,
    n_epochs=1,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    loss_fn=nn.SoftmaxLoss,
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    t0 = time.time()
    trajectories = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": [],
        "elapsed_time": [],
    }
    for i in tqdm(range(n_epochs), position=1, leave=True):
        train_acc, train_loss = epoch_general_cifar10(
            train_dataloader, model, loss_fn=loss_fn(), opt=opt
        )
        elapsed_time = time.time() - t0
        val_acc, val_loss = evaluate_cifar10(model, val_dataloader, loss_fn=loss_fn)
        print(
            "[Epoch {}] train_acc: {:.03f}, train_loss: {:.03f}, val_acc: {:.03f}, val_loss: {:.03f}".format(
                i, train_acc, train_loss, val_acc, val_loss
            )
        )
        t0 = time.time() - elapsed_time
        trajectories["train_acc"].append(train_acc)
        trajectories["train_loss"].append(train_loss)
        trajectories["val_acc"].append(val_acc)
        trajectories["val_loss"].append(val_loss)
        trajectories["elapsed_time"].append(elapsed_time)

    return train_acc, train_loss, trajectories
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    return epoch_general_cifar10(dataloader, model, loss_fn=loss_fn(), opt=None)
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(
    data,
    model,
    seq_len=40,
    loss_fn=None,
    opt=None,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    if loss_fn is None:
        loss_fn = nn.SoftmaxLoss()
    ### BEGIN YOUR SOLUTION
    model.train()
    correct, total_loss = 0, 0
    i = 1
    for i in range(data.shape[0]):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        out, _ = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        loss.backward()
        opt.step()
        i += 1
    return correct / (y.shape[0] * i), total_loss / (y.shape[0] * i)
    ### END YOUR SOLUTION


def train_ptb(
    model,
    data,
    seq_len=40,
    n_epochs=1,
    optimizer=ndl.optim.SGD,
    lr=4.0,
    weight_decay=0.0,
    loss_fn=nn.SoftmaxLoss,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss = loss_fn()
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        train_acc, train_loss = epoch_general_ptb(
            data,
            model,
            seq_len,
            loss_fn=loss,
            opt=opt,
            clip=clip,
            device=None,
            dtype="float32",
        )
        # test_acc, test_loss = evaluate_ptb()

    return train_acc, train_loss
    ### END YOUR SOLUTION


def evaluate_ptb(
    model, data, seq_len=40, loss_fn=nn.SoftmaxLoss, device=None, dtype="float32"
):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.eval()
    loss = loss_fn()
    correct, total_loss = 0, 0
    i = 1
    for i in range(data.shape[0]):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        out, _ = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        i += 1
    return correct / (y.shape[0] * i), total_loss / (y.shape[0] * i)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    # dataset = ndl.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=True)
    # dataloader = ndl.data.DataLoader(\
    #          dataset=dataset,
    #          batch_size=128,
    #          shuffle=True,
    #          collate_fn=ndl.data.collate_ndarray,
    #          drop_last=False,
    #          device=device,
    #          dtype="float32"
    #          )
    #
    # model = ResNet9(device=device, dtype="float32")
    # train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #       lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("../data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(
        corpus.train, batch_size, device=ndl.cpu(), dtype="float32"
    )
    # TODO: fix the noqa
    model = LanguageModel(  # noqa: F405
        1, len(corpus.dictionary), hidden_size, num_layers=2, device=ndl.cpu()
    )
    train_ptb(model, train_data, seq_len, n_epochs=10)
