"""CNN-based image classification on SageMaker with PyTorch

(Complete me with help from Local Notebook.ipynb, and the NLP example's src/main.py!)
"""
# Dependencies:
import argparse
# TODO: Others?

import json
import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


def parse_args():
    '''
    parese arguments
    '''
    # TODO: Standard pattern for loading parameters in from SageMaker
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories

    ##################################
    # Local or Cloud Mode
    ##################################    
    if os.environ.get('SM_CURRENT_HOST'): 
        # SageMaker Container environment for local or cloud mode
        parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
        parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
        parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
        parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    else: # Naive script mode
        parser.add_argument(
            "--train_dir",
            type=str,
            default="/tmp/mnist/training",
            help="This is training data",
        )
        parser.add_argument(
            "--test_dir",
            type=str,
            default="/tmp/mnist/testing",
            help="This is testing data",
        )
        parser.add_argument(
            "--model-dir",
            type=str,
            default="",
            help="This is loaction of model data",
        )
        parser.add_argument(
            "--output-data-dir",
            type=str,
            default="",
            help="\This is output data location",
        )

    ##################################
    # for both Local and Cloud Mode
    ##################################    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    args = parser.parse_args()

    return args

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(os.path.join(model_dir, 'model.pth'))
    return model

# TODO: Other function definitions, if you'd like to break up your code into functions?
def make_X_y_dataset(training_dir,testing_dir ):
    '''
    Create X, y dataset
    '''
    print("##: Starting X, y dataset creation")

    labels = sorted(os.listdir(training_dir))
    n_labels = len(labels)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    print("Loading label ", end="")
    for ix_label in range(n_labels):
        label_str = labels[ix_label]
        print(f"{label_str}...", end="")
        trainfiles = filter(
            lambda s: s.endswith(".png"),
            os.listdir(os.path.join(training_dir, label_str)),
        )
        for filename in trainfiles:
            with open(os.path.join(training_dir, label_str, filename), "rb") as imgfile:
                x_train.append(np.squeeze(np.asarray(Image.open(imgfile))))
                y_train.append(ix_label)
        # Repeat for test data:
        testfiles = filter(
            lambda s: s.endswith(".png"),
            os.listdir(os.path.join(testing_dir, label_str)),
        )
        for filename in testfiles:
            with open(os.path.join(testing_dir, label_str, filename), "rb") as imgfile:
                x_test.append(np.squeeze(np.asarray(Image.open(imgfile))))
                y_test.append(ix_label)
    print()

    print("Shuffling trainset...")
    train_shuffled = [(x_train[ix], y_train[ix]) for ix in range(len(y_train))]
    np.random.shuffle(train_shuffled)

    x_train = np.array([datum[0] for datum in train_shuffled])
    y_train = np.array([datum[1] for datum in train_shuffled])
    train_shuffled = None

    print("Shuffling testset...")
    test_shuffled = [(x_test[ix], y_test[ix]) for ix in range(len(y_test))]
    np.random.shuffle(test_shuffled)

    x_test = np.array([datum[0] for datum in test_shuffled])
    y_test = np.array([datum[1] for datum in test_shuffled])
    test_shuffled = None

    print("Done!")
    return x_train, y_train, x_test, y_test, n_labels

def preprocess(x_train, y_train, x_test, y_test, n_labels):
    '''
    pre-process dataset
    '''
    print("##: Starting preprocess")
    x_train = np.expand_dims(x_train, 1)
    x_test = np.expand_dims(x_test, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    input_shape = x_train.shape[1:]

    print("x_train shape:", x_train.shape)
    print("input_shape:", input_shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")


    def to_categorical(y, num_classes):
        """1-hot encodes a tensor"""
        return np.eye(num_classes, dtype="float32")[y]

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, n_labels)
    y_test = to_categorical(y_test, n_labels)

    print("n_labels:", n_labels)
    print("y_train shape:", y_train.shape)    

    return x_train, y_train, x_test, y_test

################
# Network model 
################
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten1(self.dropout1(self.max_pool2d(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(self.dropout2(x))
        return F.softmax(x, dim=-1)

################
# Pytorch dataset
################
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        """Initialization"""
        self.labels = labels
        self.data = data

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        # Load data and get label
        X = self.data[index]
        y = self.labels[index]
        return X, y

# Define train data loader 
def load_train_loader(Dataset,x_train, y_train,batch_size):
    trainloader = torch.utils.data.DataLoader(
        Dataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    print("##: trainloader is successfully loaded ")
    return trainloader

# Define test data loader 
def load_test_loader(Dataset,x_test, y_test):
    testloader = torch.utils.data.DataLoader(
        Dataset(x_test, y_test),
        batch_size= 1,
        shuffle=True,
    )
    print("##: testloader is successfully loaded ")
    return testloader

# train function
def train(trainloader, testloader, epochs, num_classes):
    print("## Start training ")
    model = Net(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adadelta(model.parameters())
    loss_function = F.binary_cross_entropy

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (x_train, y_train) in enumerate(trainloader):
            data, target = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("epoch: {}".format(epoch))
        print("train_loss: {:.6f}".format(running_loss / (len(trainloader.dataset))))
        print("Evaluating model")
        test(model, testloader, device)
    return model

# test function
def test(model, testloader, device):
    print("## Start testing ")

    loss_function = F.binary_cross_entropy
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target, reduction="mean").item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            target_index = target.max(1, keepdim=True)[1]
            correct += pred.eq(target_index).sum().item()

    test_loss /= len(testloader.dataset)
    print("val_loss: {:.4f}".format(test_loss))
    print("val_acc: {:.4f}".format(correct / len(testloader.dataset)))

# save model
def save_model(model, path):
    print(f"Saving model at {path}")
    # Ensure the subfolder exists:
    os.makedirs(path.rpartition("/")[0], exist_ok=True)

    x = torch.rand((1, 1, 28, 28), dtype=torch.float)
    model = model.cpu()
    model.eval()
    m = torch.jit.trace(model, x)
    torch.jit.save(m, path)
    print(f"## model is saved at {path}")

# Training script:
if __name__ == "__main__":

    # Check out if navie mode or sagemaker mode is
    if os.environ.get('SM_CURRENT_HOST'):
        print("## SageMaker Local or Cloud mode is set")
    else:
        print("## Naive mode is set")

    # TODO: Load arguments from CLI / environment variables?
    args = parse_args()

    print("## args: \n", args)
    print("args.training data: ", args.train_dir)
    print("args.test data: ", args.test_dir)
    training_dir = args.train_dir
    testing_dir = args.test_dir
    batch_size = args.batch_size
    epochs = args.epochs
    save_model_dir = args.model_dir

    # TODO: Load images from container filesystem into training / test data sets?
    x_train, y_train, x_test, y_test, n_labels = make_X_y_dataset(training_dir,testing_dir)
    x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test, n_labels)

    # TODO: Load dataset into a PyTorch Data Loader with correct batch size
    trainloader = load_train_loader(Dataset,x_train, y_train,batch_size)
    testloader = load_test_loader(Dataset,x_test, y_test)

    # TODO: Fit the PyTorch model?

    model = train(trainloader, testloader, epochs, n_labels)
        
    # TODO: Save outputs (trained model) to specified folder?
    save_model_dir = os.path.join(save_model_dir, "model.pth")
    save_model(model, save_model_dir)