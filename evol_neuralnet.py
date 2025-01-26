import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os


# Define a dynamic feedforward neural network
class DynamicNN(nn.Module):
    """
    a dynamically configurable feedforward neural network with dropout layers.
    allows for flexible architecture based on the number and sizes of hidden layers.

    @param input_size int: the size of the input layer (e.g., 28*28 for MNIST images).
    @param hidden_sizes list[int]: a list of integers defining the size of each hidden layer.
    @param output_size int: the size of the output layer (e.g., number of classes).
    @param dropout_rate float: the dropout rate for regularization.
    @param activation_function callable: the activation function to use (e.g., nn.ReLU).
    """

    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate, activation_function):
        super(DynamicNN, self).__init__()
        layers = []
        prev_size = input_size

        # dynamically build the network based on hidden_sizes
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))  # fully connected layer
            layers.append(activation_function())      # activation function
            layers.append(nn.Dropout(p=dropout_rate)) # dropout for regularization
            prev_size = size

        # final output layer
        layers.append(nn.Linear(prev_size, output_size))  # map to output classes
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        @param x torch.Tensor: Input tensor (flattened images in this case).
        @return torch.Tensor: Output tensor after passing through the network.
        """
        return self.model(x)


# Train and evaluate the model
def train_and_evaluate(model_hyperparams):
    """
    Trains and evaluates a neural network based on provided hyperparameters.

    @param hyperparams dict: A dictionary containing hyperparameters:
           - 'learning_rate' (float): Learning rate for the optimizer.
           - 'hidden_sizes' (list[int]): Sizes of the hidden layers.
           - 'num_hidden_layers' (int): Number of active hidden layers.
           - 'batch_size' (int): Size of the training and testing batches.
           - 'epochs' (int): Number of training epochs.
           - 'dropout_rate' (float): Dropout rate for regularization.
           - 'activation_function' (int): Activation function (0: ReLU, 1: Tanh, 2: Sigmoid, 3: LeakyReLU).
    @return tuple: A tuple containing:
           - fitness (float): Combined fitness score (accuracy - normalized loss).
           - accuracy (float): Classification accuracy on the test set.
           - average_loss (float): Average loss on the test set.
    """

    # extract hyperparameters
    learning_rate = model_hyperparams.get('learning_rate', None)
    hidden_sizes = model_hyperparams.get('hidden_sizes', [])[:model_hyperparams.get('num_hidden_layers', 0)]
    batch_size = model_hyperparams.get('batch_size', None)
    epochs = model_hyperparams.get('epochs', None)
    dropout_rate = model_hyperparams.get('dropout_rate', None)
    activation_function = model_hyperparams.get('activation_function', None)

    # validate hyperparameters
    if not (0.0001 <= learning_rate <= 0.1):
        raise ValueError(f"invalid learning_rate: {learning_rate}. must be in range [0.0001, 0.1].")
    if not (0.0 <= dropout_rate <= 0.5):
        raise ValueError(f"invalid dropout_rate: {dropout_rate}. must be in range [0.0, 0.5].")
    if not (16 <= batch_size <= 128):
        raise ValueError(f"invalid batch_size: {batch_size}. must be in range [16, 128].")
    if not (1 <= epochs <= 50):
        raise ValueError(f"invalid epochs: {epochs}. must be in range [1, 50].")
    if not (0 <= activation_function <= 3):
        raise ValueError(f"invalid activation_function: {activation_function}. must be in range [0, 3].")
    if not (1 <= len(hidden_sizes) <= 5 and all(16 <= size <= 128 for size in hidden_sizes)):
        raise ValueError(f"invalid hidden_sizes: {hidden_sizes}. each size must be in range [16, 128].")

    # map activation function integer to PyTorch activation function
    activation_functions = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU]
    selected_activation = activation_functions[activation_function]

    # check for CUDA availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    # print("[python info]: using GPU for training.", file=sys.stderr)
    # else:
    # print("[pypthon info]: using CPU for training.", file=sys.stderr)

    # load MNIST (digits 0–4 only)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

    # filter only digits 0–4 for a subset of the dataset
    train_dataset.data = train_dataset.data[train_dataset.targets < 5]
    train_dataset.targets = train_dataset.targets[train_dataset.targets < 5]
    test_dataset.data = test_dataset.data[test_dataset.targets < 5]
    test_dataset.targets = test_dataset.targets[test_dataset.targets < 5]

    # create data loaders for batching
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # initialize the model, loss function, and optimizer
    input_size = 28 * 28  # each MNIST image is 28x28
    output_size = 5       # only digits 0–4 are used
    model = DynamicNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate,
        activation_function=selected_activation
    ).to(device)  # move model to GPU/CPU
    criterion = nn.CrossEntropyLoss()  # cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # adam optimizer

    # create logs folder if it doesn't exist
    log_folder = "../logs"
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, "training_log.txt")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"Training started with hyperparameters: {model_hyperparams}\n")

        # track max loss during training
        max_loss = 0.0

        # training loop
        model.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)   # move data to GPU/CPU
                data = data.view(data.size(0), -1)                  # flatten images
                optimizer.zero_grad()                               # reset gradients
                output = model(data)                                # forward pass
                # print(f"[python debug]: current GPU memory usage: {torch.cuda.memory_allocated()} bytes", file=sys.stderr)
                model_loss = criterion(output, target)              # compute loss
                max_loss = max(max_loss, model_loss.item())         # update max loss
                model_loss.backward()                               # backward pass
                optimizer.step()                                    # update weights

        # evaluation loop
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                # move data and target to the same device as the model
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)  # flatten images

                # forward pass
                output = model(data)
                # print(f"[python debug]: current GPU memory usage: {torch.cuda.memory_allocated()} bytes", file=sys.stderr)

                # compute loss
                total_loss += criterion(output, target).item() * data.size(0)

                # get predicted labels
                _, predicted = torch.max(output, 1)

                # print(f"[python debug]: predicted type: {type(predicted)}, device: {predicted.device}", file=sys.stderr)
                # print(f"[python debug]: target type: {type(target)}, device: {target.device}", file=sys.stderr)

                # ensure the comparison happens on tensors
                correct += (predicted == target).sum().item()  # count correct predictions
                total += target.size(0)  # total number of samples

        # compute metrics
        model_accuracy = correct / total
        average_loss = total_loss / total  # normalize total loss by dataset size

        # normalize loss and compute performance score
        # scale loss to match the accuracy range, adjust based on the range of observed losses
        normalized_loss = min(1.0, max(0.0, average_loss / max_loss))  # clamp to ensure it stays between 0 and 1

        # combine accuracy and loss into a performance score
        alpha = 0.7  # weight for accuracy
        performance_score = alpha * model_accuracy + (1 - alpha) * (1 - normalized_loss)

        # clamp final performance score to be between 0 and 1
        performance_score = min(1.0, max(0.0, performance_score))

        # log evaluation results to adjust factor for normalizing loss and validating performance score calculation
        log_file.write(f"Evaluation - Accuracy: {model_accuracy:.6f}, Loss: {average_loss:.6f}, Performance Score: {performance_score:.6f}\n")
        log_file.write("-" * 50 + "\n")

    # return performance score, accuracy, and average loss
    return performance_score, model_accuracy, average_loss


if __name__ == '__main__':
    """
    main entry point for the script. parses command-line arguments,
    trains and evaluates the neural network, and outputs the fitness score.
    """
    try:
        # parse hyperparameters from command-line arguments
        hyperparams = json.loads(sys.argv[1])

        # train and evaluate the neural network
        fitness, accuracy, loss = train_and_evaluate(hyperparams)

        # output the fitness score (used by the C++ program)
        print(fitness)

    except (IndexError, json.JSONDecodeError, ValueError) as e:
        print(f"error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"unexpected error: {e}")
        sys.exit(1)
