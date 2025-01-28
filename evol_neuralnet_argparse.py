# standard python library imports for various functionalities
import sys         # system-specific parameters and functions
import torch       # deep learning framework
import torch.nn as nn       # neural network modules
import torch.optim as optim # optimization algorithms
from torchvision import datasets, transforms  # dataset and transformation utilities
import os          # operating system interactions
import argparse    # for argument parsing

# define a dynamic feedforward neural network
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
        # call parent class constructor
        super(DynamicNN, self).__init__()
        layers = []
        prev_size = input_size

        # dynamically build the network based on hidden_sizes
        for size in hidden_sizes:
            # fully connected layer connecting previous layer to current layer size
            layers.append(nn.Linear(prev_size, size))
            # add activation function to introduce non-linearity
            layers.append(activation_function())
            # add dropout layer for regularization to prevent overfitting
            layers.append(nn.Dropout(p=dropout_rate))
            prev_size = size

        # final output layer maps last hidden layer to output classes
        layers.append(nn.Linear(prev_size, output_size))
        # convert list of layers into a sequential neural network model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        forward pass through the network.

        @param x torch.Tensor: input tensor (flattened images in this case).
        @return torch.Tensor: output tensor after passing through the network.
        """
        # pass input through the sequentially defined network model
        return self.model(x)


# train and evaluate the model
def train_and_evaluate(model_hyperparams):
    """
    trains and evaluates a neural network based on provided hyperparameters.

    @param hyperparams dict: a dictionary containing neural network configuration:
           - 'learning_rate' (float): learning rate for the optimizer.
           - 'hidden_sizes' (list[int]): sizes of the hidden layers.
           - 'num_hidden_layers' (int): number of active hidden layers.
           - 'batch_size' (int): size of the training and testing batches.
           - 'epochs' (int): number of training epochs.
           - 'dropout_rate' (float): dropout rate for regularization.
           - 'activation_function' (int): activation function (0: ReLU, 1: Tanh, 2: Sigmoid, 3: LeakyReLU).
    @return tuple: a tuple containing:
           - fitness (float): combined fitness score (accuracy - normalized loss).
           - accuracy (float): classification accuracy on the test set.
           - average_loss (float): average loss on the test set.
    """

    # extract hyperparameters from input dictionary with default protection
    learning_rate = model_hyperparams.get('learning_rate', None)
    hidden_sizes = model_hyperparams.get('hidden_sizes', [])[:model_hyperparams.get('num_hidden_layers', 0)]
    batch_size = model_hyperparams.get('batch_size', None)
    epochs = model_hyperparams.get('epochs', None)
    dropout_rate = model_hyperparams.get('dropout_rate', None)
    activation_function = model_hyperparams.get('activation_function', None)

    # map activation function integer to PyTorch activation function
    activation_functions = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU]
    selected_activation = activation_functions[activation_function]

    # check for CUDA availability and set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load MNIST dataset (subset of digits 0–4)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

    # filter dataset to include only digits 0–4
    train_dataset.data = train_dataset.data[train_dataset.targets < 5]
    train_dataset.targets = train_dataset.targets[train_dataset.targets < 5]
    test_dataset.data = test_dataset.data[test_dataset.targets < 5]
    test_dataset.targets = test_dataset.targets[test_dataset.targets < 5]

    # create data loaders for efficient batching and processing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # initialize the model with dynamic configuration
    input_size = 28 * 28  # each MNIST image is 28x28 pixels
    output_size = 5       # classifying only digits 0–4
    model = DynamicNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate,
        activation_function=selected_activation
    ).to(device)  # move model to appropriate computation device

    # configure loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # cross-entropy loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # adaptive moment estimation optimizer

    # create logs directory if it doesn't exist
    log_folder = "../logs"
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, "training_log.txt")

    # log training details and track maximum loss
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Training started with hyperparameters: {model_hyperparams}\n")
        max_loss = 0.0

        # training loop
        model.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                # move data and target to computation device and flatten images
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)

                # zero out previous gradients to prevent accumulation
                optimizer.zero_grad()

                # forward pass through the network
                output = model(data)

                # compute loss
                model_loss = criterion(output, target)
                max_loss = max(max_loss, model_loss.item())

                # backward pass to compute gradients
                model_loss.backward()

                # update model weights
                optimizer.step()

        # evaluation loop
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # disable gradient computation during evaluation
        with torch.no_grad():
            for data, target in test_loader:
                # move data and target to computation device and flatten images
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)

                # forward pass
                output = model(data)

                # compute loss
                total_loss += criterion(output, target).item() * data.size(0)

                # get predicted labels
                _, predicted = torch.max(output, 1)

                # count correct predictions
                correct += (predicted == target).sum().item()
                total += target.size(0)

        # compute performance metrics
        model_accuracy = correct / total
        average_loss = total_loss / total

        # normalize loss relative to maximum observed loss
        normalized_loss = min(1.0, max(0.0, average_loss / max_loss))

        # combine accuracy and loss into a performance score
        alpha = 0.7  # weight for accuracy
        performance_score = alpha * model_accuracy + (1 - alpha) * (1 - normalized_loss)

        # ensure performance score is between 0 and 1
        performance_score = min(1.0, max(0.0, performance_score))

        # log evaluation results
        log_file.write(f"Evaluation - Accuracy: {model_accuracy:.6f}, Loss: {average_loss:.6f}, Performance Score: {performance_score:.6f}\n")
        log_file.write("-" * 50 + "\n")

    # return performance metrics
    return performance_score, model_accuracy, average_loss

def parse_arguments():
    """
    Parse and validate command line arguments for the neural network.

    Returns:
        dict: Validated hyperparameters for the neural network
    Raises:
        ValueError: If arguments are invalid or missing
    """
    parser = argparse.ArgumentParser(description='Neural Network Training and Evaluation')

    # Add all arguments
    parser.add_argument('--learning-rate', type=float, required=True,
                        help='Learning rate for the optimizer')
    parser.add_argument('--dropout-rate', type=float, required=True,
                        help='Dropout rate for regularization')
    parser.add_argument('--batch-size', type=int, required=True,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of training epochs')
    parser.add_argument('--activation-function', type=int, required=True,
                        choices=[0, 1, 2, 3],
                        help='Activation function (0: ReLU, 1: Tanh, 2: Sigmoid, 3: LeakyReLU)')
    parser.add_argument('--num-hidden-layers', type=int, required=True,
                        help='Number of hidden layers')
    parser.add_argument('--hidden-sizes', type=str, required=True,
                        help='Hidden layer sizes as comma-separated values (e.g., "64,32,16")')

    # Parse arguments
    args = parser.parse_args()

    # Process hidden sizes
    try:
        # Remove any brackets and split by commas
        hidden_sizes_str = args.hidden_sizes.strip('[]')
        hidden_sizes = [int(x.strip()) for x in hidden_sizes_str.split(',')]

        # Validate number of layers
        if len(hidden_sizes) < args.num_hidden_layers:
            raise ValueError(
                f"Not enough hidden sizes provided. Need {args.num_hidden_layers}, got {len(hidden_sizes)}")

        # Truncate to requested number of layers
        hidden_sizes = hidden_sizes[:args.num_hidden_layers]

    except ValueError as e:
        raise ValueError(f"Error parsing hidden sizes: {e}")

    # Create and return hyperparameters dictionary
    return {
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'activation_function': args.activation_function,
        'num_hidden_layers': args.num_hidden_layers,
        'hidden_sizes': hidden_sizes
    }

# main entry point for the script
if __name__ == '__main__':
    """
    main entry point for the script. parses command-line arguments,
    trains and evaluates the neural network, and outputs the fitness score.
    """
    try:
        # parse hyperparameters from command-line arguments
        hyperparams = parse_arguments()

        # train and evaluate the neural network
        fitness, accuracy, loss = train_and_evaluate(hyperparams)

        # output the fitness score (used by the C++ program)
        print(fitness)

    except Exception as e:
        print(f"unexpected error: {e}")
        sys.exit(1)