from torch import nn

class SmallModelCNN(nn.Module):
  def __init__(self, input_neurons: int, hidden_layers: int, neurons_per_hidden_layer: list[int], output_neurons: int, image_size: tuple[int, int]):
    super().__init__()
    assert hidden_layers +1 == len(neurons_per_hidden_layer), "Number of items in the list of neurons per hidden layer must be 1 greater than the number hidden layers"

    self.input_neurons = input_neurons
    self.hidden_layers = hidden_layers
    self.neurons_per_hidden_layer = neurons_per_hidden_layer
    self.output_neurons = output_neurons

    self.hidden_neurons = neurons_per_hidden_layer[0]

    self.input_conv_block = nn.Sequential(
        nn.Conv2d(in_channels= self.input_neurons, out_channels=self.hidden_neurons, kernel_size= 3, stride= 1, padding= 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2, stride= 2)
    )

    self.hidden_conv_block = nn.Sequential(
        nn.Conv2d(in_channels= self.hidden_neurons, out_channels=self.hidden_neurons, kernel_size= 3, stride= 1, padding= 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2, stride= 2)
    )

    num_pool_layers = 1 + hidden_layers
    num = image_size[0] / (2**num_pool_layers)
    linear_hidden_neurons: int = self.hidden_neurons * int((num**2))
    self.output_linear_block = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features= linear_hidden_neurons, out_features= self.output_neurons),
        nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.input_conv_block(x)
    for i in range(1, self.hidden_layers +1):
      x = self.hidden_conv_block(x)
      self.hidden_neurons = self.neurons_per_hidden_layer[i]
    return self.output_linear_block(x)