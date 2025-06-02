from torch import nn

class CNNModel(nn.Module):
  def __init__(self, input_channels: int, neurons_per_hidden_layer: list[int], output_channels: int, output_block_divisor: int, image_size: tuple[int, int]):
    super().__init__()
    assert len(neurons_per_hidden_layer) >= 1, "Neurons per hidden layer must have at least on element"
    
    num_hidden_layers = len(neurons_per_hidden_layer) - 1
    self.neurons_per_hidden_layer = neurons_per_hidden_layer

    self.input_conv_block = nn.Sequential(
        nn.Conv2d(in_channels= input_channels, out_channels=self.neurons_per_hidden_layer[0], kernel_size= 3, stride= 1, padding= 1),
        nn.BatchNorm2d(self.neurons_per_hidden_layer[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2, stride= 2)
    )

    self.hidden_blocks = nn.ModuleList()
    for i in range(1, num_hidden_layers +1):
        self.hidden_blocks.append(nn.Sequential(
            nn.Conv2d(in_channels=self.neurons_per_hidden_layer[i-1], out_channels=self.neurons_per_hidden_layer[i], kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(self.neurons_per_hidden_layer[i]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ))


    flatten_in_neurons: int = self.get_linear_in_neurons(image_size)
    flatten_out_neurons: int = flatten_in_neurons // output_block_divisor
    self.flatten_block = nn.Sequential(nn.Flatten(),
        nn.Linear(in_features= flatten_in_neurons, out_features= flatten_out_neurons),
        nn.BatchNorm1d(flatten_out_neurons),
        nn.ReLU(),
        nn.Dropout(p=0.5),
    )

    self.output_block = nn.Sequential(
        nn.Linear(in_features=flatten_out_neurons, out_features=output_channels),
        nn.Softmax(dim=1)
    )

  def get_linear_in_neurons(self, image_size: tuple[int, int]) -> int:
    height, width = image_size
    for _ in range(len(self.hidden_blocks) + 1):
        height //= 2
        width //= 2
    return self.neurons_per_hidden_layer[-1] * height * width

  def forward(self, x):
    x = self.input_conv_block(x)
    for hidden_layer in self.hidden_blocks:
      x = hidden_layer(x)
    return self.output_block(self.flatten_block(x))