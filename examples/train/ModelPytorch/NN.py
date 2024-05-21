from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, model_config, labelmap):
        input_length = len(model_config["features"]["used_features"])
        output_length = len(labelmap["class2idx"].keys())
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_length, 120),
            nn.ReLU(),
            nn.Linear(120, 240),
            nn.ReLU(),
            nn.Linear(240, output_length),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits