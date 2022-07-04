import torch.nn


class CNN1DModel(torch.nn.Module):

    def __init__(self, in_channels, feature_channels, kernel_size, dropout):
        super(CNN1DModel, self).__init__()
        stride = 1
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=feature_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv1d(in_channels=feature_channels, out_channels=4, kernel_size=kernel_size,
                            stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x
