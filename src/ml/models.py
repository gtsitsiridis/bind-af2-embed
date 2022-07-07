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


class CNN2DModel1(torch.nn.Module):

    def __init__(self):
        super(CNN2DModel1, self).__init__()
        in_channels = 2
        feature_channels = 128
        kernel_size = 5
        stride = 1
        dropout = 0.7
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Flatten(1, 2),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            # torch.nn.Conv1d(in_channels=feature_channels, out_channels=4, kernel_size=kernel_size,
            #                 stride=stride, padding=padding),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=4, kernel_size=kernel_size,
                            stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape((x.shape[0], 16, -1))
        x = self.conv2(x)
        return x


class CNN2DModel2(torch.nn.Module):

    def __init__(self, max_length: int):
        super(CNN2DModel2, self).__init__()
        kernel_size = 5
        stride = 1
        dropout = 0.7
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=256, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(in_channels=256, out_channels=4, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.AdaptiveAvgPool2d((max_length, 1)),
            torch.nn.Flatten(2),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=4, kernel_size=kernel_size,
                            stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class CNN2DModel3(torch.nn.Module):

    def __init__(self, max_length: int):
        super(CNN2DModel3, self).__init__()
        kernel_size = 5
        stride = 1
        dropout = 0.7
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=16, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Conv2d(in_channels=64, out_channels=31, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Flatten(1, 2),
            torch.nn.Conv1d(in_channels=max_length * 14, out_channels=max_length, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=max_length, out_channels=16, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=16, out_channels=4, kernel_size=kernel_size,
                            stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class CNN2DModel(torch.nn.Module):

    def __init__(self, max_length: int):
        super(CNN2DModel, self).__init__()
        kernel_size = 5
        stride = 1
        dropout = 0.7
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=1024, kernel_size=(kernel_size, max_length),
                            stride=stride, padding=(padding, 0)),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Flatten(2),
            torch.nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels=128, out_channels=4, kernel_size=kernel_size,
                            stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class CNNCombinedModel(torch.nn.Module):

    def __init__(self, max_length: int):
        super(CNNCombinedModel, self).__init__()
        activation = torch.nn.ReLU

        stride = 1
        kernel_size = 5
        dropout = 0.7
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm1d(128),
            # torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=kernel_size,
            #                 stride=stride, padding=padding),
            # activation(),
            # torch.nn.Dropout(dropout),
            # torch.nn.BatchNorm1d(128),
        )

        kernel_size = 5
        stride = 1
        dropout = 0.7
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(kernel_size, max_length),
                            stride=stride, padding=(padding, 0)),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm2d(16),
            torch.nn.Flatten(2),
            # torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size,
            #                 stride=stride, padding=padding),
            # activation(),
            # torch.nn.Dropout(dropout),
            # torch.nn.BatchNorm1d(64),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=128 + 16, out_channels=4, kernel_size=kernel_size,
                            stride=stride, padding=padding),
        )

    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        z = torch.cat((x, y), dim=1)
        z = self.conv3(z)
        return z
