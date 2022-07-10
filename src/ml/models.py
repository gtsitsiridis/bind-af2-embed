import torch.nn


class BCEModel(torch.nn.Module):

    def __init__(self, in_channels, kernel_size):
        super(BCEModel, self).__init__()
        stride = 1
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=4, kernel_size=kernel_size,
                                     stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        return x


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
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class EmbeddingsModel(torch.nn.Module):

    def __init__(self, feature_channels, kernel_size, dropout):
        super(EmbeddingsModel, self).__init__()
        self.conv1 = torch.nn.Sequential(
            CNN1DModel(1024, feature_channels, kernel_size, dropout),
            BCEModel(in_channels=feature_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class DistMapsModel(torch.nn.Module):

    def __init__(self, max_length: int, feature_channels, kernel_size, dropout):
        super(DistMapsModel, self).__init__()
        stride = 1
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)
        activation = torch.nn.ReLU

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=feature_channels, kernel_size=(kernel_size, max_length),
                            stride=stride, padding=(padding, 0)),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm2d(feature_channels),
            torch.nn.Flatten(2),
            BCEModel(in_channels=feature_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class CombinedModelV1(torch.nn.Module):

    def __init__(self, in_channels, feature_channels, kernel_size, dropout):
        super(CombinedModelV1, self).__init__()
        self.conv1 = torch.nn.Sequential(
            CNN1DModel(in_channels, feature_channels, kernel_size, dropout),
            BCEModel(in_channels=feature_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class CombinedModelV2(torch.nn.Module):

    def __init__(self, max_length: int, emb_feature_channels, distmap_feature_channels, distmap_depth,
                 kernel_size, dropout):
        super(CombinedModelV2, self).__init__()
        activation = torch.nn.ReLU
        stride = 1
        # for stride = 1, then P = (kernel_size -) / 2, so that output_dim = input_dim
        padding = int((kernel_size - 1) / 2)

        self.emb_conv = CNN1DModel(1024, emb_feature_channels, kernel_size, dropout)

        self.distmaps_convs = []
        distmap_in_channels = 2
        for i in range(distmap_depth):
            self.distmaps_convs.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=distmap_in_channels, out_channels=distmap_feature_channels,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding),
                activation(),
                torch.nn.Dropout(dropout),
                torch.nn.BatchNorm2d(distmap_feature_channels),
            ))
            distmap_in_channels = distmap_feature_channels
            distmap_feature_channels = distmap_feature_channels * 2

        self.distmap_final_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=distmap_in_channels, out_channels=distmap_feature_channels,
                            kernel_size=(kernel_size, max_length),
                            stride=stride, padding=(padding, 0)),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm2d(distmap_feature_channels),
            torch.nn.Flatten(2),
        )

        self.bce_conv = BCEModel(in_channels=emb_feature_channels + distmap_feature_channels,
                                 kernel_size=kernel_size)

    def forward(self, x, y):
        x = self.emb_conv(x)

        for conv in self.distmaps_convs:
            y = conv(y)
        y = self.distmap_final_conv(y)
        z = torch.cat((x, y), dim=1)
        z = self.bce_conv(z)
        return z
