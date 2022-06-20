import torch
import numpy as np
from data.dataset import Dataset


class CNN1DDataset(torch.utils.data.Dataset):
    """custom Dataset"""

    def __init__(self, samples: list, dataset: Dataset):
        self.features_dict = dataset.to_feature_tensor_dict()
        self.labels_dict = dataset.to_bind_annot_tensor_dict()
        self.seqs_dict = dataset.to_sequence_str_dict()
        self.samples_dict = samples
        self.n_features = self.get_input_dimensions()
        self.max_length = dataset.determine_max_length()

        print('Number of input features: {}'.format(self.n_features))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        prot_id = self.samples_dict[item]
        prot_length = len(self.seqs_dict[prot_id])
        features_tmp = self.features_dict[prot_id]

        # pad all inputs to the maximum length & add another feature to encode whether the element is a position
        # in the sequence or padded
        feature_batch = np.zeros((self.n_features + 1, self.max_length), dtype=np.float32)
        feature_batch[:self.n_features, :prot_length] = np.transpose(features_tmp)  # set feature maps to embedding values
        feature_batch[self.n_features, :prot_length] = 1  # set last element to 1 because positions are not padded

        target_batch = np.zeros((3, self.max_length), dtype=np.float32)
        target_batch[:3, :prot_length] = np.transpose(self.labels_dict[prot_id])
        loss_mask_batch = np.zeros((3, self.max_length), dtype=np.float32)
        loss_mask_batch[:3, :prot_length] = prot_length

        return feature_batch, target_batch, loss_mask_batch, prot_id

    def get_input_dimensions(self) -> int:
        first_key = list(self.features_dict.keys())[0]
        first_feature = self.features_dict[first_key]

        return np.shape(first_feature)[1]


class CNN2DDataset(torch.utils.data.Dataset):
    """custom Dataset"""

    def __init__(self, samples: list, dataset: Dataset):
        self.features_dict = dataset.to_feature_tensor_dict()
        self.labels_dict = dataset.to_bind_annot_tensor_dict()
        self.seqs_dict = dataset.to_sequence_str_dict()
        self.samples_dict = samples
        self.n_features = self.get_input_dimensions()
        self.max_length = dataset.determine_max_length()

        print('Number of input features: {}'.format(self.n_features))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        prot_id = self.samples_dict[item]
        prot_length = len(self.seqs_dict[prot_id])
        features_tmp = self.features_dict[prot_id]

        # pad all inputs to the maximum length & add another feature to encode whether the element is a position
        # in the sequence or padded
        features = np.zeros((self.n_features + 1, self.max_length), dtype=np.float32)
        features[:self.n_features, :prot_length] = np.transpose(features_tmp)  # set feature maps to embedding values
        features[self.n_features, :prot_length] = 1  # set last element to 1 because positions are not padded

        target = np.zeros((3, self.max_length), dtype=np.float32)
        target[:3, :prot_length] = np.transpose(self.labels_dict[prot_id])
        loss_mask = np.zeros((3, self.max_length), dtype=np.float32)
        loss_mask[:3, :prot_length] = prot_length

        return features, target, loss_mask

    def get_input_dimensions(self) -> int:
        first_key = list(self.features_dict.keys())[0]
        first_feature = self.features_dict[first_key]

        return np.shape(first_feature)[1]
