import torch
import numpy as np
from data.dataset import Dataset
from logging import getLogger
from ml.summary_writer import MySummaryWriter

logger = getLogger('app')


class MLDataset(torch.utils.data.Dataset):
    """custom Dataset"""

    def __init__(self, samples: list, dataset: Dataset, writer: MySummaryWriter = None):
        self.dataset = dataset
        self.samples = samples
        logger.info('Number of samples: {}'.format(len(self.samples)))

        if writer is not None:
            plddt_tensors = []
            prot_length_list = []
            for prot_id in samples:
                protein = dataset.proteins[prot_id]
                plddt_tensors.append(protein.structure.plddt_tensor)
                prot_length_list.append(len(protein))

            plddt_tensor = np.concatenate(plddt_tensors)
            writer.add_plddt_info(plddt_tensor)
            writer.add_protein_length_info(np.array(prot_length_list))


class CNN1DAllDataset(MLDataset):
    """custom Dataset"""

    def __init__(self, samples: list, dataset: Dataset, max_length: int, embedding_size: int,
                 writer: MySummaryWriter = None):
        super(CNN1DAllDataset, self).__init__(dataset=dataset, samples=samples, writer=writer)
        self.features_dict = dataset.to_feature_tensor_dict(max_length=max_length)
        self.labels_dict = dataset.to_bind_annot_tensor_dict()
        self.seqs_dict = dataset.to_sequence_str_dict()
        self.samples = samples
        self.n_features = 2 * max_length + embedding_size
        self.max_length = max_length
        logger.info('Number of input features: {}'.format(self.n_features))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        prot_id = self.samples[item]
        prot_length = len(self.seqs_dict[prot_id])
        features_tmp = self.features_dict[prot_id]
        target_tmp = self.labels_dict[prot_id]
        if prot_length > self.max_length:
            prot_length = self.max_length
            features_tmp = features_tmp[:self.max_length, :self.n_features]
            target_tmp = target_tmp[:self.max_length, :]

        # pad all inputs to the maximum length
        features = np.zeros((self.n_features, self.max_length), dtype=np.float32)
        features[:self.n_features, :prot_length] = np.transpose(
            features_tmp)  # set feature maps to embedding values

        # padding array
        padding = np.zeros(self.max_length, dtype=np.float32)
        padding[:prot_length] = 1  # set last element to 1 because positions are not padded

        target = np.zeros((4, self.max_length), dtype=np.float32)
        target[:4, :prot_length] = np.transpose(target_tmp)
        loss_mask = np.zeros((4, self.max_length), dtype=np.float32)
        loss_mask[:4, :prot_length] = prot_length

        return features, padding, target, loss_mask, prot_id

    def get_input_dimensions(self) -> int:
        first_key = list(self.features_dict.keys())[0]
        first_feature = self.features_dict[first_key]

        return np.shape(first_feature)[1]


class CNN1DEmbeddingsDataset(MLDataset):
    """custom Dataset"""

    def __init__(self, samples: list, dataset: Dataset, writer: MySummaryWriter = None):
        super().__init__(samples, dataset, writer)
        self.n_features = self.get_input_dimensions()
        self.max_length = dataset.determine_max_length()

        logger.info('Number of input features: {}'.format(self.n_features))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        prot_id = self.samples[item]
        protein = self.dataset.proteins[prot_id]
        prot_length = len(protein.sequence)
        features_tmp = protein.embedding.tensor

        # pad all inputs to the maximum length
        features = np.zeros((self.n_features, self.max_length), dtype=np.float32)
        features[:self.n_features, :prot_length] = np.transpose(
            features_tmp)  # set feature maps to embedding values

        # padding array
        padding = np.zeros(self.max_length, dtype=np.float32)
        padding[:prot_length] = 1  # set last element to 1 because positions are not padded

        target = np.zeros((4, self.max_length), dtype=np.float32)
        target[:4, :prot_length] = np.transpose(protein.bind_annotation.tensor)
        loss_mask = np.zeros((4, self.max_length), dtype=np.float32)
        loss_mask[:4, :prot_length] = prot_length

        return features, padding, target, loss_mask, prot_id

    def get_input_dimensions(self) -> int:
        first_key = list(self.dataset.proteins.keys())[0]
        first_protein = self.dataset.proteins[first_key]

        return np.shape(first_protein.embedding.tensor)[1]


class CNN2DDistMaps(MLDataset):
    """custom Dataset"""

    def __init__(self, samples: list, dataset: Dataset, max_length: int, writer: MySummaryWriter = None):
        super().__init__(samples, dataset, writer)
        self.n_features = self.get_input_dimensions()
        # self.n_features = 1
        self.max_length = max_length

        logger.info('Number of input features: {}'.format(self.n_features))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        prot_id = self.samples[item]
        protein = self.dataset.proteins[prot_id]
        prot_length = len(protein.sequence)
        features_tmp = protein.structure.distogram_tensor

        # pad all inputs to the maximum length
        features = np.zeros((self.n_features, self.max_length, self.max_length), dtype=np.float32)
        features[:self.n_features, :prot_length, :prot_length] = np.transpose(
            features_tmp)  # set feature maps to embedding values

        # padding array
        padding = np.zeros(self.max_length, dtype=np.float32)
        padding[:prot_length] = 1  # set last element to 1 because positions are not padded

        target = np.zeros((4, self.max_length), dtype=np.float32)
        target[:4, :prot_length] = np.transpose(protein.bind_annotation.tensor)
        loss_mask = np.zeros((4, self.max_length), dtype=np.float32)
        loss_mask[:4, :prot_length] = prot_length

        return features, padding, target, loss_mask, prot_id

    def get_input_dimensions(self) -> int:
        first_key = list(self.dataset.proteins.keys())[0]
        first_protein = self.dataset.proteins[first_key]

        return first_protein.structure.distogram_tensor.shape[2]


class CNNCombinedDataset(MLDataset):
    """custom Dataset"""

    def __init__(self, samples: list, dataset: Dataset, max_length: int, writer: MySummaryWriter = None):
        super().__init__(samples, dataset, writer)
        # self.n_features = 1
        self.max_length = max_length
        self._num_dist_features = 2
        self._num_emb_features = 1024

        logger.info('Number of input features: {}'.format(self._num_dist_features + self._num_emb_features))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        prot_id = self.samples[item]
        protein = self.dataset.proteins[prot_id]
        prot_length = len(protein.sequence)
        distmaps_tmp = protein.structure.distogram_tensor
        embeddings_tmp = protein.embedding.tensor
        num_dist_features = self._num_dist_features
        num_emb_features = self._num_emb_features

        # pad all inputs to the maximum length
        distmaps = np.zeros((num_dist_features, self.max_length, self.max_length), dtype=np.float32)
        distmaps[:num_dist_features, :prot_length, :prot_length] = np.transpose(
            distmaps_tmp)  # set feature maps to embedding values

        # pad all inputs to the maximum length
        embeddings = np.zeros((num_emb_features, self.max_length), dtype=np.float32)
        embeddings[:num_emb_features, :prot_length] = np.transpose(
            embeddings_tmp)  # set feature maps to embedding values

        features = (embeddings, distmaps)

        # padding array
        padding = np.zeros(self.max_length, dtype=np.float32)
        padding[:prot_length] = 1  # set last element to 1 because positions are not padded

        target = np.zeros((4, self.max_length), dtype=np.float32)
        target[:4, :prot_length] = np.transpose(protein.bind_annotation.tensor)
        loss_mask = np.zeros((4, self.max_length), dtype=np.float32)
        loss_mask[:4, :prot_length] = prot_length

        return features, padding, target, loss_mask, prot_id
