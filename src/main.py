from dataset import Dataset
from config import AppConfig


def __main():
    dataset = Dataset(AppConfig())
    ids = dataset.get_ids()
    fold_array = dataset.get_fold_array()
    embeddings = dataset.get_embeddings()
    sequences = dataset.get_sequences()
    labels = dataset.get_labels()
    pass


if __name__ == '__main__':
    __main()
