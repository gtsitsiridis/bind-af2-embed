import ml.datasets
from data.dataset import Dataset
from config import AppConfig
from utils import Logging
from logging import getLogger


logger = getLogger('app')


def __main():
    config = AppConfig()
    Logging.setup_app_logger(config=config, write=True)
    # all_dataset = Dataset(config)
    train_dataset = Dataset(config, mode='train')
    # test_dataset = Dataset(config, mode='test')
    # logger.info(len(all_dataset))
    logger.info(len(train_dataset))
    # logger.info(len(test_dataset))

    samples = list(train_dataset.proteins.keys())[:20]
    torch_dataset = ml.torch_dataset.CNN1DDataset(samples=samples, dataset=train_dataset)
    in_feature, target, loss_mask = torch_dataset.__getitem__(1)
    print(1)

    # i = 0
    # for prot_id, protein in dataset.proteins.items():
    #     i += 1
    #     print(prot_id)
    #     print(protein.structure.pdb_file)
    #     print(protein.structure.avg_plddt())
    #     if i == 100:
    #         break
    #     long_df, tensor_dict = dataset.long_data()
    #     long_embeddings = tensor_dict['embeddings']
    #     long_labels = tensor_dict['binding_annotations']
    #     long_distograms = tensor_dict['distograms']
    # proteins = dataset.proteins
    # prot = proteins['P27797']
    # # prot.show_umap()
    # prot.show_structure(color='ligand')
    pass


if __name__ == '__main__':
    __main()
