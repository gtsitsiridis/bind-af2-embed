from config import AppConfig
from utils import Logging
from logging import getLogger
from ml.pipeline import Pipeline
from ml.method import MethodName
from pathlib import Path
from datetime import datetime

logger = getLogger('app')


def test_ml_pipeline():
    config = AppConfig()
    Logging.setup_app_logger(config=config, write=True)
    method_name = MethodName.CNN1D
    tag = f'{datetime.now().strftime("%Y%m%d%H%M")}_{method_name.name}'
    results = Pipeline.cross_training(config=config, method_name=method_name, subset=2, tag=tag)

    df = results.to_df()
    print(1)

    # Pipeline.testing(config=config, method_name=MethodName.CNN1D, tag=tag)


def __main():
    test_ml_pipeline()

    # config = AppConfig()
    # Logging.setup_app_logger(config=config, write=True)
    # all_dataset = Dataset(config)
    # train_dataset = Dataset(config, mode='train')
    # test_dataset = Dataset(config, mode='test')
    # logger.info(len(all_dataset))
    # logger.info(len(train_dataset))
    # logger.info(len(test_dataset))

    # samples = list(train_dataset.proteins.keys())[:20]
    # torch_dataset = ml.torch_dataset.CNN1DDataset(samples=samples, dataset=train_dataset)
    # in_feature, target, loss_mask = torch_dataset.__getitem__(1)
    # print(1)

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


if __name__ == '__main__':
    __main()
