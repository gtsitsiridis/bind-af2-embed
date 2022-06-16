from data.dataset import Dataset
from config import AppConfig


def __main():
    dataset = Dataset(AppConfig())
    i = 0
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
    proteins = dataset.proteins
    prot = proteins['P27797']
    # prot.show_umap()
    prot.show_structure(color='ligand')
    pass


if __name__ == '__main__':
    __main()
