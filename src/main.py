from dataset import Dataset


def __main():
    embeddings_file = '../data/bindembed_embeddings.h5'
    fasta_file = '../data/biolip/development_set/all.fasta'
    splits_ids_files = [f'../data/biolip/development_set/ids_split{i}.txt' for i in range(1, 6)]
    binding_residues_file_dict = {x: f'../data/biolip/development_set/binding_residues_2.5_{x}.txt' for x in
                                  ['small', 'metal', 'nuclear']}
    distogram_dir = '../data/af2/dist_maps_af2'

    dataset = Dataset(embeddings_file=embeddings_file,
                      splits_ids_files=splits_ids_files,
                      fasta_file=fasta_file,
                      binding_residues_file_dict=binding_residues_file_dict,
                      distogram_dir=distogram_dir)
    ids = dataset.get_ids()
    fold_array = dataset.get_fold_array()
    embeddings = dataset.get_embeddings()
    sequences = dataset.get_sequences()
    labels = dataset.get_labels()
    pass


if __name__ == '__main__':
    __main()
