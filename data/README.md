# Data

## Directory structure

```
data/
    input/
        af2/
            dist_maps_af2/
            pdb_structures/
            collabfold_log.txt
        biolip/
        splits/
        sequences.fasta
        bindembed_embeddings.h5
        download_input_data.sh
```

## Data description

- `data/input/`: This dataset includes AlphaFold2 predictions, ProtT5 embeddings and BioLiP annotations for the proteins
  used in the [bindEmbed21](https://github.com/Rostlab/bindPredict) method and was downloaded from external sources.
  The data can be downloaded using the script `data/input/download_input_data.sh`.
- `data/input/af2/dist_maps_af2/`: This directory includes distance maps produced from the AF2 structures. Each distance
  map is a 3-dimensional array of shape LxLx2,
  where L is the length of the corresponding protein. The 3rd dimension refers to 2 kinds of distances: The distance
  between the C-alpha and the C-beta atoms.
- `data/input/af2/pdb_structures/`: This directory includes structures(`.pdb` files) as predicted by AF2. The second to
  last column gives you the per-residue pLDDT score.
- `data/input/af2/collabfold_log.txt`: This is the log file for the colabfold run.
- `data/input/biolip/`: This directory includes BioLiP annotations for metal, nuclear and small molecule binding residues. 
- `data/input/splits/`: This directory includes different splits of the dataset. Each file contains uniprot IDs.
- `data/input/sequences.fasta`: This files contains the sequences of all the proteins in the dataset.
- `data/input/bindembed_embeddings.h5`:
  This file includes 1024-dimensional vectors for each residue from the last hidden layer of ProtT5-XL-UniRef5028
  (ProtT5). The key is the Uniprot ID and the value is the Embedding of the corresponding protein.
- `data/output/`: This data is produced by this package.
