# Data

## Directory structure

```
data/
    input/
        download_input_data.sh
        af2/
            dist_maps_af2/
            pdb_structures/
            collabfold_log.txt
        biolip/
            development_set/
        bindembed_embeddings.h5
```

## Data description

- `data/input`: This dataset includes AlphaFold2 predictions, ProtT5 embeddings and BioLiP annotations for the proteins
  used in the [bindEmbed21](https://github.com/Rostlab/bindPredict) method and was downloaded from external sources.
  The data can be downloaded using the script `data/input/download_input_data.sh`.
- `data/input/af2/dist_maps_af2`: This directory includes distance maps produced from the AF2 structures. Each distance
  map is a 3-dimensional array of shape LxLx2,
  where L is the length of the corresponding protein. The 3rd dimension refers to 2 kinds of distances: The distance
  between the C-alpha and the C-beta atoms.
- `data/input/af2/pdb_structures`: This directory includes structures(`.pdb` files) as predicted by AF2. The second to
  last column gives you the per-residue pLDDT score.
- `data/input/af2/collabfold_log.txt`: This is the log file for the colabfold run.
- `data/input/biolip`: This directory includes BioLiP annotations.
- `data/input/bindembed_embeddings.h5`:
- `data/output`: This data is produced by this package.
