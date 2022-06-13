#!/bin/bash

set -e

# Install subversion
sudo apt install subversion

# AlphaFold2 files
mkdir -p af2
cd af2
wget 'http://www.rostlab.org/~bindpredict/bindembed_distance_maps.tar.gz'
tar -xzvf bindembed_distance_maps.tar.gz
rm bindembed_distance_maps.tar.gz
wget 'https://www.rostlab.org/~bindpredict/collabfold_log.txt'
wget 'www.rostlab.org/~bindpredict/af2_structures.tar.gz'
tar -xzvf af2_structures.tar.gz
rm af2_structures.tar.gz
cd ..

# Embeddings
wget 'http://www.rostlab.org/~bindpredict/bindembed_embeddings.h5'

# sequences, splits, BioLiP
svn checkout 'https://github.com/Rostlab/bindPredict/trunk/data/development_set'
mkdir -p biolip splits
cp development_set/ids_split*.txt development_set/uniprot_test.txt splits
cp development_set/all.fasta sequences.fasta
cp development_set/binding_residues*.txt biolip

