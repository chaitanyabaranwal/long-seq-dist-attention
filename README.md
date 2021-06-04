# Sequence Parallelism

We implemented sequende parallelism based on [Megatron](https://github.com/NVIDIA/Megatron-LM) proposed by NVIDIA. We provide some scripts to run the experiments as follows.

## Prepare Dataset

```
# download raw data
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# prepare extraction scripts
git clone https://github.com/attardi/wikiextractor.git
cd ./wikiextractor
pip install .

# extract
wikiextractor --json enwiki-latest-pages-articles.xml.bz2
cat text/*/* > ./corpus


# preprocess some data
cd ./long-seq-dist-attention
python tools/preprocess_data.py \
    --input <PATH_TO>/corpus \
    --output-prefix my-bert \
    --vocab  <PATH_TO>/bert-large-uncased-vocab.txt \
    --dataset-impl mmap \
    --tokenizer-type BertWordPieceLowerCase \
    --split-sentences
```

## PyTorch Image

As Megatron needs PyTorch 1.8 and above, we used a Singularity container to run the experiments. To pull the image, you can do as follows:

```bash
singularity build pytorch18.simg docker://nvcr.io/nvidia/pytorch:20.12-py3
```

## Run an Experiment

You can launch an experiment with the following command.

```bash
cd ./long-seq-dist-attention
bash ./demo/mnode_bert.sh \
       <number of processes> \
       <pipeline parallel size> \
       <sequence parallel size> \
       <sequence length> \
       <micro batch size> \
       <global batch size> \
       <number of layers> \
       <hidden size> \
       <number of attention heads>
```
