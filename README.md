This repository extends from the original [Sequence Parallelism](https://arxiv.org/abs/2105.13120) code. A summarised brief of the model architecture can be found [here](https://chaitanyabaranwal.github.io/long-seq-dist-attention).

# Sequence Parallelism with Linear Attention

[Sequence Parallelism](https://arxiv.org/abs/2105.13120) is a method to train long sequences in Transformers by splitting the sequence and distributing it to different devices. This codebase implements Sequence Parallelism based on the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) proposed by NVIDIA. It also integrates linear complexity models of [Linformer](https://arxiv.org/abs/2006.04768) and [Big Bird](https://arxiv.org/abs/2007.14062) with Sequence Parallelism, making it possible to handle longer sequences than was possible either by just the linear complexity Transformers or just by full-attention Sequence Parallelism.

# Setup

Megatron-LM has been with python 3.8, pytorch 1.8, cuda 11.1. You can use a Docker or Singularity container to run the experiments. To pull the image, you can do as follows:

```bash
singularity build pytorch18.simg docker://nvcr.io/nvidia/pytorch:20.12-py3
```

To use this repository, please install the latest supported versions of PyTorch with GPU support (python 3.8, pytorch 1.8, cuda 11.1, and nccl 2.8.3 and above) and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start). Data preprocessing requires [NLTK](https://www.nltk.org/install.html), though this is not required for training, evaluation, or downstream tasks.

## Prepare Dataset

We recommend following the Wikipedia data extraction process specified by Google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text."

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
```

## Preprocessing Data

The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for BERT training is:
```bash
python tools/preprocess_data.py \
       --input <PATH_TO>/corpus \
       --output-prefix my-bert \
       --vocab <PATH_TO>/bert-large-uncased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
```

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data-path` specified in later BERT training is the full path and new filename, but without the file extension. **In the preprocessing script, use the `--workers <NUM_WORKERS>` flag to use multiple CPUs for speeding up the preprocessing.**

## BERT Pretraining

**For now, Sequence Parallelism with/without linear complexity attention has only been implemented for the encoder layer, which means it can be used only for pretraining BERT**.

You can launch a single-node experiment with the following command.

```bash
cd ./long-seq-dist-attention
bash ./scripts/bert_distributed.sh \
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

To run multi-node training, an example has been given for the CSCS machine which uses a SLURM scheduler. Other single-node scripts can be changed accordingly.

```bash
cd ./long-seq-dist-attention
bash ./scripts/bert_distributed.sh \
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

The Linformer model can be trained using the `bert_distributed_linformer.sh` script, which expects an additional argument for the Linformer projected dimension.

```bash
cd ./long-seq-dist-attention
bash ./scripts/bert_distributed_linformer.sh \
       <number of processes> \
       <pipeline parallel size> \
       <sequence parallel size> \
       <sequence length> \
       <micro batch size> \
       <global batch size> \
       <Linformer projected dimension> \
       <number of layers> \
       <hidden size> \
       <number of attention heads>
```

The Big Bird model can be trained using the `bert_distributed_bigbird.sh` script, which expects an additional argument for the Big Bird block size.

```bash
cd ./long-seq-dist-attention
bash ./scripts/bert_distributed_bigbird.sh \
       <number of processes> \
       <pipeline parallel size> \
       <sequence parallel size> \
       <sequence length> \
       <micro batch size> \
       <global batch size> \
       <Big Bird block size> \
       <number of layers> \
       <hidden size> \
       <number of attention heads>
```

**The above scripts are just convenience scripts to execute the actual code scripts in the `examples` directory. Make sure that the scripts in that folder are correctly set, and represent the nodes, GPUs and node IDs correctly.** We use [PyTorch distributed launcher](https://pytorch.org/docs/stable/distributed.html) for distributed training, which has a standard format for multi-node and single-node (multi-GPU) training. For example, calling the Big Bird implementation instead of the full-attention implementations requires supplying the `--bigbird` and `--block-size <BLOCK_SIZE>` flags, while calling the Linformer implementation requires supplying the `--linformer-k <PROJECTION_DIM>` flag. A collection of all customizable arguments can be found in the `megatron/arguments.py` file.
