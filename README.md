# Unsupervised Chunking with Hierarchical Recurrent Neural Networks

## ENV Requirement
```
torch
pytorch-lightning=1.6.5
transformers=4.14.1
```

## Dataset

Download the datasets from this [link](https://drive.google.com/drive/folders/1p4RMlWT9L9rUdYRw_ZIdpoMU7CR84J0P?usp=sharing)

## Training and Testing

For the summarization task:
```
bash train_summarization.sh
```

For the MNLI-Gen task:
```
bash train_paraphrasing.sh
```

For the translation task:
```
bash train_translation.sh
```
