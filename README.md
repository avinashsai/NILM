# NILM

![Alt text](https://github.com/avinashsai/NILM/blob/main/docs/nilm.png)

This is the code for the EMNLP'22 Findings paper [What do Large Language Models Learn beyond Language?](https://arxiv.org/pdf/2210.12302.pdf).

Pretrained Language Models (LMs) have shown singular succcess on a range of natural language understandings tasks, to the extent that they have become foundational for contemporary NLP systems. In this paper, we explore whether pretraining on text is inherently about learning language, or if pretraining also imbues LMs with skills for symbolic manipulation and non-linguistic reasoning. We call this NILM (measuring
Non-linguistic Inductive bias in Language Models). For this analysis, we create a set of 19 tasks from three categories of task paradigms: quantitative computation, recognizing regular expressions, and string reasoning. 

# Set-up

``` 
git clone https://github.com/avinashsai/NILM.git
cd NILM && pip install -r requirements.txt
```

# Code

There are two types of results in this paper: 

- Figures 2-7 and Appendix figures compares the pre-trained and non-pretrained models. To reproduce these results:
```
python main.py --modelname (Eg: bert) (pre-trained: bert/deberta/bert-large) (Non-pretrained: bert-nonpretrained/deberta-nonpretrained/bert-large-nonpretrained)
               --tasks (Eg: anagram palindrome mean) (List of tasks can be found in the data folder)
```

- Tables 2 and 3 explores the effect of pre-training data. To reproduce these results:

  - Download the data files from https://drive.google.com/drive/folders/1NFKisDdlECnuMaSrdPRqOaFyiVjZcPVh?usp=share_link and put them in **data/** folder.
  - Pretrain the BERT/DeBERTa model. We experiment with **straight (normal order)/shuffle/sort**. Replace the appropriate type in the **typ** variable.
  ```
  python bert_pretrain.py or python deberta_pretrain.py
  ```
  - Once the pre-training is done, run:
  ```
  python data_bias.py --modelname (Eg: bert_amazon_500k_straight_12_12_768/bert_snli_1000k_shuffle_12_12_768)
  ```
