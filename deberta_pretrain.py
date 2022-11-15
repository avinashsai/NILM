import os
import random
import numpy as np
import torch
import torch.utils.data
from transformers import DebertaConfig
from transformers import DebertaForMaskedLM
from transformers import DebertaTokenizer
from transformers import Trainer, TrainingArguments
from tokenizers.processors import BertProcessing
from tokenizers.processors import BertProcessing
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from tokenizers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Task (can be snli or amazon)
task = 'amazon'

# Type (can be straight or sort or shuffle)
typ = 'shuffle'

# Hyper parameters
if(task == 'amazon'):
    count = '500k'
else:
    count = '1000k'
layers = 12
hiddendim = 768
heads = 12
intermediate = 4 * hiddendim
min_word_freq = 2

save_path = 'deberta_' + task + '_'+ count + '_' + typ + '_' + str(heads) + '_' + str(layers) + '_' + str(hiddendim) + "/"

# Path to save the pre-trained model
if(os.path.exists(save_path) == False):
    os.mkdir(save_path)

# Load the training files
paths = 'data/' + task + '/' + task + '_train_data_' + count + '_' + typ + '.txt'

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=[paths], vocab_size=52000, min_frequency=min_word_freq, special_tokens=[
    "<PAD>",
    "<CLS>",
    "<SEP>",
    "<UNK>",
])

# Save the tokenizer to the path
tokenizer.save_model(save_path)

tokenizer = ByteLevelBPETokenizer(
    save_path + "vocab.json",
    save_path + "merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("<SEP>", tokenizer.token_to_id("<SEP>")),
    ("<CLS>", tokenizer.token_to_id("<CLS>")),
)

maxlength = 100
tokenizer.enable_truncation(max_length=maxlength)

config = DebertaConfig(
    vocab_size=52000,
    max_position_embeddings=maxlength + 2,
    type_vocab_size=1,
)

tokenizer = DebertaTokenizer.from_pretrained(save_path, max_len=maxlength)
model = DebertaForMaskedLM(config=config)
model.resize_token_embeddings(len(tokenizer))

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=paths,
    block_size=maxlength,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=save_path,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=10000000,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(save_path)
