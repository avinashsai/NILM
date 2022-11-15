import logging
logging.disable(logging.INFO) 
logging.disable(logging.WARNING)
from transformers import BertConfig, DebertaConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_model(modelname, numclasses):
    if(modelname == 'bert' or modelname == 'deberta' or modelname == 'bert-large'):
        if(modelname == 'bert'):
            model_original_name = 'bert-base-uncased'
            tokenizer = AutoTokenizer.from_pretrained(model_original_name)
        elif(modelname == 'deberta'):
            model_original_name == 'microsoft/deberta-base'
            tokenizer = AutoTokenizer.from_pretrained(model_original_name)
        elif(modelname == 'bert-large'):
            model_original_name == 'bert-large-uncased'
            tokenizer = AutoTokenizer.from_pretrained(model_original_name)

        model = AutoModelForSequenceClassification.from_pretrained(model_original_name,
                                                                output_hidden_states=False,
                                                                output_attentions=False,
                                                                num_labels=numclasses,
                                                            )
    else:
        if(modelname == 'bert-nonpretrained'):
            config = BertConfig()
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif(modelname == 'deberta-nonpretrained'):
            config = DebertaConfig()
            tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
        elif(modelname == 'bert-large-nonpretrained'):
            config = BertConfig.from_pretrained(model_original_name)
            tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
        else:
            raise ValueError('Invalid model')
    
        config.num_classes = numclasses
        model = AutoModelForSequenceClassification(config)

    return tokenizer, model