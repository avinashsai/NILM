import logging
logging.disable(logging.INFO) 
logging.disable(logging.WARNING)
import pandas as pd
from sklearn.model_selection import *
from transformers import Trainer, TrainingArguments

from utils import tokenize_sentences, compute_metrics, TaskDataset, load_portion

def train_model(train_data, dev_data, test_data, task, max_sen_len, resultspath, _tokenizer, base_model):
	samples = []
	trainaccuracies = []
	trainf1scores = []
	testaccuracies = []
	testf1scores = []
	alltestaccuracies = []
	portion = 0.001
	total_sample_size = 0.002
	numepochs = 1
	while(portion <= total_sample_size):
		trainaccuracy = 0.0
		trainf1_score = 0.0
		testaccuracy = 0.0
		testf1_score = 0.0
		eachrunaccuracy = []
		sample_size = int(portion * len(train_data))

		print("Sample Size {} ".format(sample_size))
		if(sample_size < 1000):
			n_runs = 10
		elif(sample_size >=1000):
			n_runs = 3

		if(sample_size == 10):
			batch_size = 2
		elif(sample_size == 20):
			batch_size = 4
		elif(sample_size == 40 or sample_size == 80):
			batch_size = 8
		else:
			batch_size = 16

		for runs in range(n_runs):
			train_text = train_data.text.tolist()
			train_labels = train_data.label.tolist()

			dev_text = dev_data.text.tolist()
			dev_labels = dev_data.label.tolist()

			test_text = test_data.text.tolist()
			test_labels = test_data.label.tolist()

			if(portion != 1.0):
				train_text, train_labels = load_portion(train_text, train_labels, portion_size=portion)

			train_inputids, train_attnmasks = tokenize_sentences(_tokenizer, train_text, max_sen_len)
			dev_inputids, dev_attnmasks = tokenize_sentences(_tokenizer, dev_text, max_sen_len)
			test_inputids, test_attnmasks = tokenize_sentences(_tokenizer, test_text, max_sen_len)

			train_loader = TaskDataset(train_inputids, train_attnmasks, train_labels)
			dev_loader = TaskDataset(dev_inputids, dev_attnmasks, dev_labels)
			test_loader = TaskDataset(test_inputids, test_attnmasks, test_labels)

			training_args = TrainingArguments(
									output_dir=' ',
									num_train_epochs=numepochs,
									per_device_train_batch_size=batch_size,
									per_device_eval_batch_size=32,
									warmup_steps=500,
									weight_decay=0.01,
									logging_dir=' ',
									logging_steps=10000000,
									save_steps=1000000
									)

			trainer = Trainer(
							model=base_model,
							args=training_args,
							train_dataset=train_loader,
							eval_dataset=test_loader,
							compute_metrics=compute_metrics
							)
			trainer.train()
			testscores = trainer.evaluate()
			testaccuracy += testscores['eval_accuracy']
			testf1_score += testscores['eval_f1']
			trainscores = trainer.evaluate(train_loader)
			trainaccuracy += trainscores['eval_accuracy']
			trainf1_score += trainscores['eval_f1']
			eachrunaccuracy.append(testscores['eval_accuracy'])

		samples.append(sample_size)
		trainaccuracies.append(round((trainaccuracy/n_runs), 2))
		trainf1scores.append(round((trainf1_score/n_runs), 2))
		testaccuracies.append(round((testaccuracy/n_runs), 2))
		testf1scores.append(round((testf1_score/n_runs), 2))
		alltestaccuracies.append(eachrunaccuracy)

		if(sample_size == 5120):
			portion = 0.5
			portion = round((portion + 0.1), 2)
		elif(sample_size > 5120):
			portion = round((portion + 0.1), 2)
		else:
			portion = portion * 2

	df_train = pd.DataFrame({'Samples':samples, 'Accuracy': trainaccuracies, 'F1_score':trainf1scores})
	df_train.to_excel(resultspath + '/' + task + '_' + 'train_scores_' + str(numepochs) + '.xlsx', header=True, index=False)

	df_test = pd.DataFrame({'Samples':samples, 'Accuracy': testaccuracies, 'F1_score':testf1scores, 'Accuracy_list': alltestaccuracies})
	df_test.to_excel(resultspath + '/' + task + '_' + 'test_scores_' + str(numepochs) + '.xlsx', header=True, index=False)
