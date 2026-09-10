import random
import os
import json
import argparse
import logging

import pandas as pd
import numpy as np
import torch
import gc
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, set_seed, \
    DataCollatorWithPadding
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device used:', device)

# check if CUDA is available, exit with error message if not
logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)
if not torch.cuda.is_available():
    log.error('CUDA is not available. Quitting execution...')
    exit(1)

parser = argparse.ArgumentParser(description='Script for fine-tuning models.')
parser.add_argument('model_ckpt', type=str, help='Model checkpoint to load')
parser.add_argument('data_path', type=str, help='Path to the dataset')
parser.add_argument('output_path', type=str, help='Path to save the fine-tuned model')

args = parser.parse_args()
model_ckpt = args.model_ckpt
data_path = args.data_path
output_path = args.output_path
print(f'Parameters:\nmodel_ckpt = {model_ckpt}\ndata_path = {data_path}\noutput_path = {output_path}')


def empty_cache():
    """ Free gpu memory. """
    gc.collect()
    torch.cuda.empty_cache()


def set_seed_locally(seed=None):
    """
    Set all seeds to make results reproducible (deterministic mode).
    When seed is None, disables deterministic mode.

    :param seed: an integer to your choosing
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def load_data(path):
    """
    Load training, validation, and test data from CSV files.

    :param path: path to load data from
    :return train_dataset: training set as pandas DataFrame, val_dataset: evaluation set as pandas DataFrame,
                test_dataset: test set as pandas DataFrame
    """
    train_dataset = pd.read_csv(f'{path}/training.csv', keep_default_na=False).rename(columns={'sentence': 'text', 'correction': 'target'})
    val_dataset = pd.read_csv(f'{path}/validation.csv', keep_default_na=False).rename(columns={'sentence': 'text', 'correction': 'target'})
    test_dataset = pd.read_csv(f'{path}/test.csv', keep_default_na=False).rename(columns={'sentence': 'text', 'correction': 'target'})
    return train_dataset, val_dataset, test_dataset


def convert_to_hf(train_dataset, val_dataset, test_dataset):
    """
    Convert pandas DataFrame to a HuggingFace Dataset.

    :param train_dataset: training set as pandas DataFrame
    :param val_dataset: evaluation set as pandas DataFrame
    :param test_dataset: test set as pandas DataFrame
    :return: hf_train_dataset: training set as a HF Dataset, hf_val_dataset: validation dataset as a HF Dataset,
                hf_test_dataset: test dataset as a HF Dataset
    """
    hf_train_dataset = Dataset.from_pandas(train_dataset).remove_columns(['documentID', 'target'])
    hf_val_dataset = Dataset.from_pandas(val_dataset).remove_columns(['documentID', 'target'])
    hf_test_dataset = Dataset.from_pandas(test_dataset).remove_columns(['documentID', 'target'])
    return hf_train_dataset, hf_val_dataset, hf_test_dataset


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')


def tokenize_data(hf_train_dataset, hf_val_dataset, hf_test_dataset):
    """
    Tokenizes data with map function and sets it into torch format.
    """
    # batch_size=None: tokenize() will be applied on the full dataset as a single batch
    tokenized_hf_train_dataset = hf_train_dataset.map(tokenize, batched=True, batch_size=None)
    tokenized_hf_val_dataset = hf_val_dataset.map(tokenize, batched=True, batch_size=None)
    tokenized_hf_test_dataset = hf_test_dataset.map(tokenize, batched=True, batch_size=None)
    print(f'Training dataset after tokenization:\n{tokenized_hf_train_dataset}')
    print(f'Validation dataset after tokenization:\n{tokenized_hf_val_dataset}')
    print(f'Testing dataset after tokenization:\n{tokenized_hf_test_dataset}')
    empty_cache()
    # model expects tensors as inputs -> convert the input_ids and attention_mask columns to the "torch" format
    tokenized_hf_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_hf_val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_hf_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    print(f'Training dataset after setting format:\n{tokenized_hf_train_dataset}')
    print(f'Validation dataset after setting format:\n{tokenized_hf_val_dataset}')
    print(f'Testing dataset after setting format:\n{tokenized_hf_test_dataset}')
    return tokenized_hf_train_dataset, tokenized_hf_val_dataset, tokenized_hf_test_dataset


def compute_metrics(pred):
    """
    Computes performance metrics for the predictions made by a model.

    :param pred: An object containing the model's predictions and the corresponding true labels.
                 This object should have the following attributes:
                 - label_ids: an array of actual labels.
                 - predictions: a 2D array where each row represents the logit scores for each class.
    :return: A dictionary containing the performance metrics  accuracy, F1 score, recall, and precision, calculated
                based on the true labels and the model's predictions.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='binary', pos_label=1)
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='binary', pos_label=1)
    precision = precision_score(labels, preds, average='binary', pos_label=1)
    return {'accuracy': acc, 'f1': f1, 'recall': recall, 'precision': precision}


def model_init(output_hidden_states=False):
    """
    Initializes and returns a pre-trained model for sequence classification tasks.

    This function creates an instance of a pre-trained model from the Hugging Face library.
    The model is loaded based on a specified model checkpoint.

    :param output_hidden_states: If set to True, the model will also output hidden states
            (needed for feature extraction). Default is False.
    :return: A pre-trained model instance for sequence classification, moved to the appropriate computing device
                (CPU or GPU).

    Note:
    - The model checkpoint and the computing device (referred to as 'model_ckpt' and 'device' respectively)
      are expected to be defined outside this function.
    - The model is configured for binary classification ('num_labels=2'). For different classification tasks,
      modify the 'num_labels' parameter accordingly.
    - This implementation is necessary when we want to achieve reproducible results. See, e.g.:
      https://discuss.huggingface.co/t/multiple-training-will-give-exactly-the-same-result-except-for-the-first-time/8493
    """
    m = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=2,
        output_hidden_states=output_hidden_states
    ).to(device)
    return m


def build_confusion_matrix(dataset, predictions, path):
    """
    Builds and saves a confusion matrix as a heatmap from the given predictions and actual labels.

    This function calculates the confusion matrix using the actual labels and the predicted labels from a dataset.
    It then generates a heatmap visualization of the confusion matrix using seaborn and saves it as an image file.

    :param dataset: A dataset object containing 'label' as one of its keys, representing the actual labels.
    :param predictions: A list or array of predictions corresponding to the entries in the dataset.
    :param path: The file path where the confusion matrix image will be saved.
    """
    conf = confusion_matrix(dataset['label'], predictions)
    sns.heatmap(conf, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'{path}/confusion_matrix.png')
    plt.clf()  # clear figure


def training_loop(num_inits=1):
    """
    Loops through a list of values for a hyperparameter. At each iteration we initialize a new model, train it,
    and write the results into a text file.
    """
    initializations_accuracies = {}
    for init in range(0, num_inits):
        empty_cache()
        os.makedirs(f'{output_path}/init_{init}/logs', exist_ok=True)
        # set seed for each run
        set_seed_locally(62 + init * 1000)
        set_seed(62 + init * 1000)
        # init new model
        model = model_init()
        # Initialize the data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # set up training arguments
        training_args = TrainingArguments(
            output_dir=f'{output_path}/init_{init}/{model_ckpt}',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,  # limit to save only one checkpoint to manage disk space usage
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            disable_tqdm=False,
            log_level='info',
            logging_dir=f'{output_path}/init_{init}/logs',
            logging_steps=500,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )
        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=tokenized_hf_train,
            eval_dataset=tokenized_hf_val,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        # train the model
        trainer.train()
        print(trainer.state.log_history)
        # save logs
        history = pd.DataFrame(trainer.state.log_history)
        history.to_csv(f'{output_path}/init_{init}/log_history.csv')
        empty_cache()
        trainer.model.save_pretrained(f'{output_path}/init_{init}/{model_ckpt}-final')
        # prediction and result saving
        preds_output = trainer.predict(tokenized_hf_test)
        with open(f'{output_path}/init_{init}/results.csv', 'w') as f:
            for key, value in preds_output.metrics.items():
                f.write(f'{key},{value}\n')
            f.write('Parameters used:\n')
            f.write(f'learning_rate = {learning_rate}\n')
            f.write(f'num_epochs = {num_epochs}\n')
            f.write(f'batch_size = {batch_size}\n')
            f.write(f'weight_decay = {weight_decay}\n')
            f.write(f'warmup_steps = {warmup_steps}\n')
            f.write(f'warmup_ratio = {warmup_ratio}\n')
            f.write(f'Seed used: {62 + init * 1000}')
        y_pred = np.argmax(preds_output.predictions, axis=1)
        np.save(f'{output_path}/init_{init}/y_pred.npy', y_pred)
        accuracy = preds_output.metrics['test_accuracy']
        # build and save confusion matrix
        build_confusion_matrix(tokenized_hf_test, y_pred, f'{output_path}/init_{init}')
        initializations_accuracies[f'init_{init}'] = accuracy
        print(f'Initialization {init + 1} with accuracy={accuracy}')

    with open(f'{output_path}/initialization_results.csv', 'w') as f:
        f.write(f'Accuracies for each initialization: {json.dumps(initializations_accuracies, indent=4)}')


if __name__ == '__main__':
    # create output path if it does not exist yet
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # empty cache and set seed (reproducibility)
    empty_cache()
    set_seed_locally(62)
    set_seed(62)

    # define hyperparameters
    learning_rate = 11e-06
    batch_size = 64
    num_epochs = 20
    weight_decay = 0
    warmup_steps = 0
    warmup_ratio = 0

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    # load data from .csv files
    df_train, df_val, df_test = load_data(data_path)
    # convert dataframe to huggingface datasets for more efficient calculations
    hf_train, hf_val, hf_test = convert_to_hf(df_train, df_val, df_test)
    # tokenize data
    tokenized_hf_train, tokenized_hf_val, tokenized_hf_test = tokenize_data(hf_train, hf_val, hf_test)

    # run training
    training_loop(num_inits=1)
