import random
import os
import json
import argparse

import pandas as pd
import numpy as np
import torch
import gc
import transformers
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, set_seed, \
    BitsAndBytesConfig, EarlyStoppingCallback
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformers.logging.set_verbosity_info()

# define and parse input arguments
parser = argparse.ArgumentParser(description='Script for fine-tuning models with Lora.')
parser.add_argument('model_ckpt', type=str, help='Model checkpoint to load')
parser.add_argument('data_path', type=str, help='Path to the dataset')
parser.add_argument('output_path', type=str, help='Path to save the fine-tuned model')
args = parser.parse_args()
model_ckpt = args.model_ckpt
data_path = args.data_path
output_path = args.output_path
print(model_ckpt, data_path, output_path)


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


def get_balanced_subset(df, num_samples):
    df_0 = df[df['label'] == 0]  # Select label 0
    df_1 = df[df['label'] == 1]  # Select label 1

    min_count = min(len(df_0), len(df_1), num_samples // 2)  # Ensure enough samples exist
    df_0_sample = df_0.sample(min_count, random_state=42)
    df_1_sample = df_1.sample(min_count, random_state=42)

    balanced_df = pd.concat([df_0_sample, df_1_sample]).sample(frac=1, random_state=42)  # Shuffle
    return balanced_df


def load_data(path):
    """
    Load data and remove all None entries (otherwise error at batched tokenization).

    :param path: path to load data from
    :return train_dataset: training set as pandas DataFrame, val_dataset: evaluation set as pandas DataFrame,
                test_dataset: test set as pandas DataFrame
    """
    train_dataset = pd.read_csv(f'{path}/training.csv', keep_default_na=False).rename(columns={'sentence': 'text', 'correction': 'target'})
    empty_cache()
    val_dataset = pd.read_csv(f'{path}/validation.csv', keep_default_na=False).rename(columns={'sentence': 'text', 'correction': 'target'})
    empty_cache()
    test_dataset = pd.read_csv(f'{path}/test.csv', keep_default_na=False).rename(columns={'sentence': 'text', 'correction': 'target'})
    empty_cache()

    # # Get balanced datasets (50% label=1, 50% label=0)
    # train_subset = get_balanced_subset(train_dataset, 5000)
    # val_subset = get_balanced_subset(val_dataset, 1000)
    # test_subset = get_balanced_subset(test_dataset, 4000)
    #
    # del train_dataset, val_dataset, test_dataset
    # empty_cache()
    #
    # return train_subset, val_subset, test_subset
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
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512, return_tensors='pt')


def tokenize_data(hf_train_dataset, hf_val_dataset, hf_test_dataset):
    """
    Tokenizes data with map function and sets it into torch format.

    :param hf_train_dataset:
    :param hf_val_dataset:
    :param hf_test_dataset:
    :return:
    """
    # batch_size=None: tokenize() will be applied on the full dataset as a single batch
    tokenized_hf_train_dataset = hf_train_dataset.map(tokenize, batched=True, batch_size=20000, load_from_cache_file=False)
    tokenized_hf_val_dataset = hf_val_dataset.map(tokenize, batched=True, batch_size=20000, load_from_cache_file=False)
    tokenized_hf_test_dataset = hf_test_dataset.map(tokenize, batched=True, batch_size=20000, load_from_cache_file=False)
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


def get_default_settings():
    """
    Retrieves default values of TrainingArguments and returns them.

    :return: default_lr: default learning rate, default_ne: default number of epochs, default_wr: default warmup ratio,
                default_ws: default warmup steps, default_wd: default weight decay,
                default_t_bs: default training batch size, default_e_bs: default evaluation batch size
    """
    # instantiate TrainingArguments with default settings
    default_args = TrainingArguments(output_dir='dummy')  # 'output_dir' is a required argument
    # access default values
    default_lr = default_args.learning_rate
    default_t_bs = default_args.per_device_train_batch_size
    default_e_bs = default_args.per_device_eval_batch_size
    default_ne = default_args.num_train_epochs
    default_ws = default_args.warmup_steps
    default_wr = default_args.warmup_ratio
    default_wd = default_args.weight_decay
    return default_lr, default_ne, default_wr, default_ws, default_wd, default_t_bs, default_e_bs


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


def model_init(quantization_config, output_hidden_states=False):
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
        quantization_config=quantization_config,
        output_hidden_states=output_hidden_states,
    )
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

    :param num_inits: number of random initializations, defaults to 1
    """
    initializations_accuracies = {}
    for init in range(0, num_inits):
        empty_cache()
        # set seed for each run
        set_seed_locally(62 + init * 1000)
        set_seed(62 + init * 1000)

        # define checkpoint path
        checkpoint_path = f'{output_path}/init_{init}/{model_ckpt}'
        # Check if a checkpoint exists
        resume_from_checkpoint = None
        if os.path.exists(checkpoint_path) and any(fname.startswith('checkpoint') for fname in os.listdir(checkpoint_path)):
            resume_from_checkpoint = max(
                [os.path.join(checkpoint_path, fname) for fname in os.listdir(checkpoint_path) if
                 fname.startswith('checkpoint')],
                key=os.path.getctime  # Select most recent checkpoint
            )
            print(f'Resuming training from checkpoint: {resume_from_checkpoint}')

        # quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_quant_type='nf4',  # information theoretically optimal dtype for normally distributed weights
            bnb_4bit_use_double_quant=True,  # quantize quantized weights
            bnb_4bit_compute_dtype=torch.bfloat16  # optimized fp format for ML
        )

        # init new model
        model = model_init(quantization_config)
        print(model)

        # lora config
        lora_config = LoraConfig(
            r=r,  # the dimension of the low-rank matrices
            lora_alpha=lora_alpha,  # scaling factor for LoRA activations vs pre-trained weight activations
            target_modules='all-linear',    # to get comparable results to full finetuning train all Linear layers
            lora_dropout=lora_dropout,  # dropout probability of the LoRA layers
            use_rslora=True,
            bias='none',  # whether to train bias weights, set to 'none' for attention layers
            task_type='SEQ_CLS'
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.config.pad_token_id = tokenizer.pad_token_id

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
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            disable_tqdm=False,
            log_level='info',
            logging_dir=f'{output_path}/init_{init}/logs',
            logging_steps=500,
            bf16=True,
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
            tokenizer=tokenizer
        )
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
        # train the model
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
            f.write(f'r (LORA) = {r}\n')
            f.write(f'lora_dropout = {lora_dropout}\n')
            f.write(f'lora_alpha = {lora_alpha}\n')
            f.write(f'gradient_accumulation_steps = {gradient_accumulation_steps}\n')
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
    if 'LeoLM/leo-mistral-hessianai-7b' in model_ckpt:
        learning_rate = 4e-5
        batch_size = 32
        num_epochs = 3
        weight_decay = 0.009817965092285434
        warmup_steps = 0
        warmup_ratio = 0.020068522138931667
        r = 16
        lora_alpha = 32
        lora_dropout = 0.22483624907133226
        gradient_accumulation_steps = 2
    if 'DiscoResearch/Llama3-German-8B' in model_ckpt:
        learning_rate = 4e-5
        batch_size = 32
        num_epochs = 3
        weight_decay = 0.07
        warmup_steps = 0
        warmup_ratio = 0.002
        r = 16
        lora_alpha = 32
        lora_dropout = 0.23
        gradient_accumulation_steps = 2

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    # load data from .csv files
    df_train, df_val, df_test = load_data(data_path)
    print('Loaded CSV files')
    # convert dataframe to huggingface datasets for more efficient calculations
    hf_train, hf_val, hf_test = convert_to_hf(df_train, df_val, df_test)
    print('Converted dataset to huggingface format')
    # tokenize data
    tokenized_hf_train, tokenized_hf_val, tokenized_hf_test = tokenize_data(hf_train, hf_val, hf_test)
    empty_cache()

    # run training and capture results
    training_loop(num_inits=1)
