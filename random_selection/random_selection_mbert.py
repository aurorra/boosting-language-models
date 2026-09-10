"""
Script to randomly add synthetic data from pot with cap, case, verb errors together.

Example usage:
python3 ablation_study_whole_pot.py bert-base-multilingual-cased ../data/synthetic_data_verb_V3_100000 ../results/synthetic_data_verb_V3_100000_ablation/bert-base-multilingual-cased ../data/synthetic_data/combined_V3_100000_remaining_data.pickle
"""

import random
import os
import re
import argparse
import pickle
import time

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

parser = argparse.ArgumentParser(description='Script for fine-tuning models.')
parser.add_argument('model_ckpt', type=str, help='Model checkpoint to load')
parser.add_argument('data_path', type=str, help='Path to the dataset')
parser.add_argument('output_path', type=str, help='Path to save the fine-tuned model')
parser.add_argument('synthetic_data_path', type=str, help='Path to synthetic data')


args = parser.parse_args()
model_ckpt = args.model_ckpt
data_path = args.data_path
output_path = args.output_path
synthetic_data_path = args.synthetic_data_path


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


def save_checkpoint(output_path, run, start_time):
    checkpoint = {
        'run': run,
        'start_time': start_time
    }
    checkpoint_filename = f"checkpoint_run_{run}.pkl"
    checkpoint_path = os.path.join(output_path, checkpoint_filename)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(output_path):
    checkpoints = [f for f in os.listdir(output_path) if f.startswith("checkpoint_run_")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        checkpoint_path = os.path.join(output_path, latest_checkpoint)
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        run = checkpoint.get('run', 0)
        start_time = checkpoint.get('start_time', time.time())  # Use current time if not available in checkpoint
        return run, start_time
    return None, None


def load_random_synthetic_data_from_pickle(num_syn_run, synthetic_data_path):
    """
    Load synthetic data from a pickle file and randomly select samples to return.

    :param num_syn_run: Number of synthetic samples to be returned
    :param synthetic_data_path: Path to the pickle file containing synthetic data.
    :return: synthetic_df: Pandas DataFrame with the synthetic data to be added.
    """
    # Load the synthetic data pickle file
    with open(synthetic_data_path, 'rb') as f:
        synthetic_data_list = pickle.load(f)

    synthetic_data_list_adapted_format = [
        {
            'text': error['sentences'][i],
            'target': error['corrections'][i],
            'before': error['before'],
            'after': error['after'],
            'label': 1
        }
        for error in synthetic_data_list
        for i in range(len(error['sentences']))
    ]

    # Choose randomly samples
    synthetic_samples = random.sample(synthetic_data_list_adapted_format, num_syn_run)

    # Add chosen samples to already_added and append correction cases
    extended_samples = []
    for sample in synthetic_samples:
        extended_samples.append(sample)
        extended_samples.append({
            'text': sample['target'],
            'target': sample['target'],
            'before': sample['after'],
            'after': sample['after'],
            'label': 0
        })

    # Shuffle the extended samples
    random.shuffle(extended_samples)

    # Create a DataFrame
    synthetic_df = pd.DataFrame(extended_samples)

    return synthetic_df


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')


def tokenize_data(hf_dataset, tokenizer):
    tokenized_hf_dataset = hf_dataset.map(lambda batch: tokenizer(batch['text'], padding=True, truncation=True), batched=True)
    tokenized_hf_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized_hf_dataset


def convert_to_hf(dataset):
    hf_dataset = Dataset.from_pandas(dataset)
    return hf_dataset


def compute_metrics(pred):
    """
    Computes performance metrics for the predictions made by a model.

    :param pred: An object containing the model's predictions and the corresponding true labels.
                 This object should have the following attributes:
                 - label_ids: an array of actual labels.
                 - predictions: a 2D array where each row represents the logit scores for each class.
    :return: A dictionary containing the performance metrics  accuracy, F1 score, recall, and precision, calculated
                based on the true labels and the model's predictions. The F1 score is calculated with 'weighted'
                average to account for label imbalance.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    return {"accuracy": acc, "f1": f1, 'recall': recall, 'precision': precision}


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
        num_labels=2
    ).to(device)
    return m


def build_confusion_matrix(preds, labels, output_path, iteration=None):
    """
    Builds and saves a confusion matrix as a heatmap from the given predictions and actual labels.

    :param preds: A list or array of predicted labels.
    :param labels: A list or array of actual labels.
    :param output_path: The directory where the confusion matrix image will be saved.
    :param iteration: The iteration number for saving the confusion matrix file.
    """
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(labels, preds)

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Iteration {iteration}' if iteration is not None else 'Final Confusion Matrix')

    # Save the confusion matrix as an image
    file_name = f'confusion_matrix_iter_{iteration}.png' if iteration is not None else 'final_confusion_matrix.png'
    plt.savefig(f'{output_path}/{file_name}')
    plt.close()


def find_last_checkpoint(output_dir):
    ckpts = [(int(m.group(1)), d)
             for d in os.listdir(output_dir)
             if (m:=re.match(r'checkpoint[-_]?(\d+)', d))]
    return os.path.join(output_dir, max(ckpts)[1]) if ckpts else False


# Training loop
def training_loop(learning_rate, num_epochs, batch_size, warmup_ratio, warmup_steps, weight_decay, train_set, val_set,
                  test_set, tokenizer, output_path, syn_data_to_add, synthetic_data_path, num_inits=1):
    """
    Trains the model iteratively, adding synthetic data from pickle file for errors after each iteration.

    :param synthetic_data_path: Path to synthetic data pot.
    :param syn_data_to_add: Number of synthetic data to randomly add.
    :param learning_rate: Learning rate for training.
    :param num_epochs: Number of epochs for training.
    :param batch_size: Batch size for training and evaluation.
    :param warmup_ratio: Warmup ratio for learning rate scheduling.
    :param warmup_steps: Number of warmup steps.
    :param weight_decay: Weight decay for the optimizer.
    :param train_set: Initial training dataset.
    :param val_set: Validation dataset.
    :param test_set: Test dataset.
    :param tokenizer: Pretrained tokenizer for model inputs.
    :param output_path: Path to save results and models.
    :param num_inits: number of random initializations, defaults to 1
    """
    run, start_time = load_checkpoint(output_path)
    if run is None:
        run = 0
        start_time = time.time()
    print(run, start_time)

    for current_run in range(run, num_inits):
        empty_cache()
        print(f'Run {current_run}/{num_inits - 1}')
        start_time_current_run = time.time()

        if not os.path.exists(f'{output_path}/init_{current_run}/logs'):
            os.makedirs(f'{output_path}/init_{current_run}/logs')

        # set seed for each run
        current_seed = 62 + current_run * 1000
        set_seed_locally(current_seed)
        set_seed(current_seed)

        # Check if there is synthetic data (saved if not full run)
        if os.path.exists(f'{output_path}/init_{current_run}/synthetic_data.pickle'):
            with open(f'{output_path}/init_{current_run}/synthetic_data.pickle', 'rb') as f:
                synthetic_data = pickle.load(f)
            print(f'Loaded synthetic data from {output_path}/init_{current_run}/synthetic_data.pickle')
        else:
            # Randomly add synthetic data
            synthetic_data = load_random_synthetic_data_from_pickle(syn_data_to_add, synthetic_data_path)
            with open(f'{output_path}/init_{current_run}/synthetic_data.pickle', 'wb') as f:
                pickle.dump(synthetic_data, f)
        if len(synthetic_data) != syn_data_to_add * 2:
            print(f'Not the correct amount of samples! Should be {syn_data_to_add * 2} but is {len(synthetic_data)}')
            exit()
        print(f'Adding {len(synthetic_data)} randomly selected synthetic samples.')

        # Combine the original training data with the new synthetic data by concatenating DataFrames
        train_set_df = pd.concat([train_set.to_pandas(), synthetic_data], ignore_index=True)
        # Shuffle the combined DataFrame -> additional seed injection to make sure we have the same training set as
        # previous run if continued from checkpoint
        train_set_df = train_set_df.sample(frac=1, random_state=current_seed).reset_index(drop=True)
        # Re-tokenize the combined dataset for HuggingFace Trainer
        train_set = tokenize_data(convert_to_hf(train_set_df), tokenizer)  # Re-tokenize the new augmented dataset
        print('New train dataset:')
        print(train_set)

        # init new model
        model = model_init()

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'{output_path}/{model_ckpt}',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
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
            logging_dir=f'{output_path}/init_{current_run}/logs',
            logging_steps=500,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model='recall',
        )
        # Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        # Train the model, try to find last checkpoint, if not found, initialize training from start
        prev_model_ckpt = find_last_checkpoint(f'{output_path}/{model_ckpt}')
        trainer.train(resume_from_checkpoint=prev_model_ckpt)
        empty_cache()

        # save logs
        history = pd.DataFrame(trainer.state.log_history)
        history.to_csv(f'{output_path}/init_{current_run}/logs/log_history.csv')
        empty_cache()

        # Predict on test set
        test_predictions = trainer.predict(test_set)

        end_time_current_run = time.time()
        elapsed_time_current_run = end_time_current_run - start_time_current_run

        # Save results
        with open(f'{output_path}/init_{current_run}/results.csv', 'w') as f:
            for key, value in test_predictions.metrics.items():
                f.write(f'{key},{value}\n')
            f.write('Parameters used:\n')
            f.write(f'learning_rate = {learning_rate}\n')
            f.write(f'num_epochs = {num_epochs}\n')
            f.write(f'batch_size = {batch_size}\n')
            f.write(f'weight_decay = {weight_decay}\n')
            f.write(f'warmup_steps = {warmup_steps}\n')
            f.write(f'warmup_ratio = {warmup_ratio}\n')
            f.write(f'seed = {current_seed}\n')
            f.write(f'number of samples in training set = {len(train_set)}\n')
            f.write(f'number of samples in validation set = {len(val_set)}\n')
            f.write(f'Runtime: {int(elapsed_time_current_run) // 3600} h, {int((elapsed_time_current_run % 3600) // 60)} min, {elapsed_time_current_run % 60} s')
        y_pred = np.argmax(test_predictions.predictions, axis=1)
        np.save(f'{output_path}/init_{current_run}/y_pred.npy', y_pred)
        build_confusion_matrix(test_predictions.predictions.argmax(-1), test_predictions.label_ids, f'{output_path}/init_{current_run}', iteration=current_run)

        # Save the checkpoint after each run
        save_checkpoint(output_path, current_run + 1, start_time)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save final results
    with open(f'{output_path}/final_results.csv', 'w') as f:
        f.write('Parameters used:\n')
        f.write(f'learning_rate = {learning_rate}\n')
        f.write(f'num_epochs = {num_epochs}\n')
        f.write(f'batch_size = {batch_size}\n')
        f.write(f'weight_decay = {weight_decay}\n')
        f.write(f'warmup_steps = {warmup_steps}\n')
        f.write(f'warmup_ratio = {warmup_ratio}\n')
        f.write(f'n_inits = {n_inits}\n')
        f.write(f'number of samples in training set = {len(train_set)}\n')
        f.write(f'number of samples in validation set = {len(val_set)}\n')
        f.write(f'Runtime: {int(elapsed_time) // 3600} h, {int((elapsed_time % 3600) // 60)} min, {elapsed_time % 60} s')


if __name__ == '__main__':
    # empty cache and set seed (reproducibility)
    empty_cache()
    seed = 62
    set_seed_locally(seed)
    set_seed(seed)

    # define hyperparameters
    learning_rate = 11e-06
    batch_size = 64
    num_epochs = 20
    weight_decay = 0
    warmup_steps = 0
    warmup_ratio = 0
    n_inits = 1

    # synthetic data to add
    if 'real_word' in data_path:
        syn_data_nums = int((350336 - 50000) / 2)
    elif 'verb' in data_path:
        syn_data_nums = int(72564 / 2)
    elif 'case' in data_path:
        syn_data_nums = int(126960 / 2)
    elif 'cap' in data_path:
        syn_data_nums = int(94554 / 2)
    else:
        print('Cannot add synthetic data for this error type.')
        exit(1)

    # create output path if it does not exist yet
    output_path = f'{output_path}/n_inits={n_inits}_num_samples={syn_data_nums}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load data and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    df_train = pd.read_csv(f'{data_path}/training.csv', keep_default_na=False).rename(columns={'sentence': 'text', 'correction': 'target'})
    df_val = pd.read_csv(f'{data_path}/validation.csv', keep_default_na=False).rename(columns={'sentence': 'text', 'correction': 'target'})
    df_test = pd.read_csv(f'{data_path}/test.csv', keep_default_na=False).rename(columns={'sentence': 'text', 'correction': 'target'})

    # Convert datasets
    train_set = tokenize_data(convert_to_hf(df_train), tokenizer)
    val_set = tokenize_data(convert_to_hf(df_val), tokenizer)
    test_set = tokenize_data(convert_to_hf(df_test), tokenizer)

    # Run the training loop
    training_loop(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        tokenizer=tokenizer,
        output_path=output_path,
        syn_data_to_add=syn_data_nums,
        synthetic_data_path=synthetic_data_path,
        num_inits=n_inits,
    )

