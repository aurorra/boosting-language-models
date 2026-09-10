"""
Script to strategically add synthetic data.

Example usage:
python3 language-model_boosting.py bert-base-multilingual-cased ../data/boosting_V2/synthetic_data_real_word_V3_100000 ../results/boosting_V2/synthetic_data_real_word_V3_100000/bert-base-multilingual-cased ../data/boosting_V2/synthetic_data/real_word_V3_100000_remaining_data.pickle
"""

import random
import os
import json
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


def save_checkpoint(run, model, trainer, optimizer, already_added, train_set, output_path):
    checkpoint_path = f"{output_path}/checkpoint_run_{run}.pt"
    checkpoint = {
        'run': run,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'trainer_state_dict': trainer.state,
        'already_added': already_added,
        'train_set': train_set.to_pandas(),  # Save the current state of the training set
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Saved checkpoint to {f"{output_path}/checkpoint_run_{run}.pt"}')


def load_checkpoint(output_path):
    checkpoints = [f for f in os.listdir(output_path) if f.startswith("checkpoint_run_")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        checkpoint_path = os.path.join(output_path, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        run = checkpoint['run']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        trainer_state_dict = checkpoint['trainer_state_dict']
        already_added = checkpoint['already_added']
        train_set_df = checkpoint['train_set']  # Load the training set
        return run, model_state_dict, optimizer_state_dict, trainer_state_dict, already_added, train_set_df
    return None, None, None, None, set(), None


def load_synthetic_data_from_pickle(errors, synthetic_data_path, num_samples=None, already_added=None):
    """
    Load synthetic data from a pickle file corresponding to the 'before' and 'after' errors made by the model.

    :param errors: List of tuples with incorrect (before, after) pairs.
    :param synthetic_data_path: Path to the pickle file containing synthetic data.
    :param num_samples: Maximum number of synthetic samples to load for each error. If None, load all samples.
    :param already_added: Set of (before, after) pairs that have already been added to avoid duplicates.
    :return: synthetic_df: Pandas DataFrame with the synthetic data to be added.
    """
    if already_added is None:
        already_added = set()

    # Load the synthetic data pickle file
    with open(synthetic_data_path, 'rb') as f:
        synthetic_data_list = pickle.load(f)

    synthetic_samples = []

    for error in errors:
        incorrect_word, correct_word = error
        # Skip if the error has already been added
        if (incorrect_word, correct_word) in already_added:
            continue
        # Search for the corresponding synthetic data entry
        for entry in synthetic_data_list:
            if entry['before'] == incorrect_word and entry['after'] == correct_word:
                print(f'Found match for: {incorrect_word}, {correct_word}')
                # Found a matching entry
                sentences = entry['sentences']
                corrections = entry['corrections']
                if len(sentences) > 0:
                    print(sentences[0])
                    print(corrections[0])
                print(type(sentences), type(corrections), len(sentences), len(corrections))
                print(num_samples)

                # If num_samples is specified, select a limited number of samples
                if num_samples is not None:
                    sentences = sentences[:(num_samples // 2)]
                    corrections = corrections[:(num_samples // 2)]

                # Create 'label=1' samples (error-containing: sentence != correction)
                label_1_samples = pd.DataFrame({
                    'text': sentences,
                    'target': corrections,
                    'before': incorrect_word,
                    'after': correct_word,
                    'label': [1] * len(sentences)
                }).astype({'label': int})
                # Create 'label=0' samples (error-free: correction == correction)
                label_0_samples = pd.DataFrame({
                    'text': corrections,  # Use corrected sentences as both text and target
                    'target': corrections,  # Corrected sentences (no errors)
                    'before': correct_word,  # No error, so before == after == corrected sentence
                    'after': correct_word,
                    'label': [0] * len(corrections)
                }).astype({'label': int})
                synthetic_samples.append(pd.concat([label_1_samples, label_0_samples], ignore_index=True))
                already_added.add((incorrect_word, correct_word))  # Mark this error as added
                break  # Move on to the next error after finding a match

    # Combine all synthetic samples into a single DataFrame
    if synthetic_samples:
        synthetic_df = pd.concat(synthetic_samples, ignore_index=True)
        # Shuffle DataFrame
        synthetic_df = synthetic_df.sample(frac=1).reset_index(drop=True)
        return synthetic_df

    # Return an empty DataFrame if no synthetic data is found for the errors
    return pd.DataFrame()


def get_misclassified_samples(pred, dataset):
    """
    Identifies misclassified samples from predictions and actual labels.

    :param pred: Object containing model's predictions and true labels.
    :param dataset: Validation dataset (contains actual inputs and labels, including 'before' and 'after').
    :return: List of tuples containing the incorrect samples and their corresponding corrections (before, after).
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    misclassified_samples = []

    for idx, (label, pred_label) in enumerate(zip(labels, preds)):
        if label != pred_label and label == 1:
            incorrect = dataset['before'][idx]
            correct = dataset['after'][idx]
            misclassified_samples.append((incorrect, correct))

    return misclassified_samples


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
        num_labels=2,
        output_hidden_states=output_hidden_states
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


# Training loop with boosting
def training_loop_with_boosting(learning_rate, num_epochs, batch_size, warmup_ratio, warmup_steps, weight_decay,
                                train_set, val_set, test_set, tokenizer, output_path, n_runs, synthetic_data_path,
                                num_samples=None):
    """
    Trains the model iteratively, adding synthetic data from pickle file for errors after each iteration.

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
    :param n_runs: Number of boosting iterations.
    :param synthetic_data_path: Path to the synthetic data pickle file.
    :param num_samples: Maximum number of synthetic samples to load per error.
    """
    start_time = time.time()

    run, model_state_dict, optimizer_state_dict, trainer_state_dict, already_added, train_set_df = load_checkpoint(
        output_path)
    if run is None:
        run = 0
        already_added = set()
        model = model_init()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        model = model_init()
        model.load_state_dict(model_state_dict)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(optimizer_state_dict)  # Restore optimizer state
        train_set = tokenize_data(convert_to_hf(train_set_df), tokenizer)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{output_path}/{model_ckpt}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
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
        logging_dir=f'{output_path}/logs',
        logging_steps=500,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model='recall',
    )

    for current_run in range(run, n_runs):
        empty_cache()
        print(f"Run {current_run + 1}/{n_runs} with boosting")

        start_time_boosting_run = time.time()

        if not os.path.exists(f'{output_path}/boosting_run_{current_run}/logs'):
            os.makedirs(f'{output_path}/boosting_run_{current_run}/logs')

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
        # Train the model
        trainer.train()
        empty_cache()

        # save logs
        history = pd.DataFrame(trainer.state.log_history)
        history.to_csv(f'{output_path}/boosting_run_{current_run}/logs/log_history.csv')
        empty_cache()

        # Predict on validation set and get misclassified samples
        val_predictions = trainer.predict(val_set)
        misclassified_samples = get_misclassified_samples(val_predictions, val_set)

        end_time_boosting_run = time.time()
        elapsed_time_boosting_run = end_time_boosting_run - start_time_boosting_run

        # Save results
        with open(f'{output_path}/boosting_run_{current_run}/results.csv', 'w') as f:
            for key, value in val_predictions.metrics.items():
                f.write(f'{key},{value}\n')
            f.write('Parameters used:\n')
            f.write(f'learning_rate = {learning_rate}\n')
            f.write(f'num_epochs = {num_epochs}\n')
            f.write(f'batch_size = {batch_size}\n')
            f.write(f'weight_decay = {weight_decay}\n')
            f.write(f'warmup_steps = {warmup_steps}\n')
            f.write(f'warmup_ratio = {warmup_ratio}\n')
            f.write(f'number of samples in training set = {len(train_set)}\n')
            f.write(f'number of samples in validation set = {len(val_set)}\n')
            f.write(f'Runtime: {int(elapsed_time_boosting_run) // 3600} h, {int((elapsed_time_boosting_run % 3600) // 60)} min, {elapsed_time_boosting_run % 60} s')
        y_pred = np.argmax(val_predictions.predictions, axis=1)
        np.save(f'{output_path}/boosting_run_{current_run}/y_pred.npy', y_pred)
        build_confusion_matrix(val_predictions.predictions.argmax(-1), val_predictions.label_ids, f'{output_path}/boosting_run_{current_run}', iteration=current_run+1)
        with open(f'{output_path}/boosting_run_{current_run}/already_added.json', 'w') as f:
            json.dump(list(already_added), f)

        # Log the misclassified samples
        print(f"Run {current_run + 1} - Misclassified Samples (where the actual label is 1): {len(misclassified_samples)}")
        print(f'Misclassified Samples:\n{misclassified_samples}')

        # Break out of loop when we are at last iteration to not add more synthetic data
        if current_run == n_runs - 1:
            # Save the checkpoint after each run
            save_checkpoint(current_run + 1, model, trainer, optimizer, already_added, train_set, output_path)
            break

        # Load corresponding synthetic data from pickle for the errors
        synthetic_data = load_synthetic_data_from_pickle(misclassified_samples, synthetic_data_path, num_samples,
                                                         already_added)
        if not synthetic_data.empty:
            print(f"Adding {len(synthetic_data)} synthetic samples for the next iteration.")

            # Combine the original training data with the new synthetic data by concatenating DataFrames
            train_set_df = pd.concat([train_set.to_pandas(), synthetic_data],
                                     ignore_index=True)  # Concatenate pandas DataFrames
            # Shuffle the combined DataFrame
            train_set_df = train_set_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
            # Re-tokenize the combined dataset for HuggingFace Trainer
            train_set = tokenize_data(convert_to_hf(train_set_df), tokenizer)  # Re-tokenize the new augmented dataset
            print('New train dataset:')
            print(train_set)
            # Update the already_added set
            already_added.update([(row['before'], row['after']) for _, row in synthetic_data.iterrows()
                                  if row['before'] != row['after']])
        else:
            print("No new synthetic data to add for this iteration.")
            break

        # Save the checkpoint after each run
        save_checkpoint(current_run + 1, model, trainer, optimizer, already_added, train_set, output_path)

    # Final evaluation on the test set after all boosting iterations
    final_predictions = trainer.predict(test_set)
    accuracy = final_predictions.metrics['test_accuracy']
    print(f"Final test accuracy after boosting runs: {accuracy}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save final results
    np.save(f'{output_path}/final_predictions.npy', final_predictions.predictions.argmax(-1))
    with open(f'{output_path}/final_results.csv', 'w') as f:
        for key, value in final_predictions.metrics.items():
            f.write(f'{key},{value}\n')
        f.write('Parameters used:\n')
        f.write(f'learning_rate = {learning_rate}\n')
        f.write(f'num_epochs = {num_epochs}\n')
        f.write(f'batch_size = {batch_size}\n')
        f.write(f'weight_decay = {weight_decay}\n')
        f.write(f'warmup_steps = {warmup_steps}\n')
        f.write(f'warmup_ratio = {warmup_ratio}\n')
        f.write(f'n_runs = {n_runs}\n')
        f.write(f'num_samples = {num_samples}\n')
        f.write(f'number of samples in training set = {len(train_set)}\n')
        f.write(f'number of samples in validation set = {len(val_set)}\n')
        f.write(f'Runtime: {int(elapsed_time) // 3600} h, {int((elapsed_time % 3600) // 60)} min, {elapsed_time % 60} s')
    build_confusion_matrix(final_predictions.predictions.argmax(-1), final_predictions.label_ids, output_path)
    with open(f'{output_path}/already_added.json', 'w') as f:
        json.dump(list(already_added), f)


if __name__ == '__main__':
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
    n_runs = 5
    num_samples = 500

    # create output path if it does not exist yet
    output_path = f'{output_path}/n_runs={n_runs}_num_samples={num_samples}'
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

    # Run the boosting loop
    training_loop_with_boosting(
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
        n_runs=n_runs,
        synthetic_data_path=synthetic_data_path,
        num_samples=num_samples
    )

