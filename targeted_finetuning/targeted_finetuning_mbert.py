"""
Load boosting model and train on additional training data. After that test on real-world data.
"""

import random
import os
import time
import argparse

import pandas as pd
from safetensors.torch import load_file
import numpy as np
import torch
import gc
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, set_seed, \
    DataCollatorWithPadding
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def load_checkpoint(output_path):
    safepath = os.path.join(output_path, "model.safetensors")
    if os.path.exists(safepath):
        model_state_dict = load_file(safepath)
        # if you have no optimizer/trainer saved alongside, set them to None or {}
        optimizer_state_dict = None
        trainer_state_dict = None
        already_added = set()
        train_set_df = None
        return None, model_state_dict, optimizer_state_dict, trainer_state_dict, already_added, train_set_df

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


def save_checkpoint(run, model, trainer, optimizer, output_path):
    checkpoint_path = f"{output_path}/checkpoint_run_{run}.pt"
    checkpoint = {
        'run': run,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'trainer_state_dict': trainer.state,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Saved checkpoint to {checkpoint_path}')


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
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    return {"accuracy": acc, "f1": f1, 'recall': recall, 'precision': precision}


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


def freeze_layers(model, num_layers_to_freeze):
    """
    Freeze the first `num_layers_to_freeze` transformer layers.

    :param model: The model to modify.
    :param num_layers_to_freeze: The number of transformer layers to freeze.
    """
    for name, param in model.named_parameters():
        # Freeze encoder layers
        if 'encoder.layer' in name:
            layer_idx = int(name.split('.')[3])  # Correct position of the layer index
            if layer_idx < num_layers_to_freeze:
                param.requires_grad = False
        elif 'embeddings' in name:
            # Optionally, freeze embeddings as well
            param.requires_grad = False

    return model


def model_init(output_hidden_states=False, layers=0):
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

    # Freeze the first 6 layers
    m = freeze_layers(m, num_layers_to_freeze=layers)
    print(f'Froze first {layers} layers.')
    return m


def targeted_finetuning(data_train, data_val, data_test, path_output, path_checkpoint, learning_rate, num_epochs, batch_size, warmup_ratio, warmup_steps, weight_decay, layers):
    start_time = time.time()
    # Load checkpoint
    run, model_state_dict, optimizer_state_dict, trainer_state_dict, already_added, train_set_df = load_checkpoint(
        path_checkpoint)
    model = model_init(layers=layers)
    model.load_state_dict(model_state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        print("No optimizer state found; starting from fresh optimizer.")

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

    trainer = Trainer(
        model=model,  # Your model instance
        args=training_args,  # Training arguments
        compute_metrics=compute_metrics,
        train_dataset=data_train,
        eval_dataset=data_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None)    # Provide a tuple with optimizer and None as the scheduler
    )

    # Train the model
    trainer.train()
    empty_cache()

    # Final evaluation on the test set after all boosting iterations
    final_predictions = trainer.predict(data_test)

    # Save finetuned model
    trainer.model.save_pretrained(f'{path_output}/{model_ckpt}-final')
    save_checkpoint(run, model, trainer, optimizer, path_output)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save final results
    np.save(f'{path_output}/final_predictions.npy', final_predictions.predictions.argmax(-1))
    with open(f'{path_output}/final_results.csv', 'w') as f:
        for key, value in final_predictions.metrics.items():
            f.write(f'{key},{value}\n')
        f.write('Parameters used:\n')
        f.write(f'learning_rate = {learning_rate}\n')
        f.write(f'num_epochs = {num_epochs}\n')
        f.write(f'batch_size = {batch_size}\n')
        f.write(f'weight_decay = {weight_decay}\n')
        f.write(f'warmup_steps = {warmup_steps}\n')
        f.write(f'warmup_ratio = {warmup_ratio}\n')
        f.write(f'number of samples in training set = {len(data_train)}\n')
        f.write(f'number of samples in validation set = {len(data_val)}\n')
        f.write(f'number of samples in test set = {len(data_test)}\n')
        f.write(
            f'Runtime: {int(elapsed_time) // 3600} h, {int((elapsed_time % 3600) // 60)} min, {elapsed_time % 60} s')
    build_confusion_matrix(final_predictions.predictions.argmax(-1), final_predictions.label_ids, path_output)
    print(f'Saved results in folder {path_output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Targeted fine-tuning of an mBERT error detection model.'
    )
    parser.add_argument(
        'model_ckpt',
        type=str,
        help='Base model checkpoint, e.g. bert-base-multilingual-cased'
    )
    parser.add_argument(
        'checkpoint_path',
        type=str,
        help='Path to the trained model/checkpoint used as starting point'
    )
    parser.add_argument(
        'train_file',
        type=str,
        help='CSV file containing the additional targeted training data'
    )
    parser.add_argument(
        'test_file',
        type=str,
        help='CSV file containing the evaluation data'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path to save the fine-tuned model and results'
    )
    parser.add_argument(
        '--layers',
        type=int,
        default=6,
        help='Number of transformer layers to freeze'
    )

    args = parser.parse_args()

    model_ckpt = args.model_ckpt
    checkpoint_path = args.checkpoint_path
    train_file = args.train_file
    test_file = args.test_file
    output_path = args.output_path
    layers = args.layers

    # empty cache and set seed (reproducibility)
    empty_cache()
    set_seed_locally(62)
    set_seed(62)

    # hyperparameters
    learning_rate = 11e-06
    batch_size = 64
    num_epochs = 20
    weight_decay = 0
    warmup_steps = 0
    warmup_ratio = 0

    # create output path if it does not exist yet
    os.makedirs(output_path, exist_ok=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # load targeted fine-tuning data
    df_train = pd.read_csv(
        train_file,
        keep_default_na=False
    ).rename(
        columns={'sentence': 'text', 'correction': 'target'}
    )

    # load evaluation data
    df_test = pd.read_csv(
        test_file,
        keep_default_na=False
    ).rename(
        columns={'sentence': 'text', 'correction': 'target'}
    )

    print(f'Training with {len(df_train)} additional samples.')

    # split targeted data into training and validation sets
    train_df, val_df = train_test_split(
        df_train,
        test_size=0.2,
        random_state=62
    )

    # convert datasets
    train_set = tokenize_data(convert_to_hf(train_df), tokenizer)
    val_set = tokenize_data(convert_to_hf(val_df), tokenizer)
    test_set = tokenize_data(convert_to_hf(df_test), tokenizer)

    # run targeted fine-tuning
    targeted_finetuning(
        train_set,
        val_set,
        test_set,
        output_path,
        checkpoint_path,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        layers=layers
    )
