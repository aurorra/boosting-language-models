import random
import os
import time
import argparse

import pandas as pd
import numpy as np
import torch
import gc
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, set_seed, \
    DataCollatorWithPadding, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score,
                             average_precision_score)
import seaborn as sns
from scipy.special import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Run inference for error detection models.')

parser.add_argument('model_ckpt', type=str, help='Base model checkpoint')
parser.add_argument('test_file', type=str, help='Path to the test CSV file')
parser.add_argument('checkpoint_path', type=str, help='Path to the trained model/checkpoint')
parser.add_argument('output_path', type=str, help='Path for inference results')
parser.add_argument('--strategy', choices=['boosting', 'baseline', 'ablation', 'targeted'], required=True,
                    help='Training strategy used for the model')

args = parser.parse_args()

model_ckpt = args.model_ckpt
test_file = args.test_file
checkpoint_path = args.checkpoint_path
output_path = args.output_path
strategy = args.strategy

model_from_boosting = strategy == 'boosting'


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
    # Actual labels
    labels = pred.label_ids
    # Predictions
    preds = pred.predictions.argmax(-1)
    # Positive-class probability
    probs = softmax(pred.predictions, axis=1)[:, 1]

    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)

    roc_auc = roc_auc_score(labels, probs)
    average_precision = average_precision_score(labels, probs)

    return {'accuracy': acc, 'f1': f1, 'recall': recall, 'precision': precision, 'ROC-AUC': roc_auc,
            'average precision': average_precision}


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


def model_init(path=None, quantization_config=None, output_hidden_states=False):
    """
    Initializes and returns a pre-trained model for sequence classification tasks.

    This function creates an instance of a pre-trained model from the Hugging Face library.
    The model is loaded based on a specified model checkpoint.

    :param quantization_config:
    :param path: path to load model from disk
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
    if quantization_config:
        m = AutoModelForSequenceClassification.from_pretrained(
            path if path else model_ckpt,
            num_labels=2,
            quantization_config=quantization_config,
            output_hidden_states=output_hidden_states
        )
    else:
        m = AutoModelForSequenceClassification.from_pretrained(
            path if path else model_ckpt,
            num_labels=2,
            output_hidden_states=output_hidden_states
        ).to(device)
    return m


def inference(data_test, path_output, path_checkpoint, learning_rate, num_epochs, batch_size, warmup_ratio, warmup_steps, weight_decay):
    start_time = time.time()
    if model_from_boosting:
        run, model_state_dict, optimizer_state_dict, trainer_state_dict, already_added, train_set_df = load_checkpoint(
            path_checkpoint)
        # Load checkpoint
        if 'Llama3' in path_checkpoint or 'mistral' in path_checkpoint:
            # quantization configuration for model initialization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # enable 4-bit quantization
                bnb_4bit_quant_type='nf4',  # information theoretically optimal dtype for normally distributed weights
                bnb_4bit_use_double_quant=True,  # quantize quantized weights
                bnb_4bit_compute_dtype=torch.bfloat16  # optimized fp format for ML
            )
            # lora config
            lora_config = LoraConfig(
                r=r,  # the dimension of the low-rank matrices
                lora_alpha=lora_alpha,  # scaling factor for LoRA activations vs pre-trained weight activations
                target_modules='all-linear',  # to get comparable results to full finetuning train all Linear layers
                lora_dropout=lora_dropout,  # dropout probability of the LoRA layers
                use_rslora=True,
                bias='none',  # whether to train bias weights, set to 'none' for attention layers
                task_type='SEQ_CLS'
            )
            model = model_init(quantization_config=quantization_config)
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            model.config.pad_token_id = tokenizer.pad_token_id

            model.load_state_dict(model_state_dict)
        else:
            model = model_init()
            model.load_state_dict(model_state_dict)
    else:
        if 'Llama3' in path_checkpoint or 'mistral' in path_checkpoint:
            # quantization configuration for model initialization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # enable 4-bit quantization
                bnb_4bit_quant_type='nf4',  # information theoretically optimal dtype for normally distributed weights
                bnb_4bit_use_double_quant=True,  # quantize quantized weights
                bnb_4bit_compute_dtype=torch.bfloat16  # optimized fp format for ML
            )
            # lora config
            lora_config = LoraConfig(
                r=r,  # the dimension of the low-rank matrices
                lora_alpha=lora_alpha,  # scaling factor for LoRA activations vs pre-trained weight activations
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # adjust these names based on your model
                lora_dropout=lora_dropout,  # dropout probability of the LoRA layers
                use_rslora=True,
                bias='none',  # whether to train bias weights, set to 'none' for attention layers
                task_type='SEQ_CLS'
            )
            model = model_init(path=path_checkpoint, quantization_config=quantization_config)
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            model = model_init(path_checkpoint)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{path_output}/{model_ckpt}',
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        disable_tqdm=False,
        log_level='info',
        logging_dir=f'{path_output}/logs',
        logging_steps=500,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model='recall',
    )

    trainer = Trainer(
        model=model,  # Your model instance
        args=training_args,  # Training arguments
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Final evaluation on the test set after all boosting iterations
    final_predictions = trainer.predict(data_test)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save final results
    # hard labels
    np.save(f'{path_output}/final_predictions.npy', final_predictions.predictions.argmax(-1))
    # probalities
    np.save(f'{path_output}/final_predictions_probs.npy', softmax(final_predictions.predictions, axis=1))
    with open(f'{path_output}/final_results.csv', 'w') as f:
        for key, value in final_predictions.metrics.items():
            f.write(f'{key},{value}\n')
        f.write('Parameters used:\n')
        f.write(f'learning_rate = {learning_rate}\n')
        f.write(f'number of samples in test set = {len(data_test)}\n')
        f.write(
            f'Runtime: {int(elapsed_time) // 3600} h, {int((elapsed_time % 3600) // 60)} min, {elapsed_time % 60} s')
    build_confusion_matrix(final_predictions.predictions.argmax(-1), final_predictions.label_ids, path_output)
    print(f'Saved results into {path_output}')


if __name__ == '__main__':
    # empty cache and set seed (reproducibility)
    empty_cache()
    set_seed_locally(62)
    set_seed(62)

    # create output path if it does not exist yet
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load data and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

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

        tokenizer.pad_token = tokenizer.eos_token
    elif 'Llama3' in model_ckpt:
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

        tokenizer.pad_token = tokenizer.eos_token
    else:
        learning_rate = 11e-06
        batch_size = 64
        num_epochs = 20
        weight_decay = 0
        warmup_steps = 0
        warmup_ratio = 0
        n_runs = 5
        num_samples = 500
        gradient_accumulation_steps = 4

    df = pd.read_csv(test_file, keep_default_na=False)
    print(f'Read file from {test_file}')
    if 'prompt_input' in df.columns:
        df.rename(columns={'prompt_input': 'text', 'correction': 'target'}, inplace=True)
        print('Renamed columns:', df.columns)
    else:
        df.rename(columns={'sentence': 'text', 'correction': 'target'}, inplace=True)
        print('Renamed columns:', df.columns)

    # Convert datasets
    test_set = tokenize_data(convert_to_hf(df), tokenizer)
    empty_cache()

    # Run inference
    inference(
        test_set,
        output_path,
        checkpoint_path,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay
    )



