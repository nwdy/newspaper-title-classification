import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer, 
    TrainingArguments,
    AutoConfig
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import huggingface_hub


class LoadDataset:
    def __init__(self, train_file, valid_file):
        self.train_file = train_file
        self.valid_file = valid_file
    
    def load_data(self):
        df_train = pd.read_csv(self.train_file)
        df_val = pd.read_csv(self.valid_file)
        
        self.numerize(df_train)
        self.numerize(df_val)
        
        train_dataset = Dataset.from_pandas(df_train)
        valid_dataset = Dataset.from_pandas(df_val)
        
        return train_dataset, valid_dataset
    
    def numerize(self, df):
        df.loc[df['genre'] == "kinh-te", 'label'] = 0
        df.loc[df['genre'] == "giao-duc", "label"] = 1
        df.loc[df['genre'] == "xe", "label"] = 2
        df.loc[df['genre'] == "suc-khoe", "label"] = 3
        df.loc[df['genre'] == "cong-nghe-game", "label"] = 4


class ModelTrainer:
    def __init__(self, model_name, train_dataset, valid_dataset):
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
        self.config = self.configure_model()
        self.model = self.load_model()

    def configure_model(self):
        labels = list(set(self.train_dataset["label"]))
        id2label = {k:v for k,v in enumerate(labels)}
        label2id = {v:k for k,v in enumerate(labels)}
        num_labels = len(labels)
        
        config = AutoConfig.from_pretrained(self.model_name, num_labels=num_labels,
                                            label2id=label2id, id2label=id2label)
        
        return config
    
    def load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        return model
    
    def tokenize(self, examples):
        tokenized_inputs = self.tokenizer(examples['title'], padding=True, truncation=True)
        return tokenized_inputs
    
    def preprocess_dataset(self, dataset):
        return dataset.map(self.tokenize, batched=True, batch_size=None)
    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}
    
    def train_model(self, training_args):
        train_dataset = self.preprocess_dataset(self.train_dataset)
        valid_dataset = self.preprocess_dataset(self.valid_dataset)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        trainer.train()

        return trainer

def main():
    # Usage
    train_file = 'data/train_dataset.csv'
    valid_file = 'data/valid_dataset.csv'
    
    model_name = "vinai/phobert-base"

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_steps=62,
        evaluation_strategy='epoch',
        eval_steps=10,
        report_to='none'
    )

    data_loader = LoadDataset(train_file, valid_file)
    train_dataset, valid_dataset = data_loader.load_data()

    trainer = ModelTrainer(model_name, train_dataset, valid_dataset)
    trainer.train_model(training_args)

    # # Pushing to the Hub
    # import os
    # os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    # repo_id = f'{huggingface_hub.whoami()["name"]}/{"phoBERT_finetune_news_classification"}'
    # trainer.model.push_to_hub(repo_id)
    # trainer.tokenizer.push_to_hub(repo_id)

if __name__ == '__main__':
    main()