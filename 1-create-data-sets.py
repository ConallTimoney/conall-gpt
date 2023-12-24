from utils.data import create_train_valid_whatsapp, S3_DATA, S3_BUCKET
from utils.data import create_train_valid_whatsapp
from transformers import (
    AutoModelForSeq2SeqLM
    , DataCollatorForSeq2Seq
    , AutoTokenizer
    , Seq2SeqTrainer
    , Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from utils.data import S3_DATA

MODEL_ID = 'NousResearch/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

data = create_train_valid_whatsapp(
    path=S3_DATA
    , train_sender_name='Conall'
    , prompt_max_length=min(1000, tokenizer.model_max_length)
)


def tokenize_data(data_set, tokenizer):
    model_inputs = tokenizer(
        data_set['text']
        , truncation=True
        , padding=True
        , max_length=1000
    )
    
    labels = tokenizer(
        data_set['text']
        , truncation=True
        , padding=True
        , max_length=1000
    )
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs


tokenized_train = data['train'].map(
    lambda data: tokenize_data(data, tokenizer=tokenizer)
    , batched=False
    , remove_columns=['text', 'label']
)

tokenized_test = data['test_data'].map(
    lambda data: tokenize_data(data, tokenizer=tokenizer)
    , batched=False
    , remove_columns=['text', 'label']
)


tokenized_train.save_to_disk(
    f's3://{S3_BUCKET}/train'
)

tokenized_test.save_to_disk(
    f's3://{S3_BUCKET}/test'
)
