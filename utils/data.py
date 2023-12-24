import numpy as np
import polars as pl 
from polars import col 
import polars.selectors as cs 
import regex as re
from IPython.display import display
from datasets import Dataset, DatasetDict




WHATSAPP_MESSAGE_PREFIX_REGEX = r'^[0-9]{2}\/[0-9]{2}\/[0-9]{4}\, [0-9]{1,2}:[0-9]{2}\s?(am|pm)\s?-\s?'
WHATSAPP_MESSAGE_PREFIX_AND_NAME_REGEX = f'{WHATSAPP_MESSAGE_PREFIX_REGEX}[A-Za-z]+:'


def is_new_whatsapp_message(message: str):
    return re.match(WHATSAPP_MESSAGE_PREFIX_AND_NAME_REGEX, message)


def read_whats_app_messages(path: str) -> pl.Series:
    messages = []
    with open(path, 'r') as data:
        for line in data:
            first_message = 'Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them. Tap to learn more.' in line
            if first_message: 
                continue
            
            if is_new_whatsapp_message(line):
                messages.append(line)
            else: 
                messages[-1] += line
                
    messages = pl.Series(
        'message'
        , messages
    )
    
    return messages



def messages_to_seq2seq(
    messages: pl.Series
    , train_sender_name: str
    , prompt_max_length: int
) -> pl.DataFrame:
    return (
        pl.DataFrame(messages)
        .with_columns(
            train_message = 
                col('message')
                .str.contains(f'{WHATSAPP_MESSAGE_PREFIX_REGEX}{train_sender_name}:')
        )
        .with_columns(
            last_message_in_train_chain = 
                col('train_message') & col('train_message').shift(-1).not_()
            , first_message_in_target = 
                col('train_message') & col('train_message').shift(1).not_()
        )
        .with_columns(
            message_group_index = 
                pl.when(col('last_message_in_train_chain'))
                .then(col('last_message_in_train_chain').rank('ordinal'))
                .otherwise(None)
                .pipe(
                    lambda series:
                        series - series.min()
                )
                .fill_null(strategy='backward')
        )
        .lazy()
        .with_columns(
            text = 
                pl.concat_str([
                    col('message').shift(i, fill_value='') 
                    for i in range(prompt_max_length, 0, -1)
                ])
        )
        .group_by('message_group_index')
        .agg(
            col('text')
                .filter(col('first_message_in_target'))
                .last()
            , label=
                col('message')
                .filter(col('train_message'))
                .str.concat(delimiter='')
        )
        .sort('message_group_index')
        .drop('message_group_index')
        .filter(col('label') != "")
        .collect()
    )
    


def polars_whatsapp_seq2seq(
    path
    , train_sender_name
    , prompt_max_length
):
    messages = read_whats_app_messages(path)        
    data = messages_to_seq2seq(messages, train_sender_name, prompt_max_length)
    return data 


def hf_whatsapp_seq2seq(
    data
): 
    return Dataset(
        data
        .to_arrow()
    )
    
def create_train_valid_whatsapp(
    path
    , train_sender_name
    , prompt_max_length
    , train_fraction=0.8
):
    all_data = polars_whatsapp_seq2seq(
        path
        , train_sender_name
        , prompt_max_length
    )
    
    last_train_index = int(train_fraction * all_data.shape[0])
    train_data = all_data[:last_train_index, :]
    test_data = all_data[last_train_index:, :]
    
    return DatasetDict(
        train = hf_whatsapp_seq2seq(train_data)
        , test_data = hf_whatsapp_seq2seq(test_data)
    )