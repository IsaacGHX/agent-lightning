# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
import fire
from verl.utils.hdfs_io import copy, makedirs
import argparse

prompt = "Using the numbers {numbers}, create an expression that equals 24. You must use basic arithmetic operations (+, -, \u00d7, /) and parentheses. Example: for [1, 2, 3, 4], one solution is (1+2+3)\u00d74."

def main(
    data_source='nlile/24-game',
    local_dir='~/data/gameof24',
    hdfs_dir=None,
):
    
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, split='train', trust_remote_code=True)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = prompt.format(numbers=example.pop('numbers'))
            solution = example.pop('solutions')
            
            data = {
                "id": idx,
                "question": question,
                "result": solution,
                "source": data_source,
                "extra_info": {
                    'ground_truth': solution,
                    'idx': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn('test', data_source), with_indices=True, remove_columns=test_dataset.column_names)
    
    print(train_dataset)
    print(train_dataset[0])
    

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/gameof24/data.py --data_source nlile/24-game --local_dir data/gameof24
"""