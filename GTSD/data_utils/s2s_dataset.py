import os
import torch
import datasets
import json
import random
import logging
import jsonlines

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler

# from data_utils.fairseq_dataset import load_fairseq

logger = logging.getLogger(__name__)


def load_jsonl_data(config, attri):
    data = []
    if True:
        path = os.path.join(config.data.path, config.data.name, attri + '.jsonl')

        with jsonlines.open(path, 'r') as rp:
            for line in rp:
                for i in range(0,config.batch_size):
                    data.append({
                        'src': line['concept_set'],
                        'tgt': line['scene'][0]
                    })
        rp.close()


    if 0 == 0:
        random_list = random.sample(range(len(data)), 10)
        for idx in random_list:
            logger.info(f"example of {idx} is : {data[idx]}")

    return data


def load_s2s_data(config, tokenizer):
    if 0 == 0:
        logger.info("***** load " + config.data.name + " train dataset *****")

    # if 'C4' in config.data.name:
    #     data = datasets.load_from_disk(
    #         os.path.join(config.data.path, config.data.name, 'train')
    #     )
    # else:
    train_data = load_jsonl_data(config, attri='train')

    if 0 == 0:
        logger.info("***** load " + config.data.name + " dev dataset *****")
    # if 'C4' in config.data.name:
    #     data = datasets.load_from_disk(
    #         os.path.join(config.data.path, config.data.name, 'dev')
    #     )
    # else:
    dev_data = load_jsonl_data(config, attri='dev')

    train_dataset = S2S_dataset(train_data, tokenizer, config)
    dev_dataset = S2S_dataset(dev_data, tokenizer, config)

    if 0 == 0:
        logger.info(f"example of TRAIN id lists: {train_dataset[50]}")
        logger.info(f"total query TRAIN dataset len : {len(train_dataset)}")
        logger.info(f"example of DEV id lists: {dev_dataset[50]}")
        logger.info(f"total query DEV dataset len : {len(dev_dataset)}")

    train_sample = train_dataset  # DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sample, pin_memory=True,
        batch_size=config.batch_size, drop_last=False, num_workers=config.num_workers,
        collate_fn=S2S_dataset.get_collate_fn(config)
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=config.batch_size,
        drop_last=False, pin_memory=True, num_workers=config.num_workers,
        collate_fn=S2S_dataset.get_collate_fn(config)
    )

    return train_dataloader, dev_dataloader


class S2S_dataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __getitem__(self, index):
        if isinstance(index, int):
            example = self.data[index]

            if self.config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok']:
                # and (not self.config.use_mbert):
                if self.config.use_bpe:
                    src_input_ids = torch.LongTensor(self.tokenizer.encode(example['src']).ids)
                    tgt_input_ids = torch.LongTensor(self.tokenizer.encode(example['tgt']).ids)
                else:
                    src_input_ids = example['src']
                    tgt_input_ids = example['tgt']

            else:
                # print(example)
                # Since the input(i.e. src) can be of indefinite length, padding within a batch increases efficiency.
                # But the output(i.e. tgt) length is deterministic, so padding to the output maximum length (statistics).
                src_input_ids = self.tokenizer.encode(example['src'],
                                                      padding='max_length',
                                                      truncation=True,
                                                      max_length=self.config.tgt_len,
                                                      return_tensors='pt')
                tgt_input_ids = self.tokenizer.encode(example['tgt'],
                                                      padding='max_length',
                                                      truncation=True,
                                                      max_length=self.config.tgt_len,
                                                      return_tensors='pt')

            return {'src': src_input_ids, 'tgt': tgt_input_ids, 'length': example['src']}
        return index

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_collate_fn(cls, config):
        def fn(batch):
            src, tgt, length = [], [], []
            for item in batch:
                src.append(item['src'].squeeze())
                tgt.append(item['tgt'].squeeze())

            if config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok']:
                # and (not config.use_mbert):
                src_tensor = pad_sequence(src, batch_first=True, padding_value=config.pad_value)
                src_tensor = src_tensor[:, :config.max_pos_len]
                tgt_tensor = pad_sequence(tgt, batch_first=True, padding_value=config.pad_value)
                tgt_tensor = tgt_tensor[:, :config.tgt_len]
                if not config.pred_len and config.tgt_len > tgt_tensor.size(1):
                    # padding to max target length
                    tgt_tensor = torch.cat((tgt_tensor, torch.tensor(config.pad_value).repeat(
                        tgt_tensor.size(0), (config.tgt_len - tgt_tensor.size(1)))), dim=-1)

            else:
                src_tensor = pad_sequence(src, batch_first=True, padding_value=config.pad_value)
                tgt_tensor = torch.stack(tgt)

            if config.pred_len:
                length_tensor = torch.tensor(length).long()
                tgt_attention_mask = (tgt_tensor != config.pad_value).long()
            else:
                tgt_attention_mask = torch.ones_like(tgt_tensor).long()
                length_tensor = torch.tensor(config.tgt_len).repeat(tgt_tensor.size(0)).long()

            if config.prediction:
                return {"src_input_ids": src_tensor.long(),
                        "src_attention_mask": (src_tensor != config.pad_value).long(),
                        "tgt_input_ids": tgt_tensor,
                        "tgt_attention_mask": tgt_attention_mask,
                        "length": length_tensor, }
            else:
                return {"src_input_ids": src_tensor.long(),
                        "src_attention_mask": (src_tensor != config.pad_value).long(),
                        "tgt_input_ids": tgt_tensor,
                        "tgt_attention_mask": tgt_attention_mask,
                        "length": length_tensor, }

        return fn