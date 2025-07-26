

import os
import hydra
import torch
import logging
import random
import numpy as np
import time
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, set_seed
from torch.utils.data import DataLoader
from train_utils.resample import create_named_schedule_sampler

from utils import load_states_from_checkpoint
from data_utils.s2s_dataset import load_jsonl_data, S2S_dataset
from data_utils.tokenizer_utils import create_tokenizer
from model_utils.create_model import create_model, create_gaussian_diffusion

import sys
sys.path.append(r'F:\code\yinxie\AR_DATA\new\prompt')
print(sys.path)




os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def denoised_fn_round(config, emb_model, text_emb, t):
    down_proj_emb = emb_model.weight  # (vocab_size, embed_dim)

    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            emb_norm = (down_proj_emb ** 2).sum(-1).view(-1, 1)  # (vocab, 1)
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # (emb_dim, bs*seqlen)
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # (bs*seqlen, 1)
            # down_proj_emb: (vocab, emb_dim), text_emb_t:(emb_dim, bs*seqlen)
            # a+b automatically broadcasts to the same dimension i.e. (vocab, bs*seqlen)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb, text_emb_t)
            dist = torch.clamp(dist, 0.0, np.inf)  # Limit the value of input to [min, max].
        # Select the smallest distance in the vocab dimension,
        # that is, select bs*seq_len most likely words from all vocabs.
        topk_out = torch.topk(-dist, k=1, dim=0)

        return topk_out.values, topk_out.indices  # logits, token_id (1, bs*seq_len)

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb

    val, indices = get_efficient_knn(down_proj_emb,
                                     text_emb.to(down_proj_emb.device), dist=dist)
    rounded_tokens = indices[0]  # (bs*seq_len,)
    new_embeds = emb_model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds


def split_data(data, log=False):
    data_piece = data

    # if log:
    # logger.info(f'generation for {len(data_piece)} text from idx {start_idx} to {end_idx}')

    return data_piece

def read_bit_txt(path):
    with open(path, "r") as f:
        numbers = [int(line.strip()) for line in f if line.strip()]
    return numbers

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    local_rank = 0  # int(os.environ["LOCAL_RANK"])
    config.exp.dir = os.path.join(config.exp.root, config.data.name, config.exp.name)
    generate_path = os.path.join(config.exp.dir, str(config.load_step))
    if config.load_from_ema:
        generate_path += ('_ema_' + str(config.ema_rate))
    if config.clip_denoised:
        generate_path += '_clip_denoised_'
    if config.infer_self_condition:
        generate_path += '_selfcond_'
    if config.skip_sample:
        generate_path += '_skip_'
    if config.ddim_sample:
        generate_path += '_ddim_'

    if config.schedule_sampler == 'xy_uniform':
        generate_path += ('_xy_' + str(config.gen_timesteps))
    else:
        generate_path += ('_un_' + str(config.skip_timestep))

    if (local_rank == 0) and (not os.path.exists(generate_path)):
        os.makedirs(generate_path)

    torch.cuda.set_device(local_rank)  # ddp setting
    # dist.init_process_group(backend="nccl")  # ddp setting

    set_seed(config.exp.seed + int(0))  # seed setting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    if config.data.name in ['iwslt14', 'iwslt14_tok']:
        tokenizer = None
        if config.use_bpe:
            tokenizer = create_tokenizer(path=f'./data/{config.data.name}/')
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    if tokenizer == None:
        vocab_size = config.vocab_size
    else:
        vocab_size = tokenizer.vocab_size
        if config.data.name in ['iwslt14', 'iwslt14_tok']:
            if config.use_bpe:
                config.pad_value = tokenizer.get_vocab()['<pad>']
            # else use by fairseq
        else:
            config.pad_value = tokenizer.pad_token_id

    # define model and diffusion
    model, diffusion = create_model(config, vocab_size), create_gaussian_diffusion(config)

    schedule_sampler = create_named_schedule_sampler(config, diffusion)

    # load trained model
    if config.load_from_ema:
        eval_model_path = os.path.join(
            config.exp.dir, 'model', f'ema_{config.ema_rate}_checkpoint-{config.load_step}')
    else:
        eval_model_path = os.path.join(
            config.exp.dir, 'model', f'model_checkpoint-{config.load_step}')
    print(eval_model_path)
    model_saved_state = load_states_from_checkpoint(eval_model_path, 0)
    model.load_state_dict(model_saved_state.model_dict,False)
    model.to(device)
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    if 0 == 0:
        logger.info(f'the parameter count is {pytorch_total_params}')

    # if 11 > 1:
    #   model = DDP(
    #      model, device_ids=[0], output_device=0, find_unused_parameters=False,
    # )

    if 0 == 0:
        print("sampling text from random noise...")
        logger.info(f"sample num is : {config.num_samples}")


    sample_fn = (diffusion.p_sample_loop)

    emb_model = model.word_embedding
    sum_time = 0


    if config.model.mode == 's2s':
        if 0 == 0:
            print(f"start generate query from dev dataset, for every passage,\
                    we generate {config.num_samples} querys...")
            logger.info("***** load " + config.data.name + " dev dataset*****")

        dev_data = load_jsonl_data(config, 'output'+str(config.numm))
        bits = read_bit_txt(os.path.join(config.data.path, config.data.name, 'output'+str(config.numm) + '.txt'))
        start_time = time.time()
        #dev_data=get_prompts()
        data_piece = split_data(dev_data, log=True)

        dev_dataset = S2S_dataset(data_piece, tokenizer, config)
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=config.batch_size,
            drop_last=False, pin_memory=True, num_workers=config.num_workers,
            collate_fn=S2S_dataset.get_collate_fn(config)
        )

        if 0 == 0:
            logger.info(f"total query DEV dataset len : {len(dev_dataset)}")

            random_list = random.sample(range(len(dev_dataset)), 10)
            for idx in random_list:
                logger.info(f"example of {idx} is : {dev_dataset[idx]}")

        torch.manual_seed(config.seed)
        ran=torch.randint(0, 12659, (1,config.num_samples))
        for i in range(int(config.numm),int(config.numm)+1):

            torch.cuda.empty_cache()
            dev_dataloader = tqdm(
                dev_dataloader
            )
            flagg=True
            index=0
            for _, batch in enumerate(tqdm(dev_dataloader)):
                each_sample_list=[]
                if batch['src_input_ids'].shape[0] != config.batch_size:
                    break

                with torch.no_grad():
                    encoder_hidden_states = model.encoder(input_ids=batch['src_input_ids'].cuda(),
                                                          attention_mask=batch['src_attention_mask'].cuda(),
                                                          ).last_hidden_state  # [bs, seq_len, hz]


                    pred_lengs, tgt_attention_mask = None, None
                    input_shape = (
                        batch['src_input_ids'].shape[0], config.tgt_len, config.in_channels,
                    )
                model_kwargs = {'src_attention_mask': batch['src_attention_mask'].to(device),
                                'tgt_attention_mask': tgt_attention_mask,
                                'encoder_hidden_states': encoder_hidden_states, }

                p=batch['src_input_ids'].cuda()
                condition=model.get_embeds(p)
                if int(config.numm)==10:
                    torch.manual_seed(ran[0][int(config.numm)])
                noise=torch.randn_like(torch.randn(config.batch_size, 54,64)).cuda()

                sample = sample_fn(
                    model,
                    input_shape,
                    noise=noise,
                    condition=condition,
                    device=device,
                    clip_denoised=config.clip_denoised,
                    # "Freeze" some parameters for easy recall.
                    denoised_fn=partial(denoised_fn_round,
                                        config, emb_model.cuda()),
                    progress=True,
                    model_kwargs=model_kwargs,
                    pred_lengs=pred_lengs,
                    top_p=0.92,
                    schedule_sampler=schedule_sampler,
                )

                logits = model.get_logits(sample)  # (bs, seq_len, vocab_size)

                sample_id_tensor = torch.argmax(logits, dim=-1)

                if config.data.name in ['wmt14', 'wmt14_hug', 'iwslt14', 'iwslt14_tok'] and (not config.use_mbert):
                    if config.use_bpe:
                        for sample_id in sample_id_tensor:
                            text = tokenizer.decode(sample_id.tolist(), skip_special_tokens=True)
                            for token in ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]:
                                text = text.replace(token, '')
                            each_sample_list.append(text)
                    else:
                        each_sample_list.extend(sample_id_tensor.tolist())
                else:
                    each_sample_list.extend(
                        tokenizer.batch_decode(sample_id_tensor, skip_special_tokens=True))

                output_path = os.path.join(generate_path, 'num' + str(i))
                if 0 == 0:
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                out_path = os.path.join(
                    output_path, "rank" + str(i) + "2_seed_"+str(config.batch_size)+"_" + str(config.exp.seed) + "_y.txt")
                if flagg==True:
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(str(each_sample_list[bits[index]]) + '\n')
                        index += 1
                    f.close()
                    flagg=False
                else:
                    with open(out_path, 'a', encoding='utf-8') as f:
                        f.write(str(each_sample_list[bits[index]]) + '\n')
                        index+=1
                    f.close()
                end_time = time.time()
                sum_time+=abs(start_time-end_time)
            print(sum_time/512)
            print("------------------------------------------------------------")


    else:
        return NotImplementedError


if __name__ == "__main__":
    main()
