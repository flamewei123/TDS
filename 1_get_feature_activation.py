import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from collections import Counter
import torch.nn.functional as F
from sae import Sae
import warnings
from utils import get_filenames_in_directory, read_json_data, find_tgt_pos_act
warnings.filterwarnings("ignore")

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_context_act(idx, sent, args, model, tokenizer, sae, device, prefix_length, tgt_len):
    gap=90
    prefix_length = prefix_length- gap
    tgt_len = tgt_len + gap
    tgt_layer = int(args.tgt_layer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_act = []
    tokenized_prefix = tokenizer(sent, return_tensors="pt")
    ori_ids = tokenized_prefix["input_ids"].to(device)
    required_length = prefix_length + tgt_len

    if ori_ids.size(1) < required_length:
        tokenized_prefix = tokenizer(
            sent, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=required_length, 
            truncation=True
        )
        ori_ids = tokenized_prefix["input_ids"].to(device)

    # Truncate sent_ids to the desired length
    sent_ids = ori_ids[:, :prefix_length + tgt_len]

    for i in range(tgt_len):
        ### get tgt-layer hidden states
        single_act = [] 
        prefix_ids = sent_ids[:,:-tgt_len+i] # get predfix
        golden_token_ids = prefix_ids[:,-1].unsqueeze(0)
        # 解码 golden_token_ids 为可视化的 token
        golden_token = tokenizer.decode(golden_token_ids[0], skip_special_tokens=True)

        answer_index = -golden_token_ids.shape[1]-1

        ## init hook
        hidden_states_dict = dict()
        
        def w_forward_hook_fn(module, inp, outp):
            hidden_states_dict['output'] = outp[0]
        if args.model_name == 'llama3-8b':
            hook = model.model.layers[tgt_layer].register_forward_hook(w_forward_hook_fn)
        if args.model_name == 'llama3-1b':
            hook = model.model.layers[tgt_layer].register_forward_hook(w_forward_hook_fn)
        if args.model_name == 'pythia-160m':
            hook = model.gpt_neox.layers[tgt_layer].register_forward_hook(w_forward_hook_fn)
        if args.model_name == 'pythia-70m':
            hook = model.gpt_neox.layers[tgt_layer].register_forward_hook(w_forward_hook_fn)
        # hook = model.model.register_forward_hook(w_forward_hook_fn)

        ## get outputs & probs
        outputs = model(input_ids=prefix_ids)
        # tgt_probs = F.softmax(outputs.logits, dim=-1)
        # outputs.logits.shape([1, seq_len, vocab])
        # answer_tokens_probs = outputs.logits[:, answer_index:-1:, :] #[1, answer_len, vocab]
        # answer_tokens_probs = torch.gather(answer_tokens_probs, -1, golden_token_ids.unsqueeze(-1)) # [1, answer_len, vocab] [1, answer_len, 1] -->[1, answer_len, 1]
        hidden_states = hidden_states_dict['output']
        # hidden_states = hidden_states[:, answer_index, :]
        
        # print(hidden_states.requires_grad)
        # hidden_states.requires_grad_(True)
        hook.remove()

        # ## get grad
        # all_grads_of_tokens = None
        # feature_attr_token = None
        # for j in range(golden_token_ids.shape[1]):
        #     # print("第{}个answer token".format(j))
        #     token_j_prob = answer_tokens_probs[:, j, :] #[1, 1]
        #     token_j_prob = torch.unbind(token_j_prob)
        #     token_j_grad = torch.autograd.grad(token_j_prob, hidden_states, retain_graph=True)
        #     grad = token_j_grad[0][:, answer_index, :] #[1, d]
        #     all_grads_of_tokens = grad if all_grads_of_tokens is None else torch.cat((all_grads_of_tokens, grad), dim=0) #[answer_len, d]
        #     # print(all_grads_of_tokens)
        # # all_attrs.append(all_grads_of_tokens.tolist()) #[layer_num, answer_len, d]

        # feature_matrix = sae.W_dec.to(device) #[131072, 4096]
        inter = sae.encode(hidden_states[:, answer_index, :].cpu())
        acts = inter.top_acts.to(device) #[1, 192]
    
        indices = inter.top_indices.to(device) #[1, 192]
        indices_list = indices.squeeze().tolist()
        for id, pos in enumerate(indices_list):
            pos_act = torch.tensor([[pos, acts[:, id].item()]]).to(device) #[1,2]
            single_act.append(pos_act.tolist()[0])

        all_act.append(single_act)
            
    return all_act


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--model_path", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Path to local pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="Model name as output prefix to indentify each running of experiment.")
    parser.add_argument("--tgt_layer",
                        type=str,
                        default='20',
                        help="Target layer index to extract SAE feature.")
    # Other parameters
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpu_id",
                        type=str,
                        default='0',
                        help="available gpu id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5)

    args = parser.parse_args()


    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpu_id) == 1:
        device = torch.device("cuda:%s" % args.gpu_id)
        n_gpu = 1
    else:
        # TODO: To implement multi gpus
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))


    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    

    ## load SAE
    if args.model_name == 'llama3-8b':
        model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=device)
        model.eval()

        sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.{}".format(args.tgt_layer))
        logger.info("Start extracting {}-th layer`s feature from Llama-3-8B".format(args.tgt_layer))

    if args.model_name == 'llama3-1b': # unsloth/Llama-3.2-1B
        model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=device)
        model.eval()
        # sae, cfg_dict, sparsity = SAE.from_pretrained("jbloom/Gemma-2b-Residual-Stream-SAEs", "gemma_2b_blocks.{}.hook_resid_post_16384".format(args.tgt_layer))
        sae = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint="layers.{}.mlp".format(args.tgt_layer))       
        logger.info("Start extracting {}-th layer`s feature from Llama-3.2-1B".format(args.tgt_layer))

    if args.model_name == 'pythia-160m':
        model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
        model.eval()
        sae = Sae.load_from_hub("EleutherAI/sae-pythia-160m-32k", hookpoint="layers.{}".format(args.tgt_layer))
        logger.info("Start extracting {}-th layer`s feature from pythia-160m".format(args.tgt_layer))

    if args.model_name == 'pythia-70m':
        model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
        model.eval()
        sae = Sae.load_from_hub("EleutherAI/sae-pythia-70m-deduped-32k", hookpoint="layers.{}".format(args.tgt_layer))
        logger.info("Start extracting {}-th layer`s feature from pythia-70m".format(args.tgt_layer))

    filenames = get_filenames_in_directory(args.data_path)
    for filename in filenames:
        data = read_json_data(args.data_path+'/'+filename)
        name_keys = filename.split('.json')[0].split('_')
        prefix_length = int(name_keys[1].replace('p',''))
        tgt_len = int(name_keys[2].replace('t',''))
        filter = False
        if 'unmemorized' in filename:
            filter = True
    
        count = 0
        all_distr = []
        logger.info(f"Start process: {args.data_path.split('/')[-1]+'/'+filename}, dataset size: {len(data)}")
        
        # save args
        output_prefix = f"{args.model_name}-p{prefix_length}-t{tgt_len}-context"
        output_dir = args.output_dir+'/layer'+args.tgt_layer+'/'+args.data_path.split('/')[-1]
        os.makedirs(output_dir, exist_ok=True)
        json.dump(args.__dict__, open(os.path.join(output_dir, output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

        # get feature attr
        with jsonlines.open(os.path.join(output_dir, output_prefix + '.rlt' + '.jsonl'), 'w') as fw:
            all_ppl = []
            all_high_freq_1 = []
            all_high_freq_2 = []
            all_count = []
            for idx, example in enumerate(tqdm(data)):
                sent = example['text']
                ppl = example["PPL"]
                if filter:
                    if float(ppl) < 2:
                        pass

                all_act = get_context_act(idx, sent, args, model, tokenizer, sae, device, prefix_length, tgt_len)

                if all_act:
                    res_dict = {
                        'idx': idx,
                        'text': sent,
                        'all_act': all_act,
                        'ppl': ppl
                    }
                    fw.write(res_dict)

                    all_ppl.append(ppl)

                    feature_pos = []

                    for act in all_act:
                        all_value = []    
                        for f_act in act:
                            all_value.append(f_act)
                            feature_pos.append(int(f_act[0]))
                    
                    frequency_counter = Counter(feature_pos)
                    all_count.append(len(frequency_counter))
                    

                    sorted_frequency = sorted(frequency_counter.items(), key=lambda x: x[1], reverse=True)

                    high_freq_1 = []
                    high_freq_2 = []
                    for element, frequency in sorted_frequency:
                        if frequency >= len(all_act)*args.threshold:
                            high_freq_2.append(element)
                            # print(f"{element} appeared {frequency} times")
                        if frequency >= len(all_act)*0.7:
                            high_freq_1.append(element)
                            # print(f"{element} appeared {frequency} times")
                    all_high_freq_1.append(len(high_freq_1))
                    all_high_freq_2.append(len(high_freq_2))
                    # print(f'high freq: {len(high_freq_2)},ppl: {ppl}')

                    # show_res = []
                    # for pos in high_freq:
                    #     show_res.append(find_tgt_pos_act(all_act,pos))
        logger.info(f'Average not-zero feature of {filename}: {sum(all_count)/len(all_count)}')
        logger.info(f'Average high-frequent(70%) feature of {filename}: {sum(all_high_freq_1)/len(all_high_freq_1)}')
        logger.info(f'Average high-frequent(50%) feature of {filename}: {sum(all_high_freq_2)/len(all_high_freq_2)}')
        logger.info(f'Average preplexity of {filename}: {sum(all_ppl)/len(all_ppl)}')
                    
        # logger.info(f"Saved in {os.path.join(output_dir, output_prefix + '.rlt' + '.jsonl')}")

    

if __name__ == "__main__":
    main()
