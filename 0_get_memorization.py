import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset 
from datasets import load_dataset
import argparse
import math
import random
import numpy as np
import json
import os
from tqdm import tqdm
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_ppl(model, input_ids, prompt_length, pad_token_id):

    # 确保输入形状为 2D (batch_size, seq_len)
    input_tensor = input_ids.unsqueeze(0).to(model.device) if input_ids.dim() == 1 else input_ids.to(model.device)
    labels = input_tensor.clone()

    # 忽略 prompt 部分的损失
    labels[:, :prompt_length] = -100
    # 忽略 padding 部分的损失
    labels[labels == pad_token_id] = -100

    with torch.no_grad():
        outputs = model(input_tensor, labels=labels)
        loss = outputs.loss.item()  # 获取损失值

    # 计算 PPL（通过取指数得到）
    ppl = torch.exp(torch.tensor(loss)).item()

    return ppl

# def evaluate_batch_generation(dataset, tokenizer, model, max_samples=100, target_length=10, prompt_length=50, batch_size=8):
#     exact_match_count = 0
#     processed_samples = 0
#     matched_samples = []
#     not_match_samples = []
#     mppl_list = []
#     nppl_list = []

    
#     # 逐批处理样本
#     for start_idx in tqdm(range(0, min(len(dataset), max_samples), batch_size), desc="Processing batches"):
#         batch_samples = dataset[start_idx:start_idx + batch_size]["query"]

#         prompts = []
#         targets = []
#         completes = []
        
#         for sample in batch_samples:
#             text = sample
#             tokenized = tokenizer(text, return_tensors="pt", truncation=True)
#             input_ids = tokenized["input_ids"].squeeze()

#             if len(input_ids) < prompt_length + target_length:  # 跳过太短样本
#                 continue

#             prompts.append(input_ids[:prompt_length])
#             targets.append(input_ids[prompt_length:prompt_length + target_length])
#             completes.append(input_ids[:prompt_length + target_length])

#         if not prompts:
#             continue


#         # 逐样本计算 PPL
#         for i, complete in enumerate(completes):

#             complete_tensor = complete.unsqueeze(0).to(model.device)
#             labels = complete_tensor.clone()
#             labels[:, :prompt_length] = -100  # 忽略 prompt 部分的损失

#             # 如果模型没有显式的 pad_token_id，设置为 eos_token_id
#             if tokenizer.pad_token_id is None:
#                 tokenizer.pad_token_id = tokenizer.eos_token_id
            
#             # 忽略 padding 的损失计算
#             labels[labels == tokenizer.pad_token_id] = -100

        
#         # 对 prompts 进行填充，形成批量输入
#         prompts_padded = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=tokenizer.pad_token_id).to(model.device)
#         attention_mask = (prompts_padded != tokenizer.pad_token_id).long()  # 创建 attention mask
        

#         # print('prompt size: ',prompts_padded.size())
#         # 使用批量生成
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 prompts_padded,
#                 attention_mask=attention_mask,
#                 pad_token_id=tokenizer.pad_token_id,
#                 max_length=prompts_padded.size(1) + max(len(t) for t in targets),
#                 do_sample=False
#             )

#         # 解码生成结果并进行比较
#         # print('generation size: ',generated_ids.size())
#         for idx, generated_id in enumerate(generated_ids):
#             generated_text = tokenizer.decode(generated_id[prompts[idx].size(0):], skip_special_tokens=True)
#             target_text = tokenizer.decode(targets[idx], skip_special_tokens=True)
#             complete_text = tokenizer.decode(completes[idx], skip_special_tokens=True)

#             # print(generated_text + '##' + target_text)

#             if generated_text == target_text:
#                 exact_match_count += 1
#                 matched_samples.append({
#                     "text": complete_text,
#                     "PPL": ppl
#                 })
#                 mppl_list.append(ppl)
#             else:
#                 if len(not_match_samples) < 200:
#                     not_match_samples.append({
#                         "text": complete_text,
#                         "PPL": ppl
#                     })
#                 nppl_list.append(ppl)

#         processed_samples += len(batch_samples)

#     accuracy = exact_match_count / processed_samples * 100 if processed_samples > 0 else 0
#     logger.info(f"复述准确率: {accuracy:.2f}%")
#     if len(mppl_list) != 0:
#         logger.info(f"记忆样本平均困惑度: {sum(mppl_list) / len(mppl_list) if mppl_list else float('inf'):.2f}")
#     else:
#         logger.info(f"记忆样本个数为0")
#     logger.info(f"非记忆样本平均困惑度: {sum(nppl_list) / len(nppl_list) if nppl_list else float('inf'):.2f}")
#     return matched_samples, not_match_samples

def evaluate_batch_generation(dataset, tokenizer, model, max_samples=100, target_length=10, prompt_length=50, batch_size=8):
    """
    评估批量生成的精确匹配率和困惑度（PPL）。

    参数：
        dataset: 数据集，包含 query 字段。
        tokenizer: 模型的分词器。
        model: 待评估的模型。
        max_samples: 最大评估样本数。
        target_length: 目标生成文本的长度。
        prompt_length: 提示文本的长度。
        batch_size: 批量大小。

    返回：
        matched_samples: 精确匹配的样本列表。
        not_match_samples: 未匹配的样本列表。
    """
    exact_match_count = 0
    processed_samples = 0
    matched_samples = []
    not_match_samples = []
    mppl_list = []
    nppl_list = []

    # print(dataset[0])
    for start_idx in tqdm(range(0, min(len(dataset), max_samples), batch_size), desc="Processing batches"):
        # 获取当前批次的 query 样本
        if "query" in dataset[0].keys():
            batch_samples = dataset[start_idx:start_idx + batch_size]["query"]
        else:
            batch_samples = dataset[start_idx:start_idx + batch_size]["text"]

        prompts, targets, completes = [], [], []

        # 处理每个样本
        for sample in batch_samples:
            tokenized = tokenizer(sample, return_tensors="pt", truncation=True)
            input_ids = tokenized["input_ids"].squeeze()

            # 跳过长度不足的样本
            if len(input_ids) < prompt_length + target_length:
                continue

            prompts.append(input_ids[:prompt_length])
            targets.append(input_ids[prompt_length:prompt_length + target_length])
            completes.append(input_ids[:prompt_length + target_length])

        # 如果当前批次无有效样本，跳过
        if not prompts:
            continue

        # 填充 prompts 并创建 attention mask
        prompts_padded = torch.nn.utils.rnn.pad_sequence(
            prompts, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(model.device)
        attention_mask = (prompts_padded != tokenizer.pad_token_id).long()

        # 批量生成文本
        with torch.no_grad():
            generated_ids = model.generate(
                prompts_padded,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_length=prompts_padded.size(1) + target_length,
                do_sample=False
            )

        # 逐条对比生成结果
        for idx, generated_id in enumerate(generated_ids):
            generated_text = tokenizer.decode(
                generated_id[prompts[idx].size(0):], skip_special_tokens=True
            )
            target_text = tokenizer.decode(targets[idx], skip_special_tokens=True)
            complete_text = tokenizer.decode(completes[idx], skip_special_tokens=True)

            ppl = calculate_ppl(model, completes[idx], prompt_length, tokenizer.pad_token_id)  # 计算困惑度

            if generated_text == target_text:
                exact_match_count += 1
                matched_samples.append({"text": complete_text, "PPL": ppl})
                mppl_list.append(ppl)
            else:
                if len(not_match_samples) < 200:  # 限制未匹配样本数量
                    not_match_samples.append({"text": complete_text, "PPL": ppl})
                nppl_list.append(ppl)

        processed_samples += len(batch_samples)

    # 计算评估指标
    accuracy = (exact_match_count / processed_samples * 100) if processed_samples > 0 else 0
    logger.info(f"复述准确率: {accuracy:.2f}%")
    logger.info(f"记忆样本平均困惑度: {sum(mppl_list) / len(mppl_list):.2f}" if mppl_list else "记忆样本个数为0")
    logger.info(f"非记忆样本平均困惑度: {sum(nppl_list) / len(nppl_list):.2f}" if nppl_list else "非记忆样本个数为0")

    return matched_samples, not_match_samples



def load_arxiv_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 构造 Hugging Face Dataset
    return Dataset.from_dict({"text": [item["text"] for item in data]})


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="Dataset name as output prefix to indentify each running of experiment.")
    parser.add_argument("--model_path", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Path to local pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_idx",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--prefix_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--tgt_len",
                    default=10,
                    type=int,
                    help="target text length.")
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
    parser.add_argument("--batch_size",
                        default=20,
                        type=int,
                        help="Total batch size for cut.")

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
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=device)
    model.eval()

    # 设置 pad_token 为 [PAD] 或其他特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 将 pad_token 设置为 eos_token 或其他适当标记

    # 加载数据集
    if args.dataset_name == "wikitext-103":
        dataset = load_dataset("Hieuman/wikitext-103-filtered", split="train", trust_remote_code=True)
    elif args.dataset_name == "openwebtext":
        dataset = load_dataset("stas/openwebtext-10k", split="train", trust_remote_code=True)
    # elif args.dataset_name == "bookscorpus":
    #     dataset = load_dataset("rojagtap/bookcorpus", split="train", trust_remote_code=True)
    elif args.dataset_name == "pile_arxiv":
        dataset = load_arxiv_dataset('./dataset/pile_arxiv/train.json')
    elif args.dataset_name == "2024_wiki":
        dataset = load_arxiv_dataset('./dataset/2024_new/wiki_2025.json')
    elif args.dataset_name == "2024_arxiv":
        dataset = load_arxiv_dataset('./dataset/2024_new/arxiv_2024.json')
    elif args.dataset_name == "2024_reddit":
        dataset = load_arxiv_dataset('./dataset/2024_new/reddit_2024.json')
    
    matched_samples, not_match_samples = evaluate_batch_generation(
        dataset, tokenizer, model, max_samples=100000000, target_length=args.tgt_len, 
        prompt_length=args.prefix_length, batch_size=args.batch_size
    )

    # 保存 matched_samples 和 not_match_samples 到 JSON 文件
    #dir_path = os.path.dirname(f'./data/{args.dataset_name}-{args.output_idx}/')
    dir_path = os.path.dirname(f'./data/{args.dataset_name}/')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if len(matched_samples) != 0:
        with open(f'{dir_path}/memorized_p{args.prefix_length}_t{args.tgt_len}.json', "w", encoding="utf-8") as f:
            json.dump(matched_samples, f, ensure_ascii=False, indent=4)
    with open(f'{dir_path}/unmemorized_p{args.prefix_length}_t{args.tgt_len}.json', "w", encoding="utf-8") as f:
        json.dump(not_match_samples, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
