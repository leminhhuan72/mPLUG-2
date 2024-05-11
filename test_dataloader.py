import argparse
import os
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_videoqa_mplug import MPLUG2
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer

import warnings
warnings.filterwarnings("ignore")


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=True, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 1
    warmup_iterations = warmup_steps * step_size

    len_batch = len(data_loader)
    print("Total Batch {}".format(len_batch))

    for i, (video, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video, weights = video.to(device, non_blocking=True), weights.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=args.max_input_length if config["add_ocr"] else 25, return_tensors="pt").to(
            device)
        if i == 0:
            print ("question: ", question)
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(video, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)
        if accum_steps > 1:
           loss = loss / accum_steps

        if do_amp:
           from apex import amp
           with amp.scale_loss(loss, optimizer) as scaled_loss:
               # logger.info('scaled loss: {}'.format(str(scaled_loss)))
               scaled_loss.backward()
        else:
           loss.backward()
        if (i + 1) % accum_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

        # model.backward(loss)
        # model.step()
        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        elif scheduler.step_mode:
            scheduler.step(epoch * len_batch + i)
        del video, weights, question_input,answer_input, loss
            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VideoQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    for n, (video, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video = video.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

        topk_ids, topk_probs = model(video, question_input, answer_input, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            result.append({"question_id":ques_id, "answer":ans})   

    return result

@torch.no_grad()
def evaluate_(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
    for n, (video, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        video = video.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(video, question_input, answer_input, train=False, k=config['k_test'])      
        result = []
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            if config.get('open_generation', True):
                ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                result.append({"question_id":ques_id, "answer":ans})   
            else:
                _, pred = topk_prob.max(dim=0)
                result.append({"question_id": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})
        accuracy = cal_metric(result, dataset)
        
        metric_logger.meters['acc'].update(accuracy, n=video.size(0))

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, answer_list, rerank, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    
    if answer_list.split('.')[-1] == 'json':
        answer_list = list(json.load(open(answer_list, 'r')).keys())
    else:
        answer_list = list(set([x['answer'] for x in load_jsonl(answer_list)]))
    
    answer_list_ = [answer+config['eos'] for answer in answer_list]
    answer_input = tokenizer(answer_list_, padding='longest', return_tensors='pt').to(device)    
    for n, (video, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        video = video.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(video, question_input, answer_input, train=False, k=config['k_test'], rerank=rerank)      
        result = []
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            if not rerank:
                ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                result.append({"question_id":ques_id, "answer":ans})   
            else:
                _, pred = topk_prob.max(dim=0)
                result.append({"question_id": ques_id, "answer": answer_list[topk_id[pred]]})
        accuracy = cal_metric(result, dataset)
        
        metric_logger.meters['acc'].update(accuracy, n=video.size(0))

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def cal_metric(vqa_result, val_file):
    with open(val_file, "r") as f:
        data_list = [json.loads(l.strip("\n")) for l in f.readlines()]
    id2datum = {}
    for idx, each in enumerate(data_list):
        question_id = idx
        id2datum[question_id] = {
            'question': each['question'],
            'video_id': each['video_id'],
            'answer': each['answer'],
        }
    score = 0.
    for each in vqa_result:
        quesid = each["question_id"]
        ans = each["answer"]
        label = id2datum[quesid]['answer']
        if label == ans:
            score += 1
    return score / len(vqa_result)

def main(args, config):
    print('master addr: ', os.environ['MASTER_ADDR'])
    print('master port: ', os.environ['MASTER_PORT'])

    utils.init_distributed_mode(args)    
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)


    #### Dataset ####
    print("Creating video qa datasets")
    if args.no_randaug:
        datasets = create_dataset('video_qa_no_randaug', config)
    else:
        datasets = create_dataset('video_qa', config)

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--max_input_length', default=50, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--no_randaug', action='store_true')
    parser.add_argument('--accum_steps', default=1, type=int)

     # Model architecture
    parser.add_argument('--temporal_stride', default=2, type=int)
    parser.add_argument('--temporal_downsampling', action='store_true')
    # parser.add_argument('--use_st', action='store_true')
    # parser.add_argument('--double_lmhra', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["beam_size"] = args.beam_size
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder

    config['temporal_stride'] = args.temporal_stride
    config['temporal_downsampling'] = args.temporal_downsampling
    # config['double_lmhra'] = args.double_lmhra
    # config['use_st'] = args.double_lmhra
    config['accum_steps'] = args.accum_steps
    config['no_randaug'] = args.no_randaug

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
