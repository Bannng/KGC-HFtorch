from utils.wow_trainer import WoWTrainer
from dataset.wizard_of_wikipedia_dataset import WowTorchDataset, WowEvalTorchDataset
from dataset.collator import WowCollator
from evaluation.compute_metrics import ComputeMetrics
from transformers import Seq2SeqTrainingArguments, set_seed
import torch
import argparse

from models.Transformer import Transformer
from models.TMemNet import TMemNet, TMemNetBert
from models.TitleNet import TitleNet
from models.MemBoB import MemNetBoB, BoBTMemNetBert
from models.MemBoB2 import BoBTMemNetBert2
from models.PostKS import PostKSBert


# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


def get_args():
    parser = argparse.ArgumentParser()

    # for distributed training
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_gpu_num", type=int, default=1, help='just for count save and logging step') 
    
    # arguments for collator                                                        
    parser.add_argument("--max_length", type=int, default=51)
    parser.add_argument("--max_episode_length", type=int, default=5)
    parser.add_argument("--max_knowledge", type=int, default=32)
    parser.add_argument("--knowledge_truncate", type=int, default=34)

    # arguments for model selection
    parser.add_argument("--model_type", type=str, choices=["Transformer",
                                                           "TMemNet",
                                                           "TMemNetBert",
                                                           "PostKSBert",
                                                           "SKT",
                                                           "TitleNet"
                                                           ],
                                                        default="TMemNetBert")
    parser.add_argument("--use_cs_ids", action='store_true') # becareful with this
    parser.add_argument("--knowledge_alpha", type=float, default=0.25)
    parser.add_argument("--max_title_num", type=int, default=5)

    # arguments for training
    parser.add_argument("--output_dir", type=str, default='/home/byeongjoo/works/KGC-torch/output')
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default='constant')
    parser.add_argument("--weight_decay", type=float, default=0.0) # set this 0.0 for Adam Optimizer
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.4)
    parser.add_argument("--logging_num_per_epoch", type=int, default=20)
    parser.add_argument("--save_num_per_epoch", type=int, default=100000)
    parser.add_argument("--disable_tqdm", action='store_true')
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    
    # for resume training
    parser.add_argument("--ignore_data_skip", action='store_true')
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    args = parser.parse_args()
    return args


def select_model(args):
    model_type = args.model_type

    # model type
    if model_type == "Transformer":
        model = Transformer()

    elif model_type == "TMemNet":
        model = TMemNet(
            use_cs_ids=args.use_cs_ids,
            knowledge_alpha=args.knowledge_alpha,
        )

    elif model_type == "TMemNetBert":
        model = TMemNetBert(
            use_cs_ids=args.use_cs_ids,
            knowledge_alpha=args.knowledge_alpha,
        )

    elif model_type == "TitleNet":
        model = TitleNet(
            use_cs_ids=args.use_cs_ids,
            knowledge_alpha=args.knowledge_alpha,
            max_title_num=args.max_title_num,
        )

    elif model_type == "MemBoB":
        model = MemNetBoB(
            use_cs_ids=args.use_cs_ids,
            knowledge_alpha=args.knowledge_alpha,
            max_title_num=args.max_title_num,
        )

    elif model_type == "PostKSBert":
        model = PostKSBert(
            use_cs_ids=args.use_cs_ids,
            knowledge_alpha=args.knowledge_alpha,
        )

    elif model_type == "SKT":
        raise ValueError("choose your model type properly")

    else:
        raise ValueError("choose your model type properly")

    return model




def train(args):
    # set seed and device
    set_seed(args.seed)
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)

    # set tokenizer, model, collator
    # model = select_model(args)
    # model = MemNetBoB(knowledge_mode="2")
    # model = BoBTMemNetBert(knowledge_mode="context_only", concat_query=False) # context_only, argmax, pool
    # model = BoBTMemNetBert2(knowledge_mode="argmax", concat_query=True) # context_only, argmax, pool
    model = PostKSBert()
    # 일단 argmax는 나중에 실험

    
    if args.local_rank == -1 or args.local_rank == 0:
        print(f'Model name : {type(model)}')
    
    collator = WowCollator(
        model.tokenizer,
        max_length=args.max_length,
        max_episode_length=args.max_episode_length,
        max_knowledge=args.max_knowledge,
        knowledge_truncate=args.knowledge_truncate,
    )

    # dataset path
    data_path = '/home/byeongjoo/works/KGC-torch/cache/wizard_of_wikipedia'
    cache_dir = '/home/byeongjoo/works/KGC-torch/cache'

    # load dataset
    train_dataset = WowTorchDataset(tokenizer=model.tokenizer,
                                    data_path=data_path,
                                    cache_dir=cache_dir,
                                    mode='train',
                                    token_preprocessed=False)
    eval_dataset = WowEvalTorchDataset(tokenizer=model.tokenizer,
                                       data_path=data_path,
                                       cache_dir=cache_dir,
                                       seen_mode='valid',
                                       unseen_mode='valid_unseen',
                                       token_preprocessed=False)
    test_dataset = WowEvalTorchDataset(tokenizer=model.tokenizer,
                                       data_path=data_path,
                                       cache_dir=cache_dir,
                                       seen_mode='test',
                                       unseen_mode='test_unseen',
                                       token_preprocessed=False)
    eval_seen_dataset = eval_dataset.seen_data
    eval_unseen_dataset = eval_dataset.unseen_data

    test_seen_dataset = test_dataset.seen_data
    test_unseen_dataset = test_dataset.unseen_data
    
    # set steps for logging, eval and save
    if torch.cuda.device_count() != args.total_gpu_num and args.local_rank==-1:
        args.total_gpu_num = torch.cuda.device_count()
        print('gpu num changed!')

    train_len = len(train_dataset)
    total_batch_size = args.per_device_train_batch_size * args.total_gpu_num * args.gradient_accumulation_steps
    epoch_per_steps = int(train_len / total_batch_size)

    logging_steps = int(epoch_per_steps / args.logging_num_per_epoch)
    save_steps = int(epoch_per_steps / args.save_num_per_epoch)
    logging_steps = logging_steps if logging_steps > 0 else 1
    save_steps = save_steps if save_steps > 0 else 1
    
    if args.local_rank == -1 or args.local_rank == 0:
        print(f'logging_steps : {logging_steps}, save_steps : {save_steps}, epoch_per_steps : {epoch_per_steps}')


    # generation configuration
    gen_kwargs = {
        'max_length' : args.max_length,
        'do_sample' : False,
        'num_beams' : 1,
        'no_repeat_ngram_size' : 0,
    }

    # set Metrics
    compute_metrics = ComputeMetrics(tokenizer=model.tokenizer, loss_names=['token_loss', 'knowledge_loss'])


    # set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        prediction_loss_only=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=save_steps,
        seed=args.seed,
        local_rank=args.local_rank,
        disable_tqdm=args.disable_tqdm,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        ignore_data_skip=args.ignore_data_skip,
        predict_with_generate=True,
        generation_max_length=gen_kwargs['max_length'],
        generation_num_beams=gen_kwargs['num_beams'],
        lr_scheduler_type=args.lr_scheduler_type,
    )

    # set trainer
    trainer = WoWTrainer(
        tokenizer=model.tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_seen_dataset,
        unseen_eval_dataset=eval_unseen_dataset,
        test_dataset=test_seen_dataset,
        test_unseen_dataset=test_unseen_dataset,
        gen_kwargs=gen_kwargs,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # training
    if args.resume_checkpoint is not None:
        trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    else:
        trainer.train()


if __name__ == "__main__":
    # should be transformer over 4.13
    train_args = get_args()
    if train_args.local_rank == -1 or train_args.local_rank == 0:
        print(train_args)

    train(train_args)