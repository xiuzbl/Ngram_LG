"""
Modified GPT-2 model for N-gram generation.
"""

import argparse
import importlib
import os
from typing import Any
import random
import torch
import texar.torch as tx
import torch.nn as nn
import numpy as np
from f1_score import f1_loss as F1
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint', type=str, default=None,
    help="Model checkpoint to load model weights from.")
parser.add_argument(
	"--pretrained-model-name", type=str, default="gpt2-small",
	choices=tx.modules.GPT2Decoder.available_checkpoints(),
	help="Name of the pre-trained checkpoint to load.")
parser.add_argument(
    '--config-train',type=str, default="config_train",
    help="Configurations of GPT-2 training, including data and "
         "optimization hyperparameters.")
# parser.add_argument(
#     '--config-train', action="store_true",
#     help="Configurations of GPT-2 training, including data and "
#          "optimization hyperparameters.")
parser.add_argument(
    "--output-dir", default="output/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    '--temperature', type=float, default=0.7,
    help="Softmax temperature for top-k sample decoding. Must be strictly "
         "greater than 0. Defaults to 0.7.")
parser.add_argument(
    '--top-k', type=int, default=40,
    help="The number of top most likely candidates from a vocab distribution.")
parser.add_argument(
    '--top-p', type=float, default=None,
    help="Select tokens with cumulative probability of at most 'p' when "
         "arranged in decreasing order. This will use "
         "TopPSampleEmbeddingHelper for decoding.")
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do-test", action="store_true",
    help="Whether to run test on the test set.")
parser.add_argument(
    "--neg_replace", type=str, default="one",
    help="Generate negative ngrams with different number of tokens replaced."
)

args = parser.parse_args()
config_train: Any = importlib.import_module(args.config_train)
torch.cuda.set_device(4)
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def calc_accuracy(pred,true,num_items):
	accu = (pred==true).sum(dtype=torch.float32)/num_items
	return accu

def main() -> None:
    """
    Builds the model and runs.
    """
    tx.utils.maybe_create_dir(args.output_dir)

    # parameters setting from config_train.py
    max_decoding_length = config_train.max_decoding_length
    vocab_size = config_train.vocab_size
    window_size = config_train.window_size
    num_neg = config_train.num_negatives
    # eval_neg = config_train.eval_neg

    # Build the GPT-2
    # tx.cuda.empty_cache()
    # model = tx.modules.GPT2Decoder(args.pretrained_model_name)
    model = tx.modules.GPT2Decoder() # Not use the pretrained weights.
    
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt)

  
    # print(model.state_dict().keys())
    # print(model.parameters)
    model.to(device)

    if max_decoding_length > model.hparams.position_size:
        raise ValueError(
            "max_decoding_length should not be greater than position size")

    # Create a GPT-2 tokenizer (BPE encoding)
    tokenizer = tx.data.GPT2Tokenizer(
        pretrained_model_name=args.pretrained_model_name)

    # Loads data
    datasets = {}
    if args.do_train:
        train_dataset = tx.data.RecordData(
            hparams=config_train.train_hparam, device=device)
        datasets['train'] = train_dataset
    if args.do_eval:
        eval_dataset = tx.data.RecordData(
            hparams=config_train.eval_hparam, device=device)
        datasets['eval'] = eval_dataset

    if args.do_test:
        test_dataset = tx.data.RecordData(
            hparams=config_train.test_hparam, device=device)
        datasets['test'] = test_dataset
    iterator = tx.data.DataIterator(datasets)
    train_op = tx.core.get_train_op(
        params=model.parameters(), hparams=config_train.opt)
    end_token = tokenizer.map_token_to_id('<|endoftext|>')

    def _get_helper(start_tokens):
        if args.top_p:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=end_token,
                p=args.top_p,
                softmax_temperature=args.temperature)
        else:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=end_token,
                top_k=args.top_k,
                softmax_temperature=args.temperature)
        return helper

    dis_steps = config_train.display_steps
    eval_steps = config_train.eval_steps
    eval_best = {"loss": 1e8}

    def _train_epoch():
        r"""Trains on the training set, and evaluates on the dev set
        periodically.
        """
        iterator.switch_to_dataset("train")
        model.train()
        accu = 0
        step = 0
        for batch in iterator:
            all_input_ids = batch["text_ids"]
            part_input_ids = []
            for sample in all_input_ids:
                sample = sample.tolist()

                # Calculate the sequence length without specitial tokens.
                sample = list(set(sample).difference(set([vocab_size - 1])))
                seq_length = len(sample)
                
                # Traverse the whole senquence through window slip.
                for i in range(seq_length):
                    if i + window_size <= seq_length:
                        all_ngram = []
                        pos_ngram = sample[i:i + window_size]
                         
                        if args.neg_replace=='one':
                            # Randomly replace one token based on the original samples to get the negative samples.
                            for k in range(num_neg):
                                random_position = random.randint(0,window_size - 1)  # Since all the special tokens are mapped to the last id. So we randomly choose one id instead of the last token.
                                rand_word = random.randint(0, vocab_size - 2)
                                pre_gram = sample[i:i + random_position] + sample[i + random_position + 1:i + window_size]  # need to check/
                                pre_gram.insert(random_position,rand_word)  # It's now a negative sample.
                                all_ngram.append(pre_gram)
                            # Randomly replace all tokens of the ngram to generate negative samples.
                        elif args.neg_replace=="all":
                            for k in range(num_neg):
                                neg_sample = random.choices([*range(vocab_size-1)],k=window_size)
                                all_ngram.append(neg_sample)
                        all_ngram.insert(0,pos_ngram)
                    part_input_ids.append(all_ngram)  # (all_partitions, num_samples, window_size)
            part_input_ids = torch.tensor(part_input_ids)

            all_partitions, num_sample, _ = list(part_input_ids.size())
            # Tensor flatten --> (all_partitions*num_samples, window_size)
            # Let all_partitions*num_samples=all_samples.
            all_samples = all_partitions * num_sample # Get the new batch_size.
            part_input_ids = torch.flatten(part_input_ids,end_dim=-2)
            input_ids = part_input_ids.to(device) 
            # input_ids -- (all_samples,window_size)
            
            # # Get the ground truth labels to compute loss and accuracy.
            true_labels = torch.tensor([[1]+[0]*num_neg for i in range(all_partitions)],dtype=torch.float32)
            true_labels = torch.flatten(true_labels, end_dim=-1).to(device) # shape (all_samples,1)
            true_labels_tocat =torch.tensor(true_labels[:,None],dtype=torch.long).to(device) # Add one dimension so that can concatenate with input data. 

            # Add shuffling
            cat = torch.cat((input_ids,true_labels_tocat),dim=1) # Concatenate data and their labels (all_samples, window_size+1)
            # print("After concatenation:",cat.size())
            shuffled_idx = torch.randperm(all_samples)
            shuffled_cat = cat[shuffled_idx] # Shuffle lines based on their index.
            new_input_ids = shuffled_cat[:,:window_size] # new_input_ids -- (all_samples, window_size)
            new_true_labels =torch.tensor(shuffled_cat[:,-1],dtype=torch.long) # new_true_labels -- (all_samples, 1)
            new_input_ids = torch.tensor(new_input_ids,dtype=torch.long)
            new_input_ids,new_true_labels = new_input_ids.to(device), new_true_labels.to(device)

            # Generate outputs.
            outputs = model(inputs=new_input_ids, decoding_strategy='train_greedy')
            logits = (outputs.logits).to(device)
            # logits --> (all_samples, 3, 1); ngram_logits --> (all_samples, 1, 1)
            ngram_logits = torch.mean(logits,dim=1,keepdim=True) # Choose the mean representation of n-grams to classify.
            ngram_logits = torch.flatten(ngram_logits,end_dim=-1).to(device) # ngram_logits --> (all_samples, 1)

            # Calculate loss through Binary Cross Entropy.
            cal_loss = nn.BCEWithLogitsLoss().to(device)
            new_true_labels = torch.tensor(new_true_labels, dtype=torch.float32).to(device)
            loss = cal_loss(ngram_logits,new_true_labels)

            # Compute accuracy by counting the number of matches. 
            # Get the predition labels by take the sigmoid function over the ngram logits.
            sigmoid = nn.Sigmoid()
            pred = sigmoid(ngram_logits)
            pred_labels = torch.tensor((pred>0.5)*1).to(device)

            accu = calc_accuracy(pred_labels,new_true_labels,all_samples)
            F1_score = F1(new_true_labels,pred_labels)

            loss.backward()
            train_op()

            if dis_steps > 0 and step % dis_steps == 0:
                print("step=%d, loss=%.4f, accuracy=%.4f, F1-score=%.4f"%(step, loss, accu,F1_score))
                sys.stdout.flush()

            if eval_steps > 0 and step % eval_steps == 0:
                _eval_epoch()
                model.train()

            step += 1

    @torch.no_grad()
    def _eval_epoch():
        r"""Evaluates on the dev set.
        """
        iterator.switch_to_dataset("eval")
        model.eval()

        nsamples = 0
        avg_rec = tx.utils.AverageRecorder()
        accu = 0
        # step = 0 

        """Same approches as training part."""
        for batch in iterator:
            all_input_ids = batch["text_ids"]

            part_input_ids = []
            for sample in all_input_ids:
                sample = sample.tolist()
                sample = list(set(sample).difference(set([vocab_size - 1])))
                seq_length = len(sample)
                for i in range(seq_length):
                    if i + window_size <= seq_length:
                        pos_ngram = sample[i:i + window_size]
                        all_ngram = []  # Store all the ngrams for each sequence.
                        
                        if args.neg_replace=="one":
                            for k in range(num_neg):
                                random_position = random.randint(0,window_size - 1) 
                                rand_word = random.randint(2, vocab_size - 1)
                                pre_gram = sample[i:i + random_position] + sample[i + random_position + 1:i + window_size]  # need to check
                                pre_gram.insert(random_position,rand_word)  # It's now a negative sample.
                                all_ngram.append(pre_gram)
                        elif args.neg_replace=="all":
                            for k in range(num_neg):
                                neg_sample = random.choices([*range(vocab_size-1)],k=window_size)
                                all_ngram.append(neg_sample)
                        
                        all_ngram.insert(0,pos_ngram)  # e.g. 1-pos, 10-neg
                    part_input_ids.append(all_ngram)  # (all_phrase, num_samples, window_size)
            part_input_ids = torch.tensor(part_input_ids)

            all_partitions, num_sample, _ = list(part_input_ids.size())
            part_input_ids = torch.flatten(part_input_ids,end_dim=-2)
            input_ids = part_input_ids.to(device)
            all_samples = all_partitions*num_sample # Get the new batch size.

            # Feed into the model.
            outputs = model(inputs=input_ids, decoding_strategy='train_greedy')
            logits = (outputs.logits).to(device)

            ngram_logits = torch.mean(logits,dim=1,keepdim=True) # Choose the last token (but not EOS token) representation to classify.
            ngram_logits = torch.flatten(ngram_logits,end_dim=-1).to(device)
           
            true_labels = torch.tensor([[1]+[0]*num_neg for i in range(all_partitions)],dtype=torch.float32)
            true_labels = torch.flatten(true_labels, end_dim=-1).to(device)
             
            cal_loss = nn.BCEWithLogitsLoss().to(device)
            loss = cal_loss(ngram_logits,true_labels)
            nsamples += all_samples
            
            # Get predicted labels.
            sigmoid = nn.Sigmoid()
            pred = sigmoid(ngram_logits)
            pred_labels = torch.tensor((pred>0.5)*1).to(device)

            accu = calc_accuracy(pred_labels,true_labels,all_samples)
            F1_score = F1(true_labels,pred_labels)

            avg_rec.add([loss,accu,F1_score],all_samples)

        print('*'*60)
        print("Evaluation loss:%.4f, accuracy:%.4f, F1-score:%.4f, nsamples:%d"%(avg_rec.avg(0),avg_rec.avg(1),avg_rec.avg(2),nsamples)) 
        sys.stdout.flush()

        if args.do_train and avg_rec.avg(0) < eval_best["loss"]:
            eval_best["loss"] = avg_rec.avg(0)
            eval_best["f1"] = avg_rec.avg(2)
            ckpt_fn = os.path.join(args.output_dir, 'model_best.ckpt')
            torch.save(model.state_dict(), ckpt_fn)
            print("Checkpoint best to {}".format(ckpt_fn))
            sys.stdout.flush()

    @torch.no_grad()
    def _test_epoch():
        r"""Generates samples on the test set. Have not changed this part.
        """
        iterator.switch_to_dataset("test")
        model.eval()

        _all_inputs = []
        _all_samples = []
        avg_rec = tx.utils.AverageRecorder()
        
        high_ngram = []
        mid_ngram = []
        small_ngram = []

        for batch in iterator:
            all_input_ids = batch["text_ids"]
            part_input_ids = []
            for sample in all_input_ids:
                sample = sample.tolist() 
                sample = list(set(sample).difference(set([vocab_size-1])))
                seq_length = len(sample)
                
                for i in range(seq_length):
                    if i + window_size<=seq_length:
                        pos_ngram = sample[i:i+window_size]
                        part_input_ids.append(pos_ngram)
            part_input_ids = torch.tensor(part_input_ids).to(device) # (all_pos, window_size)
            all_pos, _ = list(part_input_ids.size()) 
            
            true_labels = torch.tensor([1 for i in range(all_pos)],dtype=torch.float32).to(device)
            outputs = model(inputs=part_input_ids,decoding_strategy='train_greedy')
            logits = (outputs.logits).to(device)
            ngram_logits = torch.mean(logits,dim=1,keepdim=True)
            ngram_logits = torch.flatten(ngram_logits,end_dim=-1).to(device)
            
            sigmoid = nn.Sigmoid()
            pred = sigmoid(ngram_logits)
            pred_labels = torch.tensor((pred>0.5)*1).to(device)

            # TODO: classify ngrams based on their sigmoid scores.
            for k in range(all_pos):
                text_ngram = tokenizer.map_id_to_text(part_input_ids[k])
                if pred[k]>=0.8:
                    high_ngram.append(text_ngram)
                elif pred[k]>=0.4 and pred[k]<=0.6:
                    mid_ngram.append(text_ngram)
                elif pred[k]<=0.2:
                    small_ngram.append(text_ngram)
            
            
            accu = calc_accuracy(pred_labels,true_labels,all_pos)
            F1_score = F1(true_labels,pred_labels)
            avg_rec.add([accu, F1_score],all_pos)

        print('ngram with high score',high_ngram,flush=True)
        print('ngram with middle score',mid_ngram,flush=True)
        print('ngram with small score',small_ngram,flush=True)    
        print("Test accuracy:%.4f, F1-score:%.4f"%(avg_rec.avg(0),avg_rec.avg(1))) 
        sys.stdout.flush()

    if args.do_train:
        for _ in range(config_train.max_train_epoch):
            _train_epoch()
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, 'model.ckpt'))

    if args.do_eval:
        _eval_epoch()

    if args.do_test:
        _test_epoch()


if __name__ == "__main__":
    main()
