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

args = parser.parse_args()

config_train: Any = importlib.import_module(args.config_train)
# print(config_train)
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calc_accuracy(pred,true,num_items):
	accu = (pred==true).sum(dtype=torch.float32)/num_items
	return accu

def main() -> None:
    """
    Builds the model and runs.
    """
    tx.utils.maybe_create_dir(args.output_dir)

    max_decoding_length = config_train.max_decoding_length
    vocab_size = config_train.vocab_size
    window_size = config_train.window_size
    num_neg = config_train.num_negatives
    eval_neg = config_train.eval_neg

    # Build the GPT-2
    # tx.cuda.empty_cache()
    # model = tx.modules.GPT2Decoder(args.pretrained_model_name)
    model = tx.modules.GPT2Decoder()
    
    # if args.checkpoint:
    #     ckpt = torch.load(args.checkpoint)
    #     model.load_state_dict(ckpt['model'])
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
            hparams=test_hparam, device=device)
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
    #
    # eval_best = {"loss": 1e8, "ppl": 1e8}
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
                sample = list(set(sample).difference(set([vocab_size - 1])))
                seq_length = len(sample)
                for i in range(seq_length):
                    if i + window_size <= seq_length:
                        all_ngram = []
                        pos_ngram = sample[i:i + window_size]
                        random_position = random.randint(0,window_size - 1)  # We will replace one word randomly. If we want to replace two or more words, need to modify the code.
                        for k in range(num_neg):
                            rand_word = random.randint(2, vocab_size - 1)
                            pre_gram = sample[i:i + random_position] + sample[i + random_position + 1:i + window_size]  # need to check/
                            pre_gram.insert(random_position,rand_word)  # It's now a negative sample.
                            all_ngram.append(pre_gram)
                        all_ngram.insert(0,pos_ngram)  # e.g. 1-pos, 10-neg
                    part_input_ids.append(all_ngram)  # (all_phrase, num_samples, window_size)
            part_input_ids = torch.tensor(part_input_ids)

            all_partitions, num_sample, _ = list(part_input_ids.size())
            # Tensor flatten --> (all_phrases*num_samples, window_size)
            part_input_ids = torch.flatten(part_input_ids,end_dim=-2)
            part_input_ids = part_input_ids.to(device)
            outputs = model(inputs=part_input_ids, decoding_strategy='train_greedy')
            logits = (outputs.logits).to(device)
            # logits -- (all_samples, 3, 2); ngram_logits -- (all_samples, 1, 2)
            ngram_logits = torch.mean(logits,dim=1,keepdim=True) # Choose the last token (but not EOS token) representation to classify.
            ngram_logits = torch.flatten(ngram_logits,end_dim=-1).to(device)
            true_labels = torch.tensor([[1]+[0]*num_neg for i in range(all_partitions)],dtype=torch.float32)
            true_labels = torch.flatten(true_labels, end_dim=-1).to(device)
             
            cal_loss = nn.BCEWithLogitsLoss().to(device)
            loss = cal_loss(ngram_logits,true_labels)
            # Compute accuracy 
            sigmoid = nn.Sigmoid()
            pred = sigmoid(ngram_logits)
            pred_labels = torch.tensor((pred>0.5)*1).to(device)
            print(pred_labels)
            accu = calc_accuracy(pred_labels,true_labels,all_partitions*num_sample)

            loss.backward()
            train_op()

            if dis_steps > 0 and step % dis_steps == 0:
                print("step=%d, loss=%.4f, accuracy=%.4f"%(step, loss, accu))

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
        step = 0 
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
                        random_position = random.randint(0,window_size - 1)  # We will replace one word randomly. If we want to replace two or more words, need to modify the code.
                        all_ngram = []  # Store all the ngrams for each sequence.
                        for k in range(num_neg):
                            rand_word = random.randint(2, vocab_size - 1)
                            pre_gram = sample[i:i + random_position] + sample[i + random_position + 1:i + window_size]  # need to check
                            pre_gram.insert(random_position,rand_word)  # It's now a negative sample.
                            all_ngram.append(pre_gram)
                        all_ngram.insert(0,pos_ngram)  # e.g. 1-pos, 10-neg
                    part_input_ids.append(all_ngram)  # (all_phrase, num_samples, window_size)
            part_input_ids = torch.tensor(part_input_ids)

            all_partitions, num_sample, _ = list(part_input_ids.size())
            part_input_ids = torch.flatten(part_input_ids,end_dim=-2)
            part_input_ids = part_input_ids.to(device)
            outputs = model(inputs=part_input_ids, decoding_strategy='train_greedy')
            logits = (outputs.logits).to(device)

            ngram_logits = torch.mean(logits,dim=1,keepdim=True) # Choose the last token (but not EOS token) representation to classify.
            ngram_logits = torch.flatten(ngram_logits,end_dim=-1).to(device)
           
            true_labels = torch.tensor([[1]+[0]*num_neg for i in range(all_partitions)],dtype=torch.float32)
            true_labels = torch.flatten(true_labels, end_dim=-1).to(device)
             
            cal_loss = nn.BCEWithLogitsLoss().to(device)
            loss = cal_loss(ngram_logits,true_labels)
            # accu += calc_accuracy(pred_labels,true_labels,all_partitions*num_sample)
            nsamples += all_partitions*num_sample
            step +=1 
            avg_rec.add([loss],all_partitions)

        print("eval loss:%.4f, eval nsamples:%d"%(avg_rec.avg(0), nsamples)) # may have some problems

        if args.do_train and avg_rec.avg(0) < eval_best["loss"]:
            eval_best["loss"] = avg_rec.avg(0)
            # eval_best["ppl"] = avg_rec.avg(1)
            # ckpt_fn = os.path.join(args.output_dir, 'model_best.ckpt')
            # torch.save(model.state_dict(), ckpt_fn)
            # print("Checkpoint best to {}".format(ckpt_fn))

    @torch.no_grad()
    def _test_epoch():
        r"""Generates samples on the test set.
        """
        iterator.switch_to_dataset("test")
        model.eval()

        _all_inputs = []
        _all_samples = []

        for batch in iterator:
            input_ids = batch["text_ids"]
            length = batch["length"]
            start_tokens = input_ids[:, 0]
            helper = _get_helper(start_tokens)

            output, _ = model(
                context=input_ids,
                context_sequence_length=length,
                max_decoding_length=max_decoding_length,
                helper=helper)
            sample_id = output.sample_id

            _inputs = []
            for i, l in zip(input_ids, length):
                # Delete padding
                _inputs.append(i[:l].tolist())
            _all_inputs.extend(_inputs)

            _samples = []
            for s, l in zip(sample_id, length):
                # Delte inputs from samples
                _samples.append(s[l:].tolist())
            _all_samples.extend(_samples)

        # Parse samples and write to file

        eos_token_id = tokenizer.map_token_to_id('<|endoftext|>')

        _all_input_text = []
        for i in _all_inputs:
            if i[0] == eos_token_id:
                # '<|endoftext|>' is used as the BOS token. Delete it here
                i = i[1:]
            i_text = tokenizer.map_id_to_text(i)
            _all_input_text.append(i_text)
        # '<|endoftext|>' is used as the PAD token. Delete them here
        _all_input_text = tx.utils.strip_eos(_all_input_text,
                                             eos_token='<|endoftext|>')

        _all_samples_text = []
        for i, s in zip(_all_inputs, _all_samples):
            s_text = tokenizer.map_id_to_text(s)
            s_text = s_text.replace('\n', ' ')
            _all_samples_text.append(s_text)
        _all_samples_text = tx.utils.strip_eos(_all_samples_text,
                                               eos_token='<|endoftext|>')

        output_file = os.path.join(args.output_dir, "test_samples.tsv")
        print('Write samples to {}'.format(output_file))
        tx.utils.write_paired_text(
            _all_input_text, _all_samples_text, output_file)

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
    
    
    
    
    
    
    