import numpy as np
import random
from utils.utils import normalize_answer
from evaluation import parlai_metrics
import language_evaluation


# first, eval with bckim 's code
# second eval with huggingface module
# 여기서 perplextiy 다시 구해야 할듯? 아니면 거기 loss에서 knowledge loss 빼고? 
class ComputeMetrics:
    def __init__(self, tokenizer, loss_names=None, perplexity_key='token', show_num=3):
        self.tokenizer = tokenizer
        self.loss_names = loss_names if loss_names is not None else []
        self.perplexity_key = perplexity_key
        self.show_num = show_num
        self.rouge_evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=1, tokenization_fn=normalize_answer)


    def __call__(self, eval_preds):
        preds, labels, losses_tuple = eval_preds
        losses, correct_know_num = losses_tuple

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        label_sen_length = np.sum(labels != self.tokenizer.pad_token_id, axis=-1)
        example_mask = (label_sen_length > 0).astype(np.int64)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        predictions = [dp for idx,dp in enumerate(decoded_preds) if label_sen_length[idx] > 0]
        labels = [dl for idx,dl in enumerate(decoded_labels) if label_sen_length[idx] > 0]
        
        self.print_random_result(predictions, labels, self.show_num)

        # token matching metrics
        # Rouge-1,2,L
        rouge_result = self.rouge_evaluator.run_evaluation(predictions, labels)
        
        # F1-score
        f1_result = parlai_metrics.get_f1_result(predictions, labels)
        
        # BLEU-1,2,3,4
        bleu_result = parlai_metrics.get_bleu_result(predictions, labels)
        
        # Distinct-1,2
        distinct_result = parlai_metrics.get_distinct_result(predictions) # inter distinct scaled as total token num
        
        # get knowledge accuracy
        know_acc_result = self.get_know_acc(correct_know_num, len(labels))

        # get loss and perplexity
        loss_result = self.get_eval_loss(losses)

        # combine metrics results
        loss_result.update(know_acc_result)
        loss_result.update(f1_result)
        loss_result.update(rouge_result)
        loss_result.update(bleu_result)
        loss_result.update(distinct_result)
        loss_result.update({
            'total_sentence_num' : np.sum(example_mask),
            'total_episode_num' : losses.shape[0]
        })
        return loss_result

    # 나중에 한번더 확인하기
    def get_eval_loss(self, losses):
        _, loss_type_num = losses.shape
        len_loss_names = len(self.loss_names)
        if loss_type_num-len_loss_names > 0:
            expanded_loss_names = self.loss_names + [f'additional_loss_{i+1}' 
                                                    for i in range(loss_type_num-len_loss_names)]
        else:
            expanded_loss_names = self.loss_names
        total_loss_mean = np.mean(np.sum(losses, axis=-1))
        each_loss_mean = np.mean(losses, axis=0)
        loss_result_dict = {'total_loss': total_loss_mean.item()}

        for i in range(len(each_loss_mean)):
            loss_result_dict[expanded_loss_names[i]] = each_loss_mean.item(i)
            if self.perplexity_key in expanded_loss_names[i]:
                loss_result_dict['ppl'] = np.exp(each_loss_mean.item(i))

        return loss_result_dict


    def get_know_acc(self, correct_know_num, num_predictions):
        know_acc = correct_know_num / num_predictions
        know_acc = know_acc * 100
        return {'know_acc':know_acc}


    def print_random_result(self, generations, answers, show_num=3):
        show_indices = random.sample(range(len(generations)), show_num)
        print('='*200)
        for index in show_indices:
            print(f"res : {answers[index]}")
            print(f"gen : {generations[index]}")
            print()
        print('-'*200)



    


