from parlai.core.metrics import F1Metric, BleuMetric, RougeMetric, IntraDistinctMetric, InterDistinctMetric


def get_f1_result(preds_list, ans_list):
    total_score = None
    for idxs in range(len(ans_list)):
        cur_f1_score = F1Metric.compute(preds_list[idxs], [ans_list[idxs]])
        if total_score is None:
            total_score = cur_f1_score
        else:
            total_score += cur_f1_score
    return {'F1' : total_score.value()}


def get_bleu_result(preds_list, ans_list):
    total_bleu1 = None
    total_bleu2 = None
    total_bleu3 = None
    total_bleu4 = None
    for idxs in range(len(ans_list)):
        cur_bleu1 = BleuMetric.compute(preds_list[idxs], [ans_list[idxs]], k=1)
        cur_bleu2 = BleuMetric.compute(preds_list[idxs], [ans_list[idxs]], k=2)
        cur_bleu3 = BleuMetric.compute(preds_list[idxs], [ans_list[idxs]], k=3)
        cur_bleu4 = BleuMetric.compute(preds_list[idxs], [ans_list[idxs]], k=4)
        # bleu1
        if total_bleu1 is None: 
            total_bleu1 = cur_bleu1
        else:
            total_bleu1 += cur_bleu1

        # bleu2
        if total_bleu2 is None: 
            total_bleu2 = cur_bleu2
        else:
            total_bleu2 += cur_bleu2

        # bleu3
        if total_bleu3 is None: 
            total_bleu3 = cur_bleu3
        else:
            total_bleu3 += cur_bleu3

        # bleu4
        if total_bleu4 is None: 
            total_bleu4 = cur_bleu4
        else:
            total_bleu4 += cur_bleu4

    return {
        'BLEU1' : total_bleu1.value(),
        'BLEU2' : total_bleu2.value(),
        'BLEU3' : total_bleu3.value(),
        'BLEU4' : total_bleu4.value(),
    }


def get_rouge_result(preds_list, ans_list):
    total_rouge1 = None
    total_rouge2 = None
    total_rougeL = None
    for idxs in range(len(ans_list)):
        cur_rouge1, cur_rouge2, cur_rougeL = RougeMetric.compute_many(preds_list[idxs], [ans_list[idxs]])

        # rouge1
        if total_rouge1 is None: 
            total_rouge1 = cur_rouge1
        else:
            total_rouge1 += cur_rouge1

        # rouge2
        if total_rouge2 is None: 
            total_rouge2 = cur_rouge2
        else:
            total_rouge2 += cur_rouge2

        # rougeL
        if total_rougeL is None: 
            total_rougeL = cur_rougeL
        else:
            total_rougeL += cur_rougeL
    
    return {
        'Rouge1' : total_rouge1.value(),
        'Rouge2' : total_rouge2.value(),
        'RougeL' : total_rougeL.value(),
    }


# scale down with total token (using InterDistinctMetric) as original paper did 
def get_distinct_result(preds_list):
    total_distinct1 = None
    total_distinct2 = None
    for idxs in range(len(preds_list)):
        cur_distinct1 = InterDistinctMetric.compute(preds_list[idxs], 1)
        cur_distinct2 = InterDistinctMetric.compute(preds_list[idxs], 2)

        # distinct1
        if total_distinct1 is None: 
            total_distinct1 = cur_distinct1
        else:
            total_distinct1 += cur_distinct1

        # distinct2
        if total_distinct2 is None: 
            total_distinct2 = cur_distinct2
        else:
            total_distinct2 += cur_distinct2

    return {
        'Distinct-1' : total_distinct1.value(),
        'Distinct-2' : total_distinct2.value(),
    }



# def get_distinct_result(preds_list):
#     total_distinct1 = 0.0
#     total_distinct2 = 0.0
#     for idxs in range(len(preds_list)):
#         cur_distinct1 = IntraDistinctMetric.compute(preds_list[idxs], 1).value()
#         cur_distinct2 = IntraDistinctMetric.compute(preds_list[idxs], 2).value()

#         # distinct1
#         total_distinct1 += cur_distinct1
#         total_distinct2 += cur_distinct2

#     return {
#         'Distinct-1' : total_distinct1/len(preds_list),
#         'Distinct-2' : total_distinct2/len(preds_list),
#     }