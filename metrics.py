from rouge import Rouge


def compute_rouge_scores(pred_seq, target_seq):
    rouge = Rouge()
    pred_seq_str = ' '.join([str(x) for x in pred_seq])
    target_seq = ' '.join([str(x) for x in target_seq])

    scores = rouge.get_scores(pred_seq_str, target_seq)

    return scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']
