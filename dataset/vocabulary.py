import os
__PATH__ = os.path.abspath(os.path.dirname(__file__))

_PARLAI_PAD = '__null__'
_PARLAI_GO = '__start__'
_PARLAI_EOS = '__end__'
_PARLAI_UNK = '__unk__'
_PARLAI_START_VOCAB = [_PARLAI_PAD, _PARLAI_GO, _PARLAI_EOS, _PARLAI_UNK]
PARLAI_PAD_ID = 0
PARLAI_GO_ID = 1
PARLAI_EOS_ID = 2
PARLAI_UNK_ID = 3

_BERT_PAD = "[PAD]"
_BERT_UNK = "[UNK]"
_BERT_CLS = "[CLS]"
_BERT_SEP = "[SEP]"
_BERT_MASK = "[MASK]"
BERT_PAD_ID = 0
BERT_UNK_ID = 100
BERT_CLS_ID = 101
BERT_SEP_ID = 102
BERT_MASK_ID = 103