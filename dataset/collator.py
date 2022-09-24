import torch
import random
from operator import itemgetter


class WowCollator:
    iterator_shapes = {
        "context": 2,
        "response": 2,
        "chosen_topic": 2,
        "knowledge_sentences": 3,
        "episode_length": 0,
    }

    def __init__(self,
                 tokenizer,
                 token_preprocessed=False,
                 device='cpu',
                 max_length: int = 51,
                 max_episode_length: int = 5,
                 max_knowledge: int = 32,           # max knowledge sequence length
                 knowledge_truncate: int = 34,      # max knowledge number in candidates
                 ):            
        self.token_preprocessed = token_preprocessed
        self._max_length = max_length
        self._max_episode_length = max_episode_length
        self._max_knowledge = max_knowledge
        self._knowledge_truncate = knowledge_truncate

        self.device = device
        self.tokenizer = tokenizer

    # batch 'context'는 3차원 되어야함
    # batch 'response', 'chosen_topic' 마찬가지
    # batch 'knowledge_sentences'는 4차원 되어야 할듯
    def __call__(self, batch_tuple):
        batch = []
        collator_mode = []
        for b,cm in batch_tuple:
            batch.append(b)
            collator_mode.append(cm)
        collator_mode = list(set(collator_mode))[0]

        examples_batch_list = []
        batch_input_dict = {}

        max_episode_length = min(max([len(b) for b in batch]), self._max_episode_length)
        max_num_knowledge = self._knowledge_truncate if collator_mode=='train' else\
            max([max([len(batch_sequence['knowledge_sentences']) for batch_sequence in b]) for b in batch])

        for episode in batch:
            examples = {'context': [],
                        'response': [],
                        'chosen_topic': [],
                        'knowledge_sentences': []}
            for idx, example in enumerate(episode):
                if idx == self._max_episode_length:
                    break
                    
                examples['context'].append(example['context'])
                examples['response'].append(example['response'])
                examples['chosen_topic'].append(example['chosen_topic'])

                if self._knowledge_truncate > 0 and collator_mode == 'train':  # Do not truncate in test time
                    knowledge_sentences = example['knowledge_sentences']
                    num_knowledges = min(len(knowledge_sentences), self._knowledge_truncate)
                    keepers = list(range(1, len(knowledge_sentences)))
                    random.shuffle(keepers)
                    keepers = [0] + keepers[:num_knowledges-1]
                    sentences = itemgetter(*keepers)(knowledge_sentences)
                    examples['knowledge_sentences'].append('\n'.join(sentences))
                else:
                    knowledge_sentences = example['knowledge_sentences']
                    examples['knowledge_sentences'].append('\n'.join(knowledge_sentences))

            examples['episode_length'] = len(examples['context'])
            
            # parse(tokenize) per episode
            parsed_examples = self.parse_fn(examples, max_episode_length, max_num_knowledge)
            # append to batch list
            examples_batch_list.append(parsed_examples)
        
        # aggreate as tensor
        expanded_batch_key = sorted(list(examples_batch_list[0].keys()), reverse=True)
        for k in expanded_batch_key:
            batch_stacked = []
            for eb in examples_batch_list:
                popped_item = torch.tensor(eb[k]) if type(eb[k]) is int else eb[k]
                batch_stacked.append(popped_item)
            batch_input_dict[k] = torch.stack(batch_stacked, axis=0)

        # clip with max_length
        batch_max_context_length = torch.max(batch_input_dict['context_length']).item()
        batch_max_response_length = torch.max(batch_input_dict['response_length']).item()
        batch_max_chosen_topic_length = torch.max(batch_input_dict['chosen_topic_length']).item()
        batch_input_dict['context'] = batch_input_dict['context'][:,:,:batch_max_context_length].to(self.device)
        batch_input_dict['response'] = batch_input_dict['response'][:,:,:batch_max_response_length].to(self.device)
        batch_input_dict['chosen_topic'] = batch_input_dict['chosen_topic'][:,:,:batch_max_chosen_topic_length]
        
        batch_max_num_knowledge = torch.max(batch_input_dict['num_knowledge_sentences']).item()
        batch_max_knowledge_sentences_length = torch.max(batch_input_dict['knowledge_sentences_length']).item()
        batch_input_dict['knowledge_sentences'] = batch_input_dict['knowledge_sentences'][:,:,:batch_max_num_knowledge,:batch_max_knowledge_sentences_length].to(self.device)
        batch_input_dict['knowledge_sentences_length'] = batch_input_dict['knowledge_sentences_length'][:,:,:batch_max_num_knowledge]
        
        # flatten as batch_size * episode_length
        input_ids = batch_input_dict['context']
        attention_mask = (input_ids != self.tokenizer.pad_token_id).type(torch.DoubleTensor)
        decoder_input_ids = batch_input_dict['response'][:, :, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id).type(torch.DoubleTensor)
        labels_ = batch_input_dict['response'][:,:,1:]
        labels = labels_.masked_fill(labels_==self.tokenizer.pad_token_id, -100)
        knowledge_input_ids = batch_input_dict['knowledge_sentences']
        knowledge_attention_mask = (knowledge_input_ids != self.tokenizer.pad_token_id).type(torch.DoubleTensor)
        response = batch_input_dict['response']
        # remain batch_size shape [batch_size, episode_length, ...]
        context_length = batch_input_dict['context_length']
        response_length = batch_input_dict['response_length']
        num_knowledge_sentences = batch_input_dict['num_knowledge_sentences']
        knowledge_sentences_length = batch_input_dict['knowledge_sentences_length']

        # labels = batch_input_dict['response'][:,:,1:]
        # output_dict = {
        #     'input_ids' : batch_input_dict['context'].reshape(-1, batch_input_dict['context'].shape[-1]),
        #     'attention_mask' : (batch_input_dict['context'] != self.tokenizer.pad_token_id).type(torch.FloatTensor).reshape(-1, batch_input_dict['context'].shape[-1]),
        #     'decoder_input_ids' : batch_input_dict['response'][:, :, :-1].reshape(-1, batch_input_dict['response'].shape[-1]-1),
        #     'decoder_attention_mask' : (batch_input_dict['response'][:, :, :-1] != self.tokenizer.pad_token_id).type(torch.FloatTensor).reshape(-1, batch_input_dict['response'].shape[-1]-1),
        #     'labels' : labels.masked_fill(labels==self.tokenizer.pad_token_id, -100).reshape(-1, batch_input_dict['response'].shape[-1]-1)
        # }

        c_bs, c_el, max_context_length = input_ids.shape
        r_bs, r_el, max_dec_input_length = decoder_input_ids.shape

        output_dict = {
            'input_ids':input_ids.reshape(-1, max_context_length),
            'attention_mask':attention_mask.reshape(-1, max_context_length),
            'decoder_input_ids':decoder_input_ids.reshape(-1, max_dec_input_length),
            'decoder_attention_mask':decoder_attention_mask.reshape(-1, max_dec_input_length),
            'labels':labels.reshape(-1, max_dec_input_length),
            'knowledge_input_ids':knowledge_input_ids,
            'knowledge_attention_mask':knowledge_attention_mask,
            'response':response,
            'context_length':context_length,
            'response_length':response_length,
            'num_knowledge_sentences':num_knowledge_sentences,
            'knowledge_sentences_length':knowledge_sentences_length,
        }

        return output_dict
        
        
    def parse_fn(self, example, max_episode_length, max_num_knowledge):
        for key, value in self.iterator_shapes.items():
            dims = value
            if dims == 0: # which is scalar
                pass
            elif dims == 2: # which is matrix
                sentences, lengths = self.list_of_string_split(example[key], max_episode_length)
                example[key] = sentences
                example[f"{key}_length"] = lengths
            elif dims == 3: # which is tensor
                list_of_sentences, sentence_lengths, num_sentences = \
                    self.list_of_list_of_string_split(example[key], max_episode_length, max_num_knowledge)
                example[key] = list_of_sentences
                example[f"{key}_length"] = sentence_lengths
                example[f"num_{key}"] = num_sentences
            else:
                raise ValueError('Error in parse_fn')
        return example

    # padding 하는데 있어서 일단 model의 max_input_length 로 다 맞춘 다음에
    # max 고르고 이거를 slice하는게 훨씬 빠를듯
    def list_of_string_split(self,
                             line,
                             max_episode_length,
                             delimiter: str = ' ', 
                             padding="0"):
        # Must check data is already tokenized (self.token_preprocessed)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else padding
        if self.token_preprocessed:
            raise ValueError('not implemented yet')
        else:
            # attention_mask & token_type_id 같이 만들어 주지만 일단은 빼고 생각하기
            splitted_sentences = self.tokenizer.batch_encode_plus(line,
             return_tensors='pt',
             max_length=self._max_length,
             padding='max_length',
             truncation=True)['input_ids']
            # padd in here as max episode length
            padded_splitted_sentences = torch.nn.functional.pad(splitted_sentences, pad=(0,0,0,max_episode_length-splitted_sentences.shape[0]), value=pad_id)
            padded_sentence_lengths = torch.sum(padded_splitted_sentences != pad_id, axis=-1)

        return padded_splitted_sentences, padded_sentence_lengths
        
    def list_of_list_of_string_split(self,
                                     line,
                                     max_episode_length,
                                     max_num_knowledge,
                                     first_delimiter: str = '\n',
                                     second_delimiter: str = ' ',
                                     padding="0"):
        # Must check data is already tokenized (self.token_preprocessed)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else padding
        if self.token_preprocessed:
            raise ValueError('not implemented yet')
        else:
            splitted_sentences = [[ll for ll in l.split(first_delimiter)] for l in line]
            tokenized_sentences = [self.tokenizer.batch_encode_plus(ss,
                return_tensors='pt',
                max_length=self._max_knowledge,
                padding='max_length',
                truncation=True)['input_ids']
             for ss in splitted_sentences]
            
            # max_num_sentences = torch.max(torch.tensor([ts.shape[0] for ts in tokenized_sentences])).item()
            padded_sentences = torch.stack([torch.nn.functional.pad(ts, pad=(0,0,0,max_num_knowledge-ts.shape[0]), value=pad_id) for ts in tokenized_sentences], axis=0)
            padded_sentences = torch.nn.functional.pad(padded_sentences, pad=(0,0,0,0,0,max_episode_length-padded_sentences.shape[0]), value=pad_id)
            padded_sentence_lengths = torch.sum(padded_sentences != pad_id, axis=-1)
            padded_num_sentences = torch.sum(padded_sentence_lengths != pad_id, axis=-1)
            
        return padded_sentences, padded_sentence_lengths, padded_num_sentences



if __name__ == "__main__":
    from transformers import BertTokenizer
    from wizard_of_wikipedia_dataset import WowTorchDataset
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    wow_train = WowTorchDataset(tokenizer=None,
     data_path='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache/wizard_of_wikipedia',
     cache_dir='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache',
     mode='train',
     token_preprocessed=False)
    wc = WowCollator(tokenizer=tokenizer, mode='valid', device='cuda')

    # a = wc([wow_train[0], wow_train[13]])
    # a

    print('im here')
    dl = DataLoader(wow_train, batch_size=4, shuffle=True, collate_fn=wc)
    for a in tqdm(dl):
        pass

    print('done')