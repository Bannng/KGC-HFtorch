from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import json
import os
import colorlog
from tqdm import tqdm
from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import create_task

import dataset.vocabulary as data_vocab

from transformers import BertTokenizer, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import random

PARLAI_KNOWLEDGE_SEPARATOR = '__knowledge__'
BERT_KNOWLEDGE_SEPARATOR = '_ _ knowledge _ _'


class WowEvalTorchDataset(Dataset):
    def __init__(self,
                 tokenizer=None,
                 data_path='cache/wizard_of_wikipedia',
                 cache_dir='./cache',
                 seen_mode='valid',
                 unseen_mode='valid_unseen',
                 load_dynamic=True,
                 token_preprocessed=False,
                 ) -> None:
        self.seen_data = WowTorchDataset(tokenizer=tokenizer,
                                         data_path=data_path,
                                         cache_dir=cache_dir,
                                         mode=seen_mode,
                                         load_dynamic=load_dynamic,
                                         token_preprocessed=token_preprocessed)
        self.unseen_data = WowTorchDataset(tokenizer=tokenizer,
                                           data_path=data_path,
                                           cache_dir=cache_dir,
                                           mode=unseen_mode,
                                           load_dynamic=load_dynamic,
                                           token_preprocessed=token_preprocessed)

        self.total_num_episodes = self.seen_data.num_episodes + self.unseen_data.num_episodes
        self.total_num_examples = self.seen_data.num_examples + self.unseen_data.num_examples
        self.total_episodes = self.seen_data.episodes + self.unseen_data.episodes

    def __len__(self):
        return self.total_num_episodes


    def __getitem__(self, idx):
        # 여기 짜는 중인데 idx 가 seen 이상인게 들어오면 unseen return 하도록 짝지어줘야 할 듯
        # idx 기반으로 return 할 self.mode가 valid_seen인지 valid_unseen 인지 보내줘야
        # collator에서 이걸 받아서 처리하고, 결국 collator return으로 뭐가 뭔지 알아야 metric 계산할때 활용 가능

        picked_episode = self.total_episodes[idx]
        example = json.loads(picked_episode) if self.load_dynamic and type(picked_episode) is str else picked_episode
        return example, self.mode





class WowTorchDataset(Dataset):
    def __init__(self,
                 tokenizer=None,
                 data_path='cache/wizard_of_wikipedia',
                 cache_dir='./cache',
                 mode='train',
                 load_dynamic=True,
                 token_preprocessed=False,
                 ) -> None:
        self._datapath = data_path # load wow dataset path
        self._cache_dir = cache_dir # for download parlai wow dataset
        self.mode = mode
        self.load_dynamic = load_dynamic
        self.token_preprocessed = token_preprocessed

        # set HF tokenizer
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

        self.episodes = self.load(self.mode, self.load_dynamic)
        self.num_episodes = len(self.episodes)
        self.num_examples = sum([len(episode) for episode in self.episodes])


    def __len__(self):
        return self.num_episodes


    def __getitem__(self, idx):
        picked_episode = self.episodes[idx]
        example = json.loads(picked_episode) if self.load_dynamic and type(picked_episode) is str else picked_episode
        return example, self.mode


    def load(self, mode : str, load_dynamic=True):
        # if there's no string form episode 
        episode_fname = self._get_preprocessed_fname(mode, self.token_preprocessed)
        if os.path.exists(episode_fname):
            colorlog.info(f"Load WoW Episode from {episode_fname}")
            with open(episode_fname, 'r') as fp:
                episodes = []
                for line in fp:
                    if load_dynamic:
                        episodes.append(line)
                    else:
                        episodes.append(json.loads(line))
            return episodes
        else:
            episodes = self.load_wow_fromparlai(mode)
            new_episodes = self.save_wow_episodes(episodes, self.tokenizer, mode, self.token_preprocessed)
            return new_episodes


    # download or cache wow data from parlai
    def load_wow_fromparlai(self, mode):
        """
        As default, it returns the following action dict:
        {
            'id': 'wizard_of_wikipedia'
            'text': chosen_topic\n # if first example in episode
                    last_apprentice_message\n # if possible
                    wizard_message # if --label-type is 'chosen_sent'
            'knowledge': title_1 sentence_1\n
                                .
                                .
                                .
                         title_m sentence_n # all knowledge available to wizard
            'labels': [title_checked sentence_checked] # default
                                    OR
                      [wizard_response] # if --label-type set to 'response'
            'label_candidates': knowledge + [no_passages_used no_passages_used]
                                           OR
                                100 response candidates  # if 'validation' or 'test'
            'chosen_topic': chosen_topic as untokenized string
            'checked_sentence': checked sentence if wizard, else None # if --include_checked_sentence
            'title': title of checked sentence # if --include_checked_sentence
            --> if not exists, then checked_sentence = title = 'no_passages_used'
            'episode_done': (Boolean) whether episode is done or not
        }
        """
        parlai_opt = self._get_parlai_opt([
            '--task', 'wizard_of_wikipedia:generator:topic_split' if 'unseen' in mode else 'wizard_of_wikipedia:generator:random_split',
            '--datatype', '{}:stream'.format(mode.split('_')[0]) if 'unseen' in mode else f'{mode}:stream',  # 'train' for shuffled data and 'train:stream' for unshuffled data
            '--datapath', self._cache_dir,
            # dict_XXX will not be used if we use bert tokenizer
            '--dict_lower', 'True',
            '--dict_tokenizer', 'bpe',
            '--dict_file', f"{self._cache_dir}/wow.dict",
            '--dict_textfields', "text,labels,chosen_topic,checked_sentence,knowledge,title",  # For retrieval mode, use "text,labels"
            # By following author's code. For retrieval mode, use 250004
            # Also, note that this is the size of bpehelper dictionary.
            # So, final dictionary can be larger than this one
            # And, don't convert special tokens to index with txt2vec method, you must use tok2ind
            '--dict_maxtokens', '30000',
            '--dict_nulltoken', data_vocab._PARLAI_PAD,
            '--dict_starttoken',data_vocab._PARLAI_GO,
            '--dict_endtoken', data_vocab._PARLAI_EOS,
            '--dict_unktoken', data_vocab._PARLAI_UNK,
            '--include_knowledge_separator', 'True',  # include speical __knowledge__ token between title and passage
            '--include_checked_sentence', 'True',
            '--label_type', 'response', # choices = ['response', 'chosen_sent']
        ])
        # As a default, world use "WizardDialogKnowledgeTeacher"
        agent = DictionaryAgent(parlai_opt)
        world = create_task(parlai_opt, agent)
        num_examples = world.num_examples()
        num_episodes = world.num_episodes()

        episodes = []
        for _ in range(num_episodes):
            examples = []
            while True:
                world.parley()
                example = world.acts[0]
                examples.append(example)
                if world.episode_done():
                    episodes.append(examples)
                    break
        return episodes


    def save_wow_episodes(self, episodes, dictionary, mode, token_preprocessed=False):
        """
        Save loaded Wizard-of-Wikipedia episodes into String/Tokenized format
        """
        saving_flag_str = 'tokenized' if token_preprocessed is True else 'string'
        colorlog.info(f"Saving loaded wizard of wikipedia into {saving_flag_str} form")

        if token_preprocessed:
            tokenize = lambda x : ' '.join([str(xx) for xx in dictionary.encode(x)])
        else:
            tokenize = lambda x : x

        new_episodes = []
        for episode_num, episode in enumerate(tqdm(episodes, ncols=70)):
            new_examples = []
            for example_num, example in enumerate(episode):
                # Tokenize inputs and convert to tokens
                context = tokenize(example['text'])
                if mode == "train":
                    response = tokenize(example['labels'][0])
                else:
                    response = tokenize(example['eval_labels'][0])
                chosen_topic = tokenize(example['chosen_topic'])

                # Set up knowledge
                checked_knowledge = example['title'] + ' __knowledge__ ' + example['checked_sentence']
                knowledges = [checked_knowledge] + \
                    [k for k in example['knowledge'].rstrip().split('\n')]
                for idx, k in enumerate(knowledges[1:]):
                    if k == checked_knowledge:
                        break
                else:
                    # Sometimes, knowledge does not include checked_sentnece
                    idx = None
                    colorlog.warning("Knowledge does not include checked sentence.")
                if idx is not None:
                    del knowledges[idx + 1]

                # Tokenize knowledge
                knowledge_sentences = [tokenize(k) for k in knowledges]

                new_example = {'context': context,
                               'response': response,
                               'chosen_topic': chosen_topic,
                               'knowledge_sentences': knowledge_sentences,
                               'episode_num': episode_num,
                               'example_num': example_num}
                new_examples.append(new_example)
            new_episodes.append(new_examples)

        if self._datapath:
            episodes_fname = self._get_preprocessed_fname(mode, token_preprocessed)
            colorlog.info(f"Cache preprocessed dataset to {episodes_fname}")
            with open(episodes_fname, 'w') as fp:
                for episode in new_episodes:
                # for episode in new_episodes[:50]: 
                    fp.write(json.dumps(episode) + '\n')

        return new_episodes


    def _get_parlai_opt(self, options: List[str] = [], print_args=False):
        from parlai.scripts.build_dict import setup_args
        parser = setup_args()
        opt = parser.parse_args(options, print_args=print_args)
        return opt


    def _get_preprocessed_fname(self, mode, token_preprocessed=False):
        if self._datapath:
            if token_preprocessed:
                return os.path.join(self._datapath, f'tokenized_{mode}_episodes.json')
            return os.path.join(self._datapath, f'{mode}_episodes.json')
        else:
            return None



if __name__ == "__main__":
    wow_train = WowTorchDataset(tokenizer=None,
     data_path='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache/wizard_of_wikipedia',
     cache_dir='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache',
     mode='train',
     token_preprocessed=False)

    wow_valid = WowTorchDataset(tokenizer=None,
     data_path='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache/wizard_of_wikipedia',
     cache_dir='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache',
     mode='valid',
     token_preprocessed=False)
    
    wow_valid = WowTorchDataset(tokenizer=None,
     data_path='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache/wizard_of_wikipedia',
     cache_dir='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache',
     mode='valid_unseen',
     token_preprocessed=False)

    wow_test = WowTorchDataset(tokenizer=None,
     data_path='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache/wizard_of_wikipedia',
     cache_dir='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache',
     mode='test',
     token_preprocessed=False)
    
    wow_valid = WowTorchDataset(tokenizer=None,
     data_path='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache/wizard_of_wikipedia',
     cache_dir='/home/byeongjoo/KGC/SequentialKnowledgeTransformer-torch/cache',
     mode='test_unseen',
     token_preprocessed=False)

    print('done')
    