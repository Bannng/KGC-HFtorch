from typing import Dict, Optional, List, Tuple, Union
import torch
from transformers import BertTokenizer, BertModel, EncoderDecoderModel, BertConfig, EncoderDecoderConfig
from modules.knowledge_encdec_model import KnowledgeEncoderDecoderModel


# Transformer without knowledge selection
class Transformer(KnowledgeEncoderDecoderModel):
    def __init__(self,
        encoder_config='bert-base-uncased'
    ):
        # use bert tokenizer as default (to match perplexity)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        bert_base_uncased_config = BertConfig.from_pretrained(encoder_config)
        transformer_encoder_config = {
            "hidden_dropout_prob": 0.1,
            "intermediate_size":768,
            "hidden_size":768,
            "num_hidden_layers":5,
            "num_attention_heads":2,
            'hidden_act':"relu",
        }
        bert_base_uncased_config.update(transformer_encoder_config)

        transformer_encoder = BertModel(config=bert_base_uncased_config)
        self.encoder_config_ = transformer_encoder.config

        default_decoder_config = {
            'is_decoder':True,
            'add_cross_attention':True,
            "hidden_dropout_prob": 0.1,
            "intermediate_size":768,
            "hidden_size":768,
            "num_hidden_layers":5,
            "num_attention_heads":2,
            'hidden_act':"relu",
            'bos_token_id':self.tokenizer.cls_token_id,
            'eos_token_id':self.tokenizer.sep_token_id,
            'decoder_start_token_id':self.tokenizer.cls_token_id,
            'num_beams':1,
            'no_repeat_ngram_size':0,
            'max_length':51,
        }
        self.decoder_config_ = BertConfig(**default_decoder_config)
        self.encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=self.encoder_config_, decoder_config=self.decoder_config_)
        super().__init__(config=self.encoder_decoder_config, encoder=transformer_encoder)
        
        self.config.bos_token_id = self.tokenizer.cls_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.sep_token_id
        self.config.num_beams = 1
        self.config.max_length = 51
        self.config.no_repeat_ngram_size = 0

        self.ignore_kwargs_key = ['knowledge_input_ids', 'knowledge_attention_mask', 'response',
                                  'context_length', 'response_length', 'num_knowledge_sentences',
                                  'knowledge_sentences_length']


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):  
        #######################################################################################
        # flatten episode batch
        flatten_main_batch = self.flat_episode_batch_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

        input_ids = flatten_main_batch['input_ids']
        attention_mask = flatten_main_batch['attention_mask']
        decoder_input_ids = flatten_main_batch['decoder_input_ids']
        decoder_attention_mask = flatten_main_batch['decoder_attention_mask']
        labels = flatten_main_batch['labels']
        #######################################################################################

        filtered_kwargs = kwargs
        if self.ignore_kwargs_key is not None:
            filtered_kwargs = {argument: value for argument, value in kwargs.items()
                if not argument in self.ignore_kwargs_key}

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **filtered_kwargs
        )