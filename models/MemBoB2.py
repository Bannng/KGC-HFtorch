from typing import Dict, Optional, List, Tuple, Union
from matplotlib.style import context
import torch as torch
import torch.nn as nn
import numpy as np

from transformers import BertTokenizer, BertModel, EncoderDecoderModel, BertConfig, EncoderDecoderConfig
from transformers.modeling_outputs import BaseModelOutput

from modules.universal_sentence_embedding import universal_sentence_embedding
from modules.knowledge_encdec_model import KnowledgeEncoderDecoderModel
from utils.model_outputs import AdditionalSeq2SeqLMOutputWithKnow, AdditionalKnowledgeModelOutput
from utils.utils import neginf, shift_tokens_right

from modules.bert import MyBertLMHeadModel




# TransformerMemoryNetwork with Bert-base-uncased encoder
class MemNetBoB(KnowledgeEncoderDecoderModel):
    def __init__(self,
        encoder_config='bert-base-uncased',
        use_cs_ids=False,
        knowledge_alpha=0.25,
        knowledge_mode="1",
    ):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        knowledge_encoder = BoBTMemNetBert(encoder_config,
            use_cs_ids=use_cs_ids,
            knowledge_alpha=knowledge_alpha,
            knowledge_mode=knowledge_mode
        )
        self.knowledge_mode = knowledge_mode
        self.encoder_config_ = knowledge_encoder.encoder.config

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
        super().__init__(config=self.encoder_decoder_config, encoder=knowledge_encoder)

        self.use_cs_ids = use_cs_ids
        self.knowledge_alpha = knowledge_alpha

        self.config.bos_token_id = self.tokenizer.cls_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.sep_token_id
        self.config.num_beams = 1
        self.config.max_length = 51
        self.config.no_repeat_ngram_size = 0
    
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
        knowledge_input_ids = None,
        knowledge_attention_mask = None,
        response = None,
        context_length = None,
        response_length = None,
        num_knowledge_sentences = None,
        knowledge_sentences_length = None,
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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
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
                knowledge_input_ids=knowledge_input_ids,
                knowledge_attention_mask=knowledge_attention_mask,
                response=response,
                context_length=context_length,
                response_length=response_length,
                num_knowledge_sentences=num_knowledge_sentences,
                knowledge_sentences_length=knowledge_sentences_length,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)


        ##########################################################
        # unpack encoder outputs
        encoder_hidden_states = encoder_outputs.last_hidden_state
        attention_mask = encoder_outputs.extended_attention_mask
        knowledge_attention = encoder_outputs.knowledge_attention
        knowledge_loss = encoder_outputs.additional_losses[0] if type(encoder_outputs.additional_losses) is tuple else encoder_outputs.additional_losses
        correct_know_num = encoder_outputs.correct_know_nums
        ##########################################################


        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            # warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))


        ########################################################
        # loss aggregation step
        total_loss = None
        token_loss = torch.zeros(1)
        know_loss = knowledge_loss.detach()
        correct_know_num = correct_know_num
        
        if loss is not None:
            total_loss = loss
            token_loss = loss.detach()

        if total_loss is not None and knowledge_loss is not None:
            total_loss += self.knowledge_alpha * knowledge_loss
        ###########################################################


        if not return_dict:
            if total_loss is not None:
                return (total_loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return AdditionalSeq2SeqLMOutputWithKnow(
            loss=total_loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            additional_losses=(token_loss, know_loss),
            correct_know_nums=correct_know_num,
        )




# TransformerMemoryNetwork with Bert-base-uncased encoder
class BoBTMemNetBert2(KnowledgeEncoderDecoderModel):
    def __init__(self,
        encoder_config='bert-base-uncased',
        use_cs_ids=False,
        knowledge_alpha=0.25,
        knowledge_mode="1",
        concat_query=False,
    ):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        knowledge_encoder = BoBContextKnowledgeEncoder.from_pretrained(encoder_config,
            use_cs_ids=use_cs_ids,
            knowledge_alpha=knowledge_alpha,
            knowledge_mode=knowledge_mode
        )
        self.knowledge_mode = knowledge_mode
        self.encoder_config_ = knowledge_encoder.config

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
        my_decoder = MyBertLMHeadModel(self.decoder_config_)
        self.encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=self.encoder_config_, decoder_config=self.decoder_config_)
        super().__init__(config=self.encoder_decoder_config, encoder=knowledge_encoder, decoder=my_decoder)

        self.use_cs_ids = use_cs_ids
        self.knowledge_alpha = knowledge_alpha

        self.config.bos_token_id = self.tokenizer.cls_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.sep_token_id
        self.config.num_beams = 1
        self.config.max_length = 51
        self.config.no_repeat_ngram_size = 0

        self.concat_query = concat_query
        self.context_response_query_linear = torch.nn.Linear(768*2,768)
    
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
        knowledge_input_ids = None,
        knowledge_attention_mask = None,
        response = None,
        context_length = None,
        response_length = None,
        num_knowledge_sentences = None,
        knowledge_sentences_length = None,
        generation_mode = False,
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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                knowledge_input_ids=knowledge_input_ids,
                knowledge_attention_mask=knowledge_attention_mask,
                response=response,
                context_length=context_length,
                response_length=response_length,
                num_knowledge_sentences=num_knowledge_sentences,
                knowledge_sentences_length=knowledge_sentences_length,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)


        ##########################################################
        # unpack encoder outputs
        encoder_hidden_states = encoder_outputs.last_hidden_state
        attention_mask = encoder_outputs.extended_attention_mask
        knowledge_attention = encoder_outputs.knowledge_attention
        knowledge_loss = encoder_outputs.additional_losses[0] if type(encoder_outputs.additional_losses) is tuple else encoder_outputs.additional_losses
        correct_know_num = encoder_outputs.correct_know_nums

        correct_know_num = torch.tensor(0., device=encoder_hidden_states.device) if correct_know_num is None else correct_know_num
        knowledge_loss = torch.tensor(0., device=encoder_hidden_states.device) if knowledge_loss is None else knowledge_loss

        knowledge_pooled, knowledge_encoder_outputs,\
         context_encoder_output, context_attention_mask = encoder_outputs.hidden_states
        ##########################################################


        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            # warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            logits = logits[:, :labels.shape[-1], :]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        #############################################################################################
        p_knowledge_attention = None
        # decoder_output_sequence = decoder_outputs.hidden_states[-1]
        decoder_output_sequence = decoder_outputs.hidden_states
        if not self.training:
            gen_result = torch.argmax(decoder_outputs.logits, dim=-1)
            # make decoder attention mask based on eos_token_id
            eos_tokens = [np.where(g==self.tokenizer.sep_token_id)[0] for g in gen_result.cpu().numpy()]
            eos_tokens = [et.item() if len(et) != 0 else gen_result.shape[-1] for et in eos_tokens]
            eos_tokens = torch.tensor(eos_tokens).unsqueeze(-1).repeat(1, gen_result.shape[-1])
            range_tokens = torch.arange(gen_result.shape[-1]).repeat(gen_result.shape[0], 1)
            decoder_attention_mask = (range_tokens < eos_tokens).type(torch.DoubleTensor).to(decoder_output_sequence.device)

        p_encoder_outputs, p_attention_mask, p_knowledge_attention = self.knowledge_selection_decoder(
                                                                decoder_output_sequence,
                                                                decoder_attention_mask,
                                                                knowledge_pooled,
                                                                knowledge_encoder_outputs,
                                                                context_encoder_output,
                                                                context_attention_mask,
                                                                knowledge_attention_mask,
                                                                knowledge_sentences_length,
                                                                knowledge_input_ids.shape,
                                                                self.use_cs_ids,
                                                                )
        # for knowledge loss
        if self.knowledge_alpha != 0.0:
            p_know_loss, p_correct_know_num = self.get_p_knowledge_loss(
                p_knowledge_attention,
                num_knowledge_sentences,
            )
        knowledge_loss += p_know_loss
        correct_know_num = p_correct_know_num
        

        ########################################################
        # loss aggregation step
        total_loss = None
        token_loss = torch.zeros(1)
        know_loss = knowledge_loss.detach()
        correct_know_num = correct_know_num
        
        if loss is not None:
            total_loss = loss
            token_loss = loss.detach()

        if total_loss is not None and knowledge_loss is not None:
            total_loss += self.knowledge_alpha * knowledge_loss
        ###########################################################


        if not return_dict:
            if total_loss is not None:
                return (total_loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return AdditionalSeq2SeqLMOutputWithKnow(
            loss=total_loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            additional_losses=(token_loss, know_loss),
            correct_know_nums=correct_know_num,
        )


    def knowledge_selection_decoder(
        self,
        decoder_output_sequence,
        decoder_attention_mask,
        knowledge_pooled,
        knowledge_encoder_output,
        context_encoder_output,
        context_attention_mask,
        knowledge_attention_mask=None,
        knowledge_sentences_length=None,
        knowledge_shape=None,
        use_cs_ids=False,
        decoder_pool_mode="use",
    ):

        know_use = knowledge_pooled
        k_bs, k_el, nk, kl = knowledge_shape
        embed_dim = know_use.shape[-1]
        knowledge_attention_mask = knowledge_attention_mask.reshape(-1, kl)

        if decoder_pool_mode=="use":
            # decoder_attention_mask.sum(-1).to(torch.int64)
            posterior_use = universal_sentence_embedding(decoder_output_sequence, decoder_attention_mask)

            if self.concat_query:
                context_use = universal_sentence_embedding(context_encoder_output, context_attention_mask)
                posterior_use = self.context_response_query_linear(torch.concat([posterior_use, context_use], dim=-1))

            know_use = know_use.reshape(k_bs*k_el, nk, embed_dim)
            posterior_use /= np.sqrt(embed_dim)
            know_use /= np.sqrt(embed_dim) # 여기서 backward 에러 생김

            ck_attn = torch.bmm(know_use, posterior_use.unsqueeze(-1)).squeeze(-1)
            # fill with near -inf
            ck_mask = (knowledge_sentences_length.reshape(-1,nk) != 0).to(posterior_use.device)
            ck_attn = ck_attn.masked_fill(~ck_mask, neginf(context_encoder_output.dtype))

            if not use_cs_ids:
                # if we're not given the true chosen_sentence (test time), pick our
                # best guess
                _, cs_ids = ck_attn.max(1)
            else:
                cs_ids = torch.zeros_like(ck_attn.max(1)[1])

            # pick the true chosen sentence. remember that TransformerEncoder outputs
            #   (batch, time, embed)
            # but because know_encoded is a flattened, it's really
            #   (N * K, T, D)
            # We need to compute the offsets of the chosen_sentences
            cs_offsets = torch.arange(k_bs*k_el, device=cs_ids.device) * nk + cs_ids
            cs_encoded = knowledge_encoder_output[cs_offsets]
            # but padding is (N * K, T)
            cs_mask = knowledge_attention_mask[cs_offsets]
        
        else:
            pass

        # finally, concatenate it all
        full_enc = torch.cat([cs_encoded, context_encoder_output], dim=1)
        full_mask = torch.cat([cs_mask, context_attention_mask], dim=1)

        # also return the knowledge selection mask for the loss
        return full_enc, full_mask, ck_attn


    def get_p_knowledge_loss(
        self,
        knowledge_attention,
        num_knowledge_sentences,
    ):
        if knowledge_attention is None:
            return torch.tensor(0., device=num_knowledge_sentences.device), torch.tensor(0, device=num_knowledge_sentences.device)

        num_know = num_knowledge_sentences.view(-1)

        # arg max prediction
        _, know_pred = knowledge_attention.max(1)
        knowledge_gold_ids = torch.zeros_like(know_pred).view(-1)
        masked_knowledge_gold_ids = knowledge_gold_ids.masked_fill(num_know.eq(0), -100)

        # get knowledge loss
        know_loss = torch.nn.functional.cross_entropy(
            knowledge_attention,
            masked_knowledge_gold_ids,
            reduction='mean',
            ignore_index=-100,
        )

        # get correct num
        correct_know_num = (know_pred == masked_knowledge_gold_ids).float().sum()
        
        return know_loss, correct_know_num


    def get_output_embeddings(self):
        return None



    def generate(self, **kwargs):
        kwargs.pop('max_length')
        kwargs.pop('do_sample')
        kwargs.pop('num_beams')
        kwargs.pop('no_repeat_ngram_size')
        kwargs.pop('synced_gpus')
        kwargs['decoder_input_ids'] = kwargs['input_ids']
        with torch.no_grad():
            outputs = self(**kwargs)
        gen_result = torch.argmax(outputs.logits, dim=-1)
        return gen_result


    # def prepare_inputs_for_generation(
    #     self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    # ):
    #     decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
    #     decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
    #     input_dict = {
    #         "attention_mask": attention_mask,
    #         "decoder_attention_mask": decoder_attention_mask,
    #         "decoder_input_ids": decoder_inputs["input_ids"],
    #         "encoder_outputs": encoder_outputs,
    #         "past_key_values": decoder_inputs["past_key_values"],
    #         "use_cache": False,
    #         "knowledge_input_ids":kwargs['knowledge_input_ids'],
    #         "knowledge_attention_mask":kwargs['knowledge_attention_mask'],
    #         "response":kwargs['response'],
    #         "context_length":kwargs['context_length'],
    #         "response_length":kwargs['response_length'],
    #         "num_knowledge_sentences":kwargs['num_knowledge_sentences'],
    #         "knowledge_sentences_length":kwargs['knowledge_sentences_length'],
    #         "generation_mode":True,
    #     }
    #     return input_dict







# # TransformerMemoryNetwork just using bert vocab (not pretrained, just tokenization)
# class TMemNet(TMemNetBert):
#     def __init__(self,
#         encoder_config='bert-base-uncased',
#         use_cs_ids=False,
#         knowledge_alpha=0.25
#     ):
#         # use bert tokenizer as default (to match perplexity)
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#         bert_base_uncased_config = BertConfig.from_pretrained(encoder_config)
#         transformer_encoder_config = {
#             "hidden_dropout_prob": 0.1,
#             "intermediate_size":768,
#             "hidden_size":768,
#             "num_hidden_layers":5,
#             "num_attention_heads":2,
#             'hidden_act':"relu",
#         }
#         bert_base_uncased_config.update(transformer_encoder_config)

#         knowledge_encoder = ContextKnowledgeEncoder(
#             config=bert_base_uncased_config,
#             use_cs_ids=use_cs_ids,
#             knowledge_alpha=knowledge_alpha,
#         )
#         self.encoder_config_ = knowledge_encoder.config

#         default_decoder_config = {
#             'is_decoder':True,
#             'add_cross_attention':True,
#             "hidden_dropout_prob": 0.1,
#             "intermediate_size":768,
#             "hidden_size":768,
#             "num_hidden_layers":5,
#             "num_attention_heads":2,
#             'hidden_act':"relu",
#             'bos_token_id':self.tokenizer.cls_token_id,
#             'eos_token_id':self.tokenizer.sep_token_id,
#             'decoder_start_token_id':self.tokenizer.cls_token_id,
#             'num_beams':1,
#             'no_repeat_ngram_size':0,
#             'max_length':51,
#         }
#         self.decoder_config_ = BertConfig(**default_decoder_config)
#         self.encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=self.encoder_config_, decoder_config=self.decoder_config_)
#         super(TMemNetBert, self).__init__(config=self.encoder_decoder_config, encoder=knowledge_encoder)
        
#         self.use_cs_ids = use_cs_ids
#         self.knowledge_alpha = knowledge_alpha

#         self.config.bos_token_id = self.tokenizer.cls_token_id
#         self.config.pad_token_id = self.tokenizer.pad_token_id
#         self.config.eos_token_id = self.tokenizer.sep_token_id
#         self.config.num_beams = 1
#         self.config.max_length = 51
#         self.config.no_repeat_ngram_size = 0



# Encoder with Knowledge Selection in MemNet
class BoBContextKnowledgeEncoder(BertModel):
    def __init__(self,
        config,
        add_pooling_layer=True,
        use_cs_ids=False,
        knowledge_alpha=0.25,
        knowledge_mode="1",
    ):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        
        self.use_cs_ids = use_cs_ids
        self.knowledge_alpha = knowledge_alpha
        self.knowledge_mode = knowledge_mode

    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        knowledge_input_ids = None,
        knowledge_attention_mask = None,
        response = None,
        context_length = None,
        response_length = None,
        num_knowledge_sentences = None,
        knowledge_sentences_length = None,
    ):
        #######################################################################################
        # flatten episode batch
        if input_ids is not None and len(input_ids.shape) > 2:
            c_bs, c_el, max_context_length = input_ids.shape
            input_ids = input_ids.reshape(-1, max_context_length)

        if attention_mask is not None and len(attention_mask.shape) > 2:
            c_bs, c_el, max_context_length = attention_mask.shape
            attention_mask = attention_mask.reshape(-1, max_context_length)
        #######################################################################################

        # original arguments shape are flatten, but added arguments are not!
        k_bs, k_el, nk, kl = knowledge_input_ids.shape
        knowledge_input_ids = knowledge_input_ids.reshape(-1, kl)
        knowledge_attention_mask = knowledge_attention_mask.reshape(-1, kl)

        # get context encoder output
        context_encoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask= attention_mask,
        ).last_hidden_state

        # get knowledge encoder output
        knowledge_encoder_outputs = super().forward(
            input_ids=knowledge_input_ids,
            attention_mask= knowledge_attention_mask,
        ).last_hidden_state

        # get selected knowledge-context encoder output and mask
        encoder_outputs, extended_attention_mask, knowledge_attention, knowledge_pooled = self.get_knowledge_selection_encoder_output(
            context_encoder_outputs,
            attention_mask,
            knowledge_encoder_outputs,
            knowledge_attention_mask,
            knowledge_sentences_length,
            (k_bs, k_el, nk, kl),
            self.use_cs_ids,
            self.knowledge_mode,
        )
        # encoder_outputs = BaseModelOutput(encoder_outputs)
        

        # for knowledge loss
        if self.knowledge_alpha != 0.0:
            know_loss, correct_know_num = self.get_knowledge_loss(
                knowledge_attention,
                num_knowledge_sentences,
            )
        else:
            know_loss = None
            correct_know_num = None

        if self.knowledge_mode == "argmax":
            # get knowledge encoder output
            knowledge_encoder_outputs = super().forward(
                input_ids=knowledge_input_ids,
                attention_mask= knowledge_attention_mask,
            ).last_hidden_state
            knowledge_pooled = universal_sentence_embedding(knowledge_encoder_outputs, knowledge_attention_mask)




        return AdditionalKnowledgeModelOutput(
            last_hidden_state=encoder_outputs,
            extended_attention_mask=extended_attention_mask,
            knowledge_attention=knowledge_attention,
            additional_losses=(know_loss),
            correct_know_nums=correct_know_num,
            hidden_states=(knowledge_pooled, knowledge_encoder_outputs, context_encoder_outputs, attention_mask)
        )

    def get_knowledge_selection_encoder_output(
        self,
        context_encoder_output,
        context_attention_mask,
        knowledge_encoder_output,
        knowledge_attention_mask,
        knowledge_sentences_length,
        knowledge_shape,
        use_cs_ids=False,
        knowledge_mode="argmax",
    ):

        know_use = universal_sentence_embedding(knowledge_encoder_output, knowledge_attention_mask)
        k_bs, k_el, nk, kl = knowledge_shape
        embed_dim = know_use.shape[-1]

        if knowledge_mode=="context_only":
            return context_encoder_output, context_attention_mask, None, know_use

        if knowledge_mode == "argmax":
            context_use = universal_sentence_embedding(context_encoder_output, context_attention_mask)
            know_use = know_use.reshape(k_bs*k_el, nk, embed_dim)
            context_use /= np.sqrt(embed_dim)
            know_use /= np.sqrt(embed_dim)

            ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
            # fill with near -inf
            ck_mask = (knowledge_sentences_length.reshape(-1,nk) != 0).to(context_use.device)
            ck_attn = ck_attn.masked_fill(~ck_mask, neginf(context_encoder_output.dtype))

            if not use_cs_ids:
                # if we're not given the true chosen_sentence (test time), pick our
                # best guess
                _, cs_ids = ck_attn.max(1)
            else:
                cs_ids = torch.zeros_like(ck_attn.max(1)[1])

            # pick the true chosen sentence. remember that TransformerEncoder outputs
            #   (batch, time, embed)
            # but because know_encoded is a flattened, it's really
            #   (N * K, T, D)
            # We need to compute the offsets of the chosen_sentences
            cs_offsets = torch.arange(k_bs*k_el, device=cs_ids.device) * nk + cs_ids
            cs_encoded = knowledge_encoder_output[cs_offsets]
            # but padding is (N * K, T)
            cs_mask = knowledge_attention_mask[cs_offsets]
        
        if knowledge_mode=="pool":
            cs_encoded = know_use.reshape(k_bs*k_el, nk, embed_dim)
            cs_mask = (knowledge_attention_mask.sum(-1).reshape(k_bs*k_el, nk) != 0).type(torch.DoubleTensor).to(cs_encoded.device)
            ck_attn = None

        # finally, concatenate it all
        full_enc = torch.cat([cs_encoded, context_encoder_output], dim=1)
        full_mask = torch.cat([cs_mask, context_attention_mask], dim=1)

        # also return the knowledge selection mask for the loss
        return full_enc, full_mask, ck_attn, know_use

    def get_knowledge_loss(
        self,
        knowledge_attention,
        num_knowledge_sentences,
    ):
        if knowledge_attention is None:
            return None, None

        num_know = num_knowledge_sentences.view(-1)

        # arg max prediction
        _, know_pred = knowledge_attention.max(1)
        knowledge_gold_ids = torch.zeros_like(know_pred).view(-1)
        masked_knowledge_gold_ids = knowledge_gold_ids.masked_fill(num_know.eq(0), -100)

        # get knowledge loss
        know_loss = torch.nn.functional.cross_entropy(
            knowledge_attention,
            masked_knowledge_gold_ids,
            reduction='mean',
            ignore_index=-100,
        )

        # get correct num
        correct_know_num = (know_pred == masked_knowledge_gold_ids).float().sum()
        
        return know_loss, correct_know_num
