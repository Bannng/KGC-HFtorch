from typing import Dict, Optional, List, Tuple, Union
from matplotlib.style import context
import torch as torch
import torch.nn as nn
import numpy as np

from transformers import BertTokenizer, BertModel, EncoderDecoderModel, BertConfig, EncoderDecoderConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from transformers.modeling_outputs import BaseModelOutput

from modules.universal_sentence_embedding import universal_sentence_embedding
from modules.knowledge_encdec_model import KnowledgeEncoderDecoderModel
from utils.model_outputs import AdditionalSeq2SeqLMOutputWithKnow, AdditionalKnowledgeModelOutput
from utils.utils import neginf, shift_tokens_right

from modules.bert import MyBertModel2




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
        know_loss = knowledge_loss.detach().clone()
        correct_know_num = correct_know_num
        
        if loss is not None:
            total_loss = loss
            token_loss = loss.detach().clone()

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
class KDBert(KnowledgeEncoderDecoderModel):
    def __init__(self,
        encoder_config='bert-base-uncased',
        use_cs_ids=False,
        knowledge_alpha=0.25,
        knowledge_mode="argmax",
        concat_query=False,
    ):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        knowledge_encoder = DKnowledgeEncoder.from_pretrained(encoder_config,
            self.tokenizer,
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
        my_decoder = MyBertLMHeadModel2(self.decoder_config_)
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
            eos_tokens = [et[0].item() if len(et) != 0 else gen_result.shape[-1] for et in eos_tokens]
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
        know_loss = knowledge_loss.detach().clone()
        correct_know_num = correct_know_num
        
        if loss is not None:
            total_loss = loss
            token_loss = loss.detach().clone()

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









# Encoder with Knowledge Selection in MemNet
class DKnowledgeEncoder(BertModel):
    def __init__(self,
        config,
        tokenizer,
        add_pooling_layer=True,
        use_cs_ids=False,
        knowledge_alpha=0.25,
        knowledge_mode="argmax",
    ):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        
        self.use_cs_ids = use_cs_ids
        self.knowledge_alpha = knowledge_alpha
        self.knowledge_mode = knowledge_mode
        self.tokenizer = tokenizer

    
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
        encoder_outputs, extended_attention_mask,\
         knowledge_attention, chosen_knowledge_input_id, chosen_knowledge_pusedo_label = self.get_knowledge_selection_encoder_output(
            knowledge_input_ids,
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
        knowledge_input_ids,
        context_encoder_output,
        context_attention_mask,
        knowledge_encoder_output,
        knowledge_attention_mask,
        knowledge_sentences_length,
        knowledge_shape,
        use_cs_ids=False,
        knowledge_mode="argmax",
    ):

        context_use = universal_sentence_embedding(context_encoder_output, context_attention_mask)
        know_use = universal_sentence_embedding(knowledge_encoder_output, knowledge_attention_mask)

        k_bs, k_el, nk, kl = knowledge_shape

        embed_dim = know_use.shape[-1]
        know_use = know_use.reshape(k_bs*k_el, nk, embed_dim)
        context_use /= np.sqrt(embed_dim)
        know_use /= np.sqrt(embed_dim)

        ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ck_mask = (knowledge_sentences_length.reshape(-1,nk) != 0).to(context_use.device)
        ck_attn.masked_fill_(~ck_mask, neginf(context_encoder_output.dtype))

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

        cs_gold_idxs = torch.arange(k_bs*k_el, device=cs_ids.device) * nk
        cs_gold_encoded = knowledge_encoder_output[cs_gold_idxs]
        cs_gold_attn_mask = knowledge_attention_mask[cs_gold_idxs]
        gold_knowledge_input_ids = knowledge_input_ids[cs_gold_idxs]

        cs_encoded = knowledge_encoder_output[cs_offsets]
        cs_know_ids = knowledge_input_ids[cs_offsets]
        # but padding is (N * K, T)
        cs_mask = knowledge_attention_mask[cs_offsets]


        sel_gold_mask = self.get_knowledge_mask(gold_knowledge_input_ids)
        sel_gold_attn = torch.bmm(cs_encoded.detach().clone(), cs_gold_encoded.permute(0,2,1).detach().clone())
        sel_gold_attn = sel_gold_attn.masked_fill(sel_gold_mask[:, None, :], neginf(context_encoder_output.dtype))

        psuedo_label_idxs = torch.argmax(sel_gold_attn, dim=-1)
        pusedo_knowledge_label = torch.gather(gold_knowledge_input_ids, 1, psuedo_label_idxs)
        pusedo_knowledge_label = pusedo_knowledge_label.masked_fill(sel_gold_mask, -100)

        # finally, concatenate it all
        # full_enc = torch.cat([cs_encoded, context_encoder_output], dim=1)
        # full_mask = torch.cat([cs_mask, context_attention_mask], dim=1)

        full_enc = cs_encoded
        full_mask = cs_mask

        # also return the knowledge selection mask for the loss
        return full_enc, full_mask, ck_attn, cs_know_ids, pusedo_knowledge_label


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


    def get_knowledge_mask(self, knowledge_input_ids):
        cls_mask = knowledge_input_ids == self.tokenizer.cls_token_id
        sep_mask = knowledge_input_ids == self.tokenizer.sep_token_id
        pad_mask = knowledge_input_ids == self.tokenizer.pad_token_id
        underbar_mask = knowledge_input_ids == self.tokenizer.encode('_', add_special_tokens=False)[0]
        knowledge_mask = knowledge_input_ids == self.tokenizer.encode('knowledge', add_special_tokens=False)[0]

        special_token_mask = cls_mask + sep_mask + pad_mask + underbar_mask + knowledge_mask
        return special_token_mask









#### which is testing now
class KBertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self,
        config,
        use_cs_ids=False,
        knowledge_alpha=0.25,
        concat_query=True,
    ):
        super().__init__(config)

        self.bert = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False)
        self.bert2 = MyBertModel2.from_pretrained('bert-base-uncased', add_pooling_layer=False, is_decoder=True, add_cross_attention=True)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        self.use_cs_ids = use_cs_ids
        self.knowledge_alpha = knowledge_alpha
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.ck_attn_linear = torch.nn.Linear(768,768)
        self.ck_value_linear = torch.nn.Linear(768,768)
        self.ck_key_linear = torch.nn.Linear(768,768)

        self.ck_query_linear = torch.nn.Linear(768*2,768)
        self.concat_query = concat_query


    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        context_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask= attention_mask,
        ).last_hidden_state

        # get knowledge encoder output
        knowledge_encoder_outputs = self.bert(
            input_ids=knowledge_input_ids,
            attention_mask= knowledge_attention_mask,
        ).last_hidden_state

        # get selected knowledge-context encoder output and mask
        encoder_outputs, extended_attention_mask,\
         knowledge_attention, chosen_knowledge_input_id, chosen_knowledge_pusedo_label = self.get_knowledge_selection_encoder_output(
            knowledge_input_ids,
            context_encoder_outputs,
            attention_mask,
            knowledge_encoder_outputs,
            knowledge_attention_mask,
            knowledge_sentences_length,
            (k_bs, k_el, nk, kl),
            use_cs_ids=self.use_cs_ids,
            stage="1",
        )
        # encoder_outputs = BaseModelOutput(encoder_outputs)


        outputs = self.bert2(
            chosen_knowledge_input_id,
            attention_mask=extended_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=context_encoder_outputs,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        # stage 2 selection
        encoder_outputs2, extended_attention_mask2,\
         knowledge_attention2, _, _ = self.get_knowledge_selection_encoder_output(
            knowledge_input_ids,
            sequence_output,
            extended_attention_mask,
            knowledge_encoder_outputs,
            knowledge_attention_mask,
            knowledge_sentences_length,
            (k_bs, k_el, nk, kl),
            context_encoder_outputs,
            attention_mask,
            self.use_cs_ids,
            stage="2",
        )


        # for knowledge loss
        if self.knowledge_alpha != 0.0:
            know_loss, correct_know_num = self.get_knowledge_loss(
                knowledge_attention,
                knowledge_attention2,
                num_knowledge_sentences,
            )
        else:
            know_loss = torch.tensor(0., device=encoder_hidden_states.device)
            correct_know_num = torch.tensor(0., device=encoder_hidden_states.device)

        prediction_scores = self.cls(sequence_output)

        labels = chosen_knowledge_pusedo_label

        masked_lm_loss = None
        token_loss = torch.zeros(1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            token_loss = masked_lm_loss.detach().clone()

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if know_loss is not None:
            masked_lm_loss += know_loss

        # return MaskedLMOutput(
        #     loss=masked_lm_loss,
        #     logits=prediction_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return AdditionalSeq2SeqLMOutputWithKnow(
            loss=masked_lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.last_hidden_state,
            encoder_hidden_states=outputs.hidden_states,
            encoder_attentions=outputs.attentions,
            additional_losses=(token_loss, know_loss.detach()),
            correct_know_nums=correct_know_num,
        )


    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


    def get_knowledge_selection_encoder_output(
        self,
        knowledge_input_ids,
        context_encoder_output,
        context_attention_mask,
        knowledge_encoder_output,
        knowledge_attention_mask,
        knowledge_sentences_length,
        knowledge_shape,
        context_encoder_output_stage2=None,
        context_attention_mask_stage2=None,
        use_cs_ids=False,
        stage="1",
    ):

        context_use = universal_sentence_embedding(context_encoder_output, context_attention_mask)
        know_use = universal_sentence_embedding(knowledge_encoder_output, knowledge_attention_mask)

        k_bs, k_el, nk, kl = knowledge_shape


        # weight sum for good token in context
        if stage=="2" and self.concat_query:   # here, context_use is knowledge selected
            ck_query = self.ck_attn_linear(context_use)
            context_encoder_output_stage2_key = self.ck_key_linear(context_encoder_output_stage2)
            context_token_weights = torch.bmm(context_encoder_output_stage2_key, ck_query.unsqueeze(-1)).squeeze(-1)
            ctw_mask = (context_attention_mask_stage2 != 0.)
            context_token_weights = context_token_weights.masked_fill(~ctw_mask, neginf(context_encoder_output.dtype))
            context_token_weights = nn.functional.softmax(context_token_weights, dim=-1)

            context_encoder_output_stage2_value = self.ck_value_linear(context_encoder_output_stage2)
            context_token_ws = torch.bmm(context_token_weights.unsqueeze(-2), context_encoder_output_stage2_value).squeeze(-2)
            context_use = self.ck_query_linear(torch.concat([context_use, context_token_ws], dim=-1))


        embed_dim = know_use.shape[-1]
        know_use = know_use.reshape(k_bs*k_el, nk, embed_dim)
        context_use /= np.sqrt(embed_dim)
        know_use /= np.sqrt(embed_dim)

        ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ck_mask = (knowledge_sentences_length.reshape(-1,nk) != 0).to(context_use.device)
        ck_attn.masked_fill_(~ck_mask, neginf(context_encoder_output.dtype))

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

        cs_gold_idxs = torch.arange(k_bs*k_el, device=cs_ids.device) * nk
        cs_gold_encoded = knowledge_encoder_output[cs_gold_idxs]
        cs_gold_attn_mask = knowledge_attention_mask[cs_gold_idxs]
        gold_knowledge_input_ids = knowledge_input_ids[cs_gold_idxs]

        cs_encoded = knowledge_encoder_output[cs_offsets]
        cs_know_ids = knowledge_input_ids[cs_offsets]
        # but padding is (N * K, T)
        cs_mask = knowledge_attention_mask[cs_offsets]

        pusedo_knowledge_label = None
        if stage=="1":
            sel_gold_mask = self.get_knowledge_mask(gold_knowledge_input_ids)
            sel_gold_attn = torch.bmm(cs_encoded.detach().clone(), cs_gold_encoded.permute(0,2,1).detach().clone())
            sel_gold_attn = sel_gold_attn.masked_fill(sel_gold_mask[:, None, :], neginf(context_encoder_output.dtype))

            psuedo_label_idxs = torch.argmax(sel_gold_attn, dim=-1)
            pusedo_knowledge_label = torch.gather(gold_knowledge_input_ids, 1, psuedo_label_idxs)
            pusedo_knowledge_label = pusedo_knowledge_label.masked_fill(sel_gold_mask, -100)

            # finally, concatenate it all
            # full_enc = torch.cat([cs_encoded, context_encoder_output], dim=1)
            # full_mask = torch.cat([cs_mask, context_attention_mask], dim=1)

            full_enc = cs_encoded
            full_mask = cs_mask
        else:
            # 나중에 고치기 HACK
            full_enc = cs_encoded
            full_mask = cs_mask

        # also return the knowledge selection mask for the loss
        return full_enc, full_mask, ck_attn, cs_know_ids, pusedo_knowledge_label


    def get_knowledge_loss(
        self,
        knowledge_attention,
        knowledge_attention2,
        num_knowledge_sentences,
    ):
        if knowledge_attention is None:
            return None, None

        num_know = num_knowledge_sentences.view(-1)

        # arg max prediction
        _, know_pred = knowledge_attention2.max(1)
        knowledge_gold_ids = torch.zeros_like(know_pred).view(-1)
        masked_knowledge_gold_ids = knowledge_gold_ids.masked_fill(num_know.eq(0), -100)

        # get knowledge loss
        know_loss1 = torch.nn.functional.cross_entropy(
            knowledge_attention,
            masked_knowledge_gold_ids,
            reduction='mean',
            ignore_index=-100,
        )

        know_loss2 = torch.nn.functional.cross_entropy(
            knowledge_attention2,
            masked_knowledge_gold_ids,
            reduction='mean',
            ignore_index=-100,
        )

        know_loss = know_loss1 * 0.3 + know_loss2 * 0.7
        # get correct num
        correct_know_num = (know_pred == masked_knowledge_gold_ids).float().sum()
        
        return know_loss, correct_know_num


    def get_knowledge_mask(self, knowledge_input_ids):
        cls_mask = knowledge_input_ids == self.tokenizer.cls_token_id
        sep_mask = knowledge_input_ids == self.tokenizer.sep_token_id
        pad_mask = knowledge_input_ids == self.tokenizer.pad_token_id
        underbar_mask = knowledge_input_ids == self.tokenizer.encode('_', add_special_tokens=False)[0]
        knowledge_mask = knowledge_input_ids == self.tokenizer.encode('knowledge', add_special_tokens=False)[0]

        special_token_mask = cls_mask + sep_mask + pad_mask + underbar_mask + knowledge_mask
        return special_token_mask


    def generate(self, **kwargs):
        bs, el, seqlen = kwargs['input_ids'].shape
        return kwargs['input_ids'].reshape(bs*el, seqlen)