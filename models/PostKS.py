from typing import Dict, Optional, List, Tuple, Union
import torch as torch
import torch.nn as nn
import numpy as np

from transformers import BertTokenizer, BertModel, EncoderDecoderModel, BertConfig, EncoderDecoderConfig
from transformers.modeling_outputs import BaseModelOutput

from modules.universal_sentence_embedding import universal_sentence_embedding
from modules.knowledge_encdec_model import KnowledgeEncoderDecoderModel
from utils.model_outputs import AdditionalSeq2SeqLMOutputWithKnow, AdditionalKnowledgeModelOutput
from utils.utils import neginf, shift_tokens_right




# TransformerMemoryNetwork with Bert-base-uncased encoder
class PostKSBert(KnowledgeEncoderDecoderModel):
    def __init__(self,
        encoder_config='bert-base-uncased',
        use_cs_ids=False,
        knowledge_alpha=0.25
    ):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        knowledge_encoder = PostKSEncoder.from_pretrained(encoder_config,
            self.tokenizer,
            use_cs_ids=use_cs_ids,
            knowledge_alpha=knowledge_alpha,
        )
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
        knowledge_loss, kl_loss = encoder_outputs.additional_losses
        correct_know_num = encoder_outputs.correct_know_nums

        correct_know_num = torch.tensor(0., device=encoder_hidden_states.device) if correct_know_num is None else correct_know_num
        knowledge_loss = torch.tensor(0., device=encoder_hidden_states.device) if knowledge_loss is None else knowledge_loss
        kl_loss = torch.tensor(0., device=encoder_hidden_states.device) if kl_loss is None else kl_loss
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
        correct_know_num = correct_know_num
        
        if loss is not None:
            total_loss = loss
            token_loss = loss.detach().clone()

        if total_loss is not None and knowledge_loss is not None:
            total_loss += self.knowledge_alpha * knowledge_loss

        if total_loss is not None and kl_loss is not None:
            total_loss += kl_loss
        ###########################################################
        know_loss = knowledge_loss.detach().clone()
        kldiv_loss = kl_loss.detach().clone()


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
            additional_losses=(token_loss, know_loss, kldiv_loss),
            correct_know_nums=correct_know_num,
        )




# Encoder with Knowledge Selection in MemNet
class PostKSEncoder(BertModel):
    def __init__(self,
        config,
        tokenizer=None,
        add_pooling_layer=True,
        use_cs_ids=False,
        knowledge_alpha=0.25,
        detach_posterior=True,
    ):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        
        self.use_cs_ids = use_cs_ids
        self.knowledge_alpha = knowledge_alpha
        self.tokenizer = tokenizer
        self.post_query_linear = nn.Linear(768*2, 768)

        # from Pytorch doc, if log_target = False (which is default),
        # input (prior) should be in log-space
        # but target (posterior) would be in probability space
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.detach_posterior = detach_posterior

    
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

        # flatten responses
        r_bs, r_el, max_response_length = response.shape
        response = response.reshape(-1, max_response_length)
        response_attention_mask = (response != self.tokenizer.pad_token_id).type(torch.DoubleTensor).to(response.device)


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

        # get response encoder output
        response_encoder_outputs = super().forward(
            input_ids=response,
            attention_mask= response_attention_mask,
        ).last_hidden_state


        # get selected knowledge-context encoder output and mask - posterior knowledge selection
        encoder_outputs, attention_mask,\
         knowledge_attention, posterior_attention = self.get_knowledge_selection_encoder_output(
            context_encoder_outputs,
            attention_mask,
            knowledge_encoder_outputs,
            knowledge_attention_mask,
            knowledge_sentences_length,
            (k_bs, k_el, nk, kl),
            response_encoder_outputs,
            response_attention_mask,
            self.use_cs_ids,
        )
        # encoder_outputs = BaseModelOutput(encoder_outputs)

        # knowledge loss - Prior Posterior KLDiv and Supervised Knowledge loss
        if self.knowledge_alpha != 0.0:
            know_loss, correct_know_num, kl_loss = self.get_knowledge_loss(
                knowledge_attention,
                posterior_attention,
                num_knowledge_sentences,
            )
        else:
            know_loss = None
            correct_know_num = None
            kl_loss = None

        return AdditionalKnowledgeModelOutput(
            last_hidden_state=encoder_outputs,
            extended_attention_mask=attention_mask,
            knowledge_attention=knowledge_attention,
            additional_losses=(know_loss, kl_loss),
            correct_know_nums=correct_know_num,
        )

    def get_knowledge_selection_encoder_output(
        self,
        context_encoder_output,
        context_attention_mask,
        knowledge_encoder_output,
        knowledge_attention_mask,
        knowledge_sentences_length,
        knowledge_shape,
        response_encoder_output,
        response_attention_mask,
        use_cs_ids=False,
    ):
        context_use = universal_sentence_embedding(context_encoder_output, context_attention_mask)
        know_use = universal_sentence_embedding(knowledge_encoder_output, knowledge_attention_mask)
        
        # for posterior
        response_use = None
        post_attn = None

        k_bs, k_el, nk, kl = knowledge_shape

        embed_dim = know_use.shape[-1]
        know_use = know_use.reshape(k_bs*k_el, nk, embed_dim)
        context_use /= np.sqrt(embed_dim)
        know_use /= np.sqrt(embed_dim)

        # prior attention
        ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ck_mask = (knowledge_sentences_length.reshape(-1,nk) != 0).to(context_use.device)
        ck_attn.masked_fill_(~ck_mask, neginf(context_encoder_output.dtype))
        
        
        # make posterior distribution
        # if self.training:
        response_use = universal_sentence_embedding(response_encoder_output, response_attention_mask)
        response_use /= np.sqrt(embed_dim)
        post_query = self.post_query_linear(torch.concat([context_use, response_use], dim=-1))
        # posterior 
        post_attn = torch.bmm(know_use, post_query.unsqueeze(-1)).squeeze(-1)
        post_attn.masked_fill_(~ck_mask, neginf(context_encoder_output.dtype))


        # select knowledge from gumbel softmax
        if self.training and post_attn is not None:
            cs_onehot = nn.functional.gumbel_softmax(post_attn, tau=0.1, hard=True, dim=-1)
            cs_ids = torch.argmax(cs_onehot, dim=-1)

            cs_onehot_ = cs_onehot.unsqueeze(1).unsqueeze(1)
            knowledge_encoder_output_ = knowledge_encoder_output.reshape(k_bs*k_el, nk, kl, embed_dim).transpose(1,2)
            selected_knowledges = []
            for i in range(cs_onehot.shape[0]):
                sel_know = torch.matmul(cs_onehot_[i], knowledge_encoder_output_[i]).squeeze(1)
                selected_knowledges.append(sel_know)
            
            cs_offsets = torch.arange(k_bs*k_el, device=cs_ids.device) * nk + cs_ids
            cs_encoded = torch.stack(selected_knowledges, dim=0)

        else:
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

        # finally, concatenate it all
        full_enc = torch.cat([cs_encoded, context_encoder_output], dim=1)
        full_mask = torch.cat([cs_mask, context_attention_mask], dim=1)

        # also return the knowledge selection mask for the loss
        return full_enc, full_mask, ck_attn, post_attn

    def get_knowledge_loss(
        self,
        knowledge_attention,
        posterior_attention,
        num_knowledge_sentences,
    ):
        # from Pytorch doc, if log_target = False (which is default),
        # input (prior) should be in log-space
        # but target (posterior) would be in probability space

        # 1. get KLDiv Loss for training
        #    two distributions are non-normalized logits, need to be transformed as mentioned above
        kl_loss = torch.tensor(0., device=knowledge_attention.device)
        if posterior_attention is not None:
            prior_dist = nn.functional.log_softmax(knowledge_attention, dim=-1)
            posterior_dist = nn.functional.softmax(posterior_attention, dim=-1)
            posterior_dist = posterior_dist.detach() if self.detach_posterior else posterior_dist
            kl_loss = self.kld_loss(prior_dist, posterior_dist)


        # 2. get Supervised Knowledge Loss
        num_know = num_knowledge_sentences.view(-1)

        # arg max prediction
        _, know_pred = knowledge_attention.max(1)
        knowledge_gold_ids = torch.zeros_like(know_pred).view(-1)
        masked_knowledge_gold_ids = knowledge_gold_ids.masked_fill(num_know.eq(0), -100)


        if posterior_attention is not None and self.training:
            loss_attention = posterior_attention
        else:
            loss_attention = knowledge_attention
        
        # get knowledge loss - only for posterior distribution
        know_loss = torch.nn.functional.cross_entropy(
            loss_attention,
            masked_knowledge_gold_ids,
            reduction='mean',
            ignore_index=-100,
        )

        # get correct num
        correct_know_num = (know_pred == masked_knowledge_gold_ids).float().sum()
        
        return know_loss, correct_know_num, kl_loss
