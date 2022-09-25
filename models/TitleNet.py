from models.TMemNet import TMemNetBert, ContextKnowledgeEncoder

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



class TitleNet(TMemNetBert):
    def __init__(self,
        encoder_config='bert-base-uncased',
        use_cs_ids=False,
        knowledge_alpha=0.25,
        max_title_num=5
    ):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        knowledge_encoder = TitleKnowledgeEncoder.from_pretrained(encoder_config,
            self.tokenizer,
            use_cs_ids=use_cs_ids,
            knowledge_alpha=knowledge_alpha,
            max_title_num=max_title_num,
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
        super(TMemNetBert, self).__init__(config=self.encoder_decoder_config, encoder=knowledge_encoder)

        self.use_cs_ids = use_cs_ids
        self.knowledge_alpha = knowledge_alpha
        self.max_title_num = max_title_num

        self.config.bos_token_id = self.tokenizer.cls_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.sep_token_id
        self.config.num_beams = 1
        self.config.max_length = 51
        self.config.no_repeat_ngram_size = 0



class TitleKnowledgeEncoder(BertModel):
    def __init__(self,
        config,
        tokenizer,
        add_pooling_layer=True,
        use_cs_ids=False,
        knowledge_alpha=0.25,
        max_title_num=5
    ):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        
        self.use_cs_ids = use_cs_ids
        self.knowledge_alpha = knowledge_alpha
        self.tokenizer = tokenizer
        self.max_title_num = max_title_num


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


        title_tensor, total_titles = self.extract_title(knowledge_input_ids)
        t_bs, t_el, nt, tl = title_tensor.shape
        title_tensor = title_tensor.reshape(-1, tl).to(knowledge_input_ids.device)
        title_attention_mask = (title_tensor != self.tokenizer.pad_token_id).type(torch.DoubleTensor).to(knowledge_input_ids.device)



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

        # title encoder output
        title_encoder_outputs = super().forward(
            input_ids=title_tensor,
            attention_mask=title_attention_mask,
        ).last_hidden_state

        # get selected knowledge-context encoder output and mask
        encoder_outputs, attention_mask, knowledge_attention, title_attention = self.get_knowledge_selection_encoder_output(
            context_encoder_outputs,
            attention_mask,
            knowledge_encoder_outputs,
            knowledge_attention_mask,
            knowledge_sentences_length,
            (k_bs, k_el, nk, kl),
            title_encoder_outputs,
            title_attention_mask,
            (t_bs, t_el, nt, tl),
            knowledge_input_ids,
            title_tensor,
            self.use_cs_ids,
        )
        # encoder_outputs = BaseModelOutput(encoder_outputs)

        # for knowledge loss
        if self.knowledge_alpha != 0.0:
            know_loss, correct_know_num = self.get_knowledge_loss(
                knowledge_attention,
                num_knowledge_sentences,
                title_attention,
            )
        else:
            know_loss = None
            correct_know_num = None

        return AdditionalKnowledgeModelOutput(
            last_hidden_state=encoder_outputs,
            extended_attention_mask=attention_mask,
            knowledge_attention=knowledge_attention,
            additional_losses=(know_loss),
            correct_know_nums=correct_know_num,
        )


    def extract_title(self, knowledge_input_ids):
        k_bs, k_el, nk, kl = knowledge_input_ids.shape

        sep_mark = '_ _ knowledge _ _'
        
        total_titles = []
        tokenized_total_titles = []
        title_pool_len = []
        max_title_seq_len = []
        for bs in range(k_bs):
            batch = []
            batch_tokenized = []
            for el in range(k_el):
                decoded_know = self.tokenizer.batch_decode(knowledge_input_ids[bs, el])
                titles = [dk.split(sep_mark)[0] + self.tokenizer.sep_token for dk in decoded_know]
                titles = [t for t in titles if not self.tokenizer.pad_token in t]
                
                if len(titles) > 0:
                    gold_title = titles[0]
                    false_title = [t for t in titles if t != gold_title]
                    extracted_title_pool = [gold_title] + list(set(false_title))
                    tokenized_title_pool = self.tokenizer(extracted_title_pool,
                                                          return_tensors='pt',
                                                          padding='longest',
                                                          add_special_tokens=False)['input_ids']
                else:
                    extracted_title_pool = titles
                    tokenized_title_pool = torch.tensor([[self.tokenizer.pad_token_id]])
                batch.append(extracted_title_pool)
                batch_tokenized.append(tokenized_title_pool)
                title_pool_len.append(len(extracted_title_pool))
                max_title_seq_len.append(tokenized_title_pool.shape[-1])
            total_titles.append(batch)
            tokenized_total_titles.append(batch_tokenized)

        max_pool_len = max(title_pool_len)
        max_seq_len = max(max_title_seq_len)
        
        batch_tensor = []
        for bs in range(k_bs):
            bt = []
            for el in range(k_el):
                picked = tokenized_total_titles[bs][el]
                padded_title = torch.nn.functional.pad(
                    picked,
                    pad=(0, max_seq_len-picked.shape[1], 0, max_pool_len-picked.shape[0]),
                    value=self.tokenizer.pad_token_id
                )
                bt.append(padded_title)
            batch_tensor.append(torch.stack(bt, axis=0))
        
        title_tensor = torch.stack(batch_tensor, axis=0)
        return title_tensor, total_titles



    def get_knowledge_selection_encoder_output(
        self,
        context_encoder_output,
        context_attention_mask,
        knowledge_encoder_output,
        knowledge_attention_mask,
        knowledge_sentences_length,
        knowledge_shape,
        title_encoder_output,
        title_attention_mask,
        title_shape,
        knowledge_input_ids,
        title_input_ids,
        use_cs_ids=False,
    ):
        context_use = universal_sentence_embedding(context_encoder_output, context_attention_mask)
        know_use = universal_sentence_embedding(knowledge_encoder_output, knowledge_attention_mask)
        title_use = universal_sentence_embedding(title_encoder_output, title_attention_mask)

        k_bs, k_el, nk, kl = knowledge_shape
        t_bs, t_el, nt, tl = title_shape

        embed_dim = know_use.shape[-1]
        know_use = know_use.reshape(k_bs*k_el, nk, embed_dim)
        title_use = title_use.reshape(t_bs*t_el, nt, embed_dim)

        context_use /= np.sqrt(embed_dim)
        know_use /= np.sqrt(embed_dim)
        title_use /= np.sqrt(embed_dim)


        title_sentences_length = title_attention_mask.reshape(t_bs, t_el, nt, tl).sum(-1)
        ct_attn = torch.bmm(title_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ct_mask = (title_sentences_length.reshape(-1,nt) != 0).to(context_use.device)
        ct_attn.masked_fill_(~ct_mask, neginf(context_encoder_output.dtype))


        ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)
        # fill with near -inf
        ck_mask = (knowledge_sentences_length.reshape(-1,nk) != 0).to(context_use.device)
        # ck_attn.masked_fill_(~ck_mask, neginf(context_encoder_output.dtype))

        if not use_cs_ids:
            # if we're not given the true chosen_sentence (test time), pick our
            # best guess
            top_k = self.max_title_num if self.max_title_num <= ct_attn.shape[-1] else ct_attn.shape[-1]
            _, ct_ids = torch.topk(ct_attn, k=top_k, dim=-1)
            title_available_idxs = ct_mask.sum(-1).reshape(-1,1)
            # pad 는 이미 masking 되었으니까 그냥 해도 될 듯? (ct_ids 에 pad 뽑혀있어도, 거기 index만 살리고 나머지를 masking 할거라서 상관 X)

            ct_ids = ct_ids.reshape(t_bs, t_el, -1)
            knowledge_input_ids = knowledge_input_ids.reshape(k_bs, k_el, nk, kl)
            title_input_ids = title_input_ids.reshape(t_bs, t_el, nt, tl)
            sep_mark = '_ _ knowledge _ _'

            batch_title_mask = []
            for bs in range(k_bs):
                episode_title_mask = []
                for el in range(k_el):
                    decoded_knowledge = self.tokenizer.batch_decode(knowledge_input_ids[bs][el])
                    decoded_title = self.tokenizer.batch_decode(title_input_ids[bs][el])

                    selected_title = [dt.split(self.tokenizer.sep_token)[0].strip() for idx,dt in enumerate(decoded_title) if idx in ct_ids[bs][el]]
                    title_mask = torch.tensor([1 if dk.split(sep_mark)[0].strip() in selected_title else 0 for dk in decoded_knowledge])
                    episode_title_mask.append(title_mask)
                batch_title_mask.append(torch.stack(episode_title_mask, axis=0))

            total_batch_title_mask = torch.stack(batch_title_mask, axis=0)
            flat_title_mask = total_batch_title_mask.reshape(-1, nk).to(context_use.device)

            ck_mask_sel = (flat_title_mask * ck_mask).to(torch.bool)
            ck_attn_sel = ck_attn.masked_fill(~ck_mask_sel, neginf(context_encoder_output.dtype))
            _, cs_ids = ck_attn_sel.max(1)
        else:
            cs_ids = torch.zeros_like(ck_attn.max(1)[1])

        ck_attn.masked_fill_(~ck_mask, neginf(context_encoder_output.dtype))
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
        return full_enc, full_mask, ck_attn, ct_attn

    def get_knowledge_loss(
        self,
        knowledge_attention,
        num_knowledge_sentences,
        title_attention,
    ):
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

        title_loss = torch.nn.functional.cross_entropy(
            title_attention,
            masked_knowledge_gold_ids,
            reduction='mean',
            ignore_index=-100,
        )
        know_loss = know_loss + title_loss

        # get correct num
        correct_know_num = (know_pred == masked_knowledge_gold_ids).float().sum()
        
        return know_loss, correct_know_num