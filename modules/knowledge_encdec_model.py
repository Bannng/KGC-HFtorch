from transformers import EncoderDecoderModel
from transformers.modeling_outputs import ModelOutput
import torch
from typing import Optional, Dict, Any



class KnowledgeEncoderDecoderModel(EncoderDecoderModel):
    def flat_episode_batch_data(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        #######################################################################################
        # flatten episode batch
        if input_ids is not None and len(input_ids.shape) > 2:
            c_bs, c_el, max_context_length = input_ids.shape
            input_ids = input_ids.reshape(-1, max_context_length)

        if attention_mask is not None and len(attention_mask.shape) > 2:
            c_bs, c_el, max_context_length = attention_mask.shape
            attention_mask = attention_mask.reshape(-1, max_context_length)
        
        if decoder_input_ids is not None and len(decoder_input_ids.shape) > 2:
            r_bs, r_el, max_response_length = decoder_input_ids.shape
            decoder_input_ids = decoder_input_ids.reshape(-1, max_response_length)

        if decoder_attention_mask is not None and len(decoder_attention_mask.shape) > 2:
            r_bs, r_el, max_response_length = decoder_attention_mask.shape
            decoder_attention_mask = decoder_attention_mask.reshape(-1, max_response_length)

        if labels is not None and len(labels.shape) > 2:
            r_bs, r_el, max_response_length = labels.shape
            labels = labels.reshape(-1, max_response_length)
        #######################################################################################

        return {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'decoder_input_ids':decoder_input_ids,
            'decoder_attention_mask':decoder_attention_mask,
            'labels':labels,
        }


    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        device: torch.device = None,
    ) -> torch.LongTensor:

        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            ####################################################################################
            if 'attention_mask' in model_kwargs.keys():
                if len(model_kwargs['attention_mask'].shape) > 2:
                    _, episode_len = model_kwargs['attention_mask'].shape[:2]
                    batch_size = batch_size * episode_len
            ####################################################################################

            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            if device is None:
                device = self.device
            return torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id

    
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        #####################################################################################
        for key in ['input_ids', 'attention_mask']:
            if key in encoder_kwargs.keys():
                flat_dict = {key : encoder_kwargs[key]}
                encoder_kwargs[key] = self.flat_episode_batch_data(**flat_dict)[key]
        #####################################################################################

        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs