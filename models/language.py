#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' Inherent libs '''
import os


''' Third libs '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

''' Local libs '''





class LanguageModel(nn.Module):
    """
    A standard CACMT architecture. Base for this and many other modules.
    """

    def __init__(self, language_model_name="bert"):
        super(LanguageModel, self).__init__()
        if language_model_name == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.language_model = AutoModel.from_pretrained("bert-base-uncased")
            
        self.config = self.language_model.config
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.config.pad_token_id


    def get_caption_phrase_position(self, captions, captions_phrases):
        """[get the position of phrase in the caption]

        Args:
            captions ([list]): [nested list, in which each item is a list that contains the string of the phrase
                                For example: [['Military personnel greenish gray uniforms matching hats'],
                                                ['a man in red is standing in boat with a women']]. ]
            captions_phrases ([list]): [nested list, in which each item is also a list that contains the phrases of 
                                        the corresponding caption
                                For example: [[['Military personnel'], ['greenish gray uniforms'], ['matching hats']],
                                                [['a man'], ['in red'], ['boat with a women']]]. ]
        Return:
            captions_phrases_positions (list): [nested list that is the same structure of the captions_phrases, but 
                                                each item is a tuple that contains the positions of the corresponding 
                                                phrase in the original caption]
        """
        def find_sub_list(sl,l):
            sll=len(sl)
            for ind in (i for i,e in enumerate(l) if e==sl[0]):
                if l[ind:ind+sll]==sl:
                    return [ind,ind+sll]

        captions_phrases_positions = list()

        for caption_i in range(len(captions)):
            caption = captions[caption_i][0] # a string of the caption
            tokenizer_caption = self.tokenizer(caption, return_tensors="pt", add_special_tokens=False)
            token_caption_ids = tokenizer_caption.input_ids.numpy()[0].tolist()

            caption_phrases = captions_phrases[caption_i] # a nested list
            start_point = 0
            caption_phrases_positions = list()
            for caption_phrase_i in range(len(caption_phrases)):
                caption_phrase = caption_phrases[caption_phrase_i][0]

                tokenizer_caption_phrase = self.tokenizer(caption_phrase, return_tensors="pt", add_special_tokens=False)
                token_caption_phrase_ids = tokenizer_caption_phrase.input_ids.numpy()[0].tolist()

                tg_token_caption_ids = token_caption_ids[start_point:]
                phrase_token_indexs = find_sub_list(token_caption_phrase_ids, token_caption_ids)
                
                start_point = start_point+phrase_token_indexs[1]
                caption_phrases_positions.append(phrase_token_indexs)
            captions_phrases_positions.append(caption_phrases_positions)

        return captions_phrases_positions

    def slice_phrase_embeddings(self, wordwise_embed_captions, captions_phrases_positions):
        num_captions = len(captions_phrases_positions)
        sliced_caps_phs_embed = list()
        for cap_i in range(num_captions):
            cap_embeddings = wordwise_embed_captions[cap_i, :, :] # [max_length_of_caption + 2, d]
            cap_phrases_pos = captions_phrases_positions[cap_i]
            sliced_cap_phs_embed = list()
            for cap_ph_i in range(len(cap_phrases_pos)):
                phrase_pos = cap_phrases_pos[cap_ph_i]

                # slicing the features of phrase, 
                #   add 1 because the tokenzer add start and end token in the caption
                phrase_pos[0] = phrase_pos[0] + 1
                phrase_pos[1] = phrase_pos[1] + 1
                phrase_embeddings = cap_embeddings[phrase_pos[0]:phrase_pos[1], :]

                sliced_cap_phs_embed.append(phrase_embeddings)
            sliced_caps_phs_embed.append(sliced_cap_phs_embed)

        return sliced_caps_phs_embed

    def integrate_pad_sliced_phrases_embds(self, sliced_caps_phs_embed):
        """[Integrate the word embeddings in phrase and then pad the phrases in the caption]

        Args:
            sliced_caps_phs_embed ([list]): [a list that contains the word embeddings of phrases in the captions]
        """
        num_of_captions = len(sliced_caps_phs_embed)
        max_number_of_phrases = max([len(caption_phs_embds) for caption_phs_embds in sliced_caps_phs_embed])
        embd_dim = sliced_caps_phs_embed[0][0].shape[1]
        
        captions_phrases_mask = list()
        captions_phrases_embds = list()
        for cap_i in range(num_of_captions):
            cap_phrases_embds = sliced_caps_phs_embed[cap_i]
            cap_phrases_mask = list()
            integrated_cap_phrases = list()
            for cap_phrase_i in range(max_number_of_phrases):
                if cap_phrase_i < len(cap_phrases_embds):
                    cap_phrases_mask.append(False)
                    phrase_embds = cap_phrases_embds[cap_phrase_i]
                    integrated_phrase_embds = torch.mean(phrase_embds, 0) # a tensor with shape [1, d]
                    
                else:
                    cap_phrases_mask.append(True)
                    integrated_phrase_embds = torch.zeros(embd_dim)

                integrated_cap_phrases.append(integrated_phrase_embds)

            captions_phrases_embds.append(integrated_cap_phrases)
            captions_phrases_mask.append(cap_phrases_mask)

        flatten_captions_phrases_embds = [ph_embds for cap_phs_embds in captions_phrases_embds for ph_embds in cap_phs_embds]

        integrated_captions_phrases_embds = torch.stack(flatten_captions_phrases_embds, 0)
        integrated_captions_phrases_embds = integrated_captions_phrases_embds.reshape((num_of_captions, max_number_of_phrases, -1))

        return integrated_captions_phrases_embds, captions_phrases_mask


    def forward(self, captions, captions_phrases=None):
        """[Compute the features of caption_phrases,
        
            Note: Applying this function produces the embedding for each phrase of the input captions. 
                    Besides, with the padding process, each caption is made to contain same number of phrases]

        Args:
            captions ([list]): [a list in which each item is the list that contains the caption of the image,
                                Thus, this list contains the captions of one batch of data
                                For example: 
                                    [[['Phone line workers doing repairs .'], 
                                    ["Several men working to repair a McDonald 's sign ."], 
                                    ['Two dogs running on grass']]]
                                ]
            captions_phrases ([list]): [a nested list in which each item is a list that contains the corresponding phrases of one image]

        Returns:
            [integrated_captions_phrases_embds]: [a torch with shape <batch_size, max_number_of_phrases, embd_dim>
                                                    padding with zeros]
            [captions_phrases_mask]: [a nested list that each item is a list containing the padding information of the corresponding caption]
        """
        # conver to one-depth nested list - an example of this type of caption is shown in
        #   the main test in this file. i.e., line 204 
        captions = [caption[0] for caption in captions]

        # convert strcuture of the captions to one list that each item is a string of the caption of the image. 
        flatten_captions = [caption[0] for caption in captions]

        tokenizer_captions = self.tokenizer(flatten_captions, return_tensors="pt", padding=True)
        token_captions_ids = tokenizer_captions.input_ids
        
        padded_token_captions = [self.tokenizer.decode(token_caption_ids) for token_caption_ids in token_captions_ids]

        embed_captions = self.language_model(**tokenizer_captions)
        # tensor with shape [batch_size, max_length_of_caption + 2, d]
        #   - 2 here means that tokenzier will antomically insert the start and end workds to one caption
        wordwise_embed_captions = embed_captions[0] 
        integrated_embed_captions = embed_captions[1] # # tensor with shape [batch_size, d]

        #print("wordwise_embed_captions: ", wordwise_embed_captions)

        captions_phrases_positions = self.get_caption_phrase_position(captions=captions, 
                                                                    captions_phrases=captions_phrases)
 
        sliced_caps_phs_embed = self.slice_phrase_embeddings(wordwise_embed_captions=wordwise_embed_captions, 
                                                            captions_phrases_positions=captions_phrases_positions)
        integrated_captions_phrases_embds, captions_phrases_mask = self.integrate_pad_sliced_phrases_embds(sliced_caps_phs_embed)


        return integrated_captions_phrases_embds, captions_phrases_mask




if __name__=="__main__":
    captions = [['Military personnel greenish gray uniforms matching hats'],
                ['a man in red is standing in boat with a women'],
                ['a man and a women sleep in a beautiful wooden bed with golves'],
                ['a man and a women stand in front of a man and women while other men and women are playing']]

    captions_phrases = [[['Military personnel'], ['greenish gray uniforms'], ['matching hats']],
                        [['a man'], ['in red'], ['boat with a women']],
                        [['a man'], ['a women'], ['a beautiful wooden bed with golves']],
                        [['a man'], ['a women'], ['a man'], ['women'], ['other men and women']]]
    
    lang_model = LanguageModel(language_model_name="bert")    

    lang_model(captions, captions_phrases)