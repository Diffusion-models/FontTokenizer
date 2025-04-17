# ref : rewrite_diffusion_git/font_generator/train_code/data/fontdata.py
# time : 2025-03-17
# env: conda-env (font_gen)

import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import random, re
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import List, Optional, Union
import json

#============================================================
# Tokenizer output
#============================================================
class TokenEmbeddings:
    input_ids=None
    mask=None
    length=None
    input_str=None

#============================================================
# function
#============================================================
def get_p1p2(ann_line, specifical_part:list=[]):
    """
    Input:
        丕<$U不(jsfk)$D一(j)>
    Output:
        丕/##$U/不/#j/#s/#f/#k/##$D/一/#j/
    """
    assert "<" in ann_line, "ann:[%s] error, have no '<'"%ann_line
        
    # get strokes (p1_strokes, p2_strokes)
    strokes = re.finditer(r'[a-z]+', ann_line)

    # get position encode (p1_pos, p2_pos)
    matches = re.finditer(r"(!L|!R|\$U|\$D|&LU|&LD|&RU|&RD|\*)", ann_line)

    c = ann_line[0]
    out = (c,)  # add context to output
    for match, stroke in zip(matches, strokes):
        part_pos = match.group()
        pos_end_idx = match.end()

        part_c = ann_line[pos_end_idx]
        if part_c in specifical_part:
            part_c = "#"+part_c

        part_stroke = stroke.group()
        out += ("##"+part_pos, part_c, part_stroke)     # add "##" to part_pos
    return out

#============================================================
# Main Class
#============================================================
class FontTokenizer:
    """only support json file input"""
    def __init__(self, token_dict:str, context_length:int=77):
        assert token_dict.endswith(".json"), "token_dict must be json file"
        with open(token_dict, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        self.specifical_part = json_data["specifical_part"]

        self.special_tokens = ["<bos_token_id>", "<eos_token_id>", "<sep_token_id>", "pad"]

        self.token_dict = json_data["token_dict"]
        self.len = json_data["len"]
        self.json_data = json_data

        print("=="*20)
        print("token dict length:", self.len, "contain 5000 style tokens")
        print("max annotation length:", context_length)
        print("=="*20)

    def save_pretrained(self, dest):
        pass

    def csv_encode(self, ann_line:str, max_length:int=77, return_tensors="pt")->TokenEmbeddings:
        """ Read CSV format dataset 
        add <bos_token_id> and <eos_token_id> to input_ids at first and last location 
        Args:
            ann_line : 灾 
                    or 灾|<|##$U|宀|(|#k|#d|#e|)|##$D|火|(|#d|#s|#s|#l|)|> 
                    or 灾|<|##$U|宀|(|)|##$D|火|(|)|> 
                    or <|##$U|宀|(|)|##$D|火|(|)|>
            max_length : max length of input_ids
            return_tensors : return type, "pt" or "np"
        Return :
            input_ids : list  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] or tensors shape(1, 10)
            mask : list  [True, True, True, True, True, True, True, True, True, True] or tensors
            length : 10, strokes length + 2 (bos and eos)
        """
        if ann_line[0] == "|":
            ann_line_new = ann_line[1:]
            input_str = ["|"] + [s for s in ann_line_new.split("|") if s!=""]
        else:
            input_str = ann_line.split("|")

        # Add <bos_token_id> at first and <eos_token_id> at last
        input_str = ["<bos_token_id>"] + input_str + ["<eos_token_id>"]

        num_in = len(input_str)
        assert num_in <= max_length, f"input_str:{num_in} > max_length:{max_length}"
        mask = [True]*num_in

        # Padded to max_length
        input_ids = [self.token_dict[s] for s in input_str] + \
                    [self.token_dict["pad"],]*(max_length - num_in)
        mask_padded = mask + [False] * (max_length - num_in)

        if return_tensors == "pt":
            input_ids = torch.LongTensor(input_ids).unsqueeze(0)
            mask_padded = torch.BoolTensor(mask_padded).unsqueeze(0)

        out = {'input_ids': input_ids, "mask":mask_padded, "length":num_in, "input_str":input_str}
        out = SimpleNamespace(**out)
        return out
    
    def embedding(self, input_str:list, return_tensors="pt"):
        """ Return embedding token_id without <bos> and <eos> token
        Args:
            input_str : ["s_100"] str list
            return_tensors : return type, "pt" or "np"
        Return :
            input_ids : list  [100] or tensors shape(1, 10)
            mask : list  [True] or tensors
            length : 10, strokes length + 2 (bos and eos)       "
        """
        if not isinstance(input_str, list):
            input_str = [input_str]
        input_ids = [self.token_dict[s] for s in input_str]
        mask = [True]*len(input_ids)
        if return_tensors == "pt":
            input_ids = torch.LongTensor(input_ids).unsqueeze(0)
            mask = torch.BoolTensor(mask).unsqueeze(0)
        out = {"input_ids": input_ids, "mask":mask, "length": len(input_ids)}
        return SimpleNamespace(**out)
    
    def encode(self, ann_line:str, output_type:int, max_length:int=77, return_tensors="pt",
               c_location="first")->TokenEmbeddings:
        """ Read annotation_file format dataset
        add <bos_token_id> and <eos_token_id> to input_ids at first and last location
        Args:
            ann_line : 丕                                  output=1
                    or 丕 < $U 不 ( j s f k ) $D 一 ( j ) > output=2
                    or 丕 < $U 不 ( ) $D 一 ( ) >             output=3
                    or <|##$U|宀|(|)|##$D|火|(|)|>
            output_type : 0 for csv_encode, 1 for embedding
            max_length : max length of input_ids
            return_tensors : return type, "pt" or "np"
            c_location : context location, "first"(丕<$U不()$D一()>) or "last" (<$U不()$D一()>丕), the last is better in CausualLM architecture
        Return :
            input_ids : list  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] or tensors shape(1, 10)       
            mask : list  [True, True, True, True, True, True, True, True, True, True] or tensors
            length : 10, strokes length + 2 (bos and eos)   "
        """
        if "<" not in ann_line or "<" == ann_line:
            # get unsplitable content
            
            if output_type == 1:                 # single char : [亭]
                input_str = [ann_line[0]]
            elif output_type == 2:               # char and stroke : [亭 ( #k #j #f #c #j #d #e #j #g )] 
                input_str = [ann_line[0], "(",] + ["#"+l for l in ann_line[2:-1]] +[")"] 
            elif output_type == 3:
                input_str = [ann_line[0], "(",")" ]
            else:
                input_str = [ann_line[0]]       # [亭()]

            if c_location == "last":
                input_str = input_str[1:] + input_str[0] if len(input_str) > 1 else input_str

        else:
            # get splitable content
            stroke_info = get_p1p2(ann_line)

            if output_type == 1:                       # single char : [交]
                input_str = [ann_line[0]]
            elif output_type == 2:                     # char and stroke : [交 < ##$U 亠 ( #k #j ) ##$D 父 ( #s #k #s #l ) > ]
                input_str = [stroke_info[0], "<", stroke_info[1], stroke_info[2], "("]
                input_str += ["#"+s for s in stroke_info[3]] + [")"]  
                input_str += [stroke_info[4], stroke_info[5]] + ["("] 
                input_str += ["#"+s for s in stroke_info[6]] + [")", ">"] 
            elif output_type == 3:                     # only part : [交 < ##$U 亠 ( ) ##$D 父 (  ) > ]
                input_str = [stroke_info[0], "<",stroke_info[1], stroke_info[2], "(", ")", 
                                stroke_info[4], stroke_info[5], "(", ")",  ">"]
            elif output_type == 4:                     # only part without context : [< ##$U 亠 ( ) ##$D 父 (  ) > ]
                input_str = ["<",stroke_info[1], stroke_info[2], "(", ")", 
                                stroke_info[4], stroke_info[5], "(", ")",  ">"]
            else:
                input_str = input_str

            if c_location == "last" and output_type in [2, 3]:
                input_str = input_str[1:] + input_str[0] if len(input_str) > 1 else input_str

        # Add <bos_token_id> at first and <eos_token_id> at last
        input_str = ["<bos_token_id>"] + input_str + ["<eos_token_id>"]

        num_in = len(input_str)
        assert num_in <= max_length, f"input_str len:{num_in} > max_length:{self.max_length}"
        mask = [True]*num_in

        # pad to max_length with [pad] token
        input_ids = [self.token_dict[s] for s in input_str] + \
                    [self.token_dict["pad"], ]*(max_length - num_in)
        mask_padded = mask + [False] *(max_length - num_in)

        if return_tensors == "pt":
            input_ids = torch.LongTensor(input_ids).unsqueeze(0)
            mask_padded = torch.BoolTensor(mask_padded).unsqueeze(0)

        out = {"input_ids":input_ids, "mask":mask_padded, "length":num_in, 
               "input_str":input_str}
        
        return SimpleNamespace(**out)
    
    def batch_decode(self, tokens:Union[dict, torch.tensor], skip_special_tokens=True):
        """
        tokens : bs, seq_len
        skip_special_tokens : bool, remove special tokens ["<bos_token_id>","<eos_token_id>","<sep_token_id>", "pad"]
        """
        if skip_special_tokens:
            special_token_ids = [self.token_dict[t] for t in self.special_tokens]
        else:
            special_token_ids = []
        # 21146 ~ <eos_token_id>
        result = [[self.token_dict["%05d"%t_id] for t_id in row[:row.index(21146)] if t_id not in special_token_ids] for row in tokens.tolist()]

        return result
    

    def decode(self, tokens:Union[dict, torch.tensor], rm_pad=True):
        pass

    def __call__(self, texts:Union[str, List[str]], context_length:Optional[int]=77,
                 c_location="first")->TokenEmbeddings:
        """
        :params c_location : context location, "first"(丕<$U不()$D一()>) or "last" (<$U不()$D一()>丕), the last is better in CausualLM architecture
        """

        if isinstance(texts, str):
            texts = [texts]

        if c_location == "last":
            re_location_texts = []
            for txt in texts:
                assert "<" != txt[0], "param[c_location=last] not support <only-part> mode input"
                new_txt = txt[2:] + txt[1] + txt[0] 
                re_location_texts.append(new_txt)

            texts = re_location_texts

        assert context_length, "Please set a valid context length in class"

        out_ids, out_mask = [], []
        out_str = []
        for text in texts:
            token_out = self.csv_encode(text, max_length=context_length)
            input_ids = token_out.input_ids 
            mask = token_out.mask
            input_str = token_out.input_str

            out_ids.append(input_ids)
            out_mask.append(mask)
            out_str.append(input_str)

        out_ids = torch.cat(out_ids, dim=0)     # [bs, seq_len]
        out_mask = torch.cat(out_mask, dim=0)   # [bs, seq_len]
        return SimpleNamespace(input_ids=out_ids,mask=out_mask, input_str=out_str)
        
if __name__ == "__main__":
    pass
    # tokenizer = FontTokenizer("token_dict.json")
    # tokenizer.encode("丞<$U氶(egesl)$D一(j)>", output_type=2)
    # tokenizer.encode("丞<$U氶(egesl)$D一(j)>", output_type=2, c_location="last")