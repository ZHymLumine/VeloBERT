from torch.utils.data import Dataset
from transformers import BertTokenizer
from enum import Enum
import os
from . import utils
import numpy as np
import torch

def load_data(path,nt=True,aa=True,protein=True,add_ss=True):

    #加载转换所需的词典 对应load_onehot
    aa_table = utils.aa_table
    codon2aa=utils.codon2aa
    SS = "-STIGEBH"
    nts = "ATCG"
    codon2idx = {}
    for nt1 in nts:
        for nt2 in nts:
            for nt3 in nts:
                codon2idx[nt1 + nt2 + nt3] = len(codon2idx)
    #print(codon2idx)
    onehot_nt = {nts[i]: np.eye(4)[i] for i in range(len(nts))}
    onehot_codon = {codon: np.eye(64)[codon2idx[codon]] for codon in codon2idx} # 4^3=64种密码子，one-hot编码
    onehot_aa = {aa_table[i, 2]: np.eye(21)[i] for i in range(len(aa_table))}
    onehot_ss={SS[i]: np.eye(len(SS))[i] for i in range(len(SS))}

    #读取文件 #load_data
    with open(path, "r") as f:
        data = f.readlines()
    assert len(data) % 4 == 0
    list_name = []
    list_seq = []
    list_speed = []
    list_pro_feature = []  # add protein feature1
    list_density = []
    list_ss=[]
    list_avg = []
    for i in range(len(data) // 4):
        name = data[4 * i + 0].split('>')[1].split()[0]
        seq = data[4 * i + 1].split()   # coden sequence
        count = [float(e) for e in data[4 * i + 2].split()] # velocity
        pro_feature = [list(map(float, e.split(",")[:-1])) for e in data[4 * i + 3].split()]# add protein feature
        ss=[e.split(",")[-1] for e in data[4 * i + 3].split()]  # structure
        avg = np.mean(np.array(count)[np.array(count) > 0.5])
        density = ((np.array(count) > 0.5) * np.array(count) / avg).tolist()

        # criteria containing ribosome density AND coverage percentage
        list_name.append(name)
        list_seq.append(seq)
        list_speed.append(count)
        list_density.append(density)
        list_pro_feature.append(pro_feature)  # add protein feature3 #1d=per gene, 2d=each codon, 3d=each feature
        list_ss.append(ss)
        list_avg.append(avg)

    # list_name = np.array(list_name,dtype=object)
    # list_pro_feature = np.array(list_pro_feature,dtype=object)  # add protein feature4
    # list_ss=np.array(list_ss,dtype=object)
    # list_seq = np.array(list_seq,dtype=object)
    # list_density = np.array(list_density,dtype=object)
    # list_avg = np.array(list_avg,dtype=object)
    # list_gene_all = list_name

    # length = 1170
    # print(f'list_name:{len(list_name)}\n')
    # print(f'list_pro_feature:{len(list_pro_feature)}\n')
    # print(f'list_ss:{len(list_ss)}\n')
    # print(f'list_seq:{len(list_seq)}\n')
    # print(f'list_density:{len(list_density)}\n')
    # print(f'list_avg:{len(list_avg)}\n')
    # print(f'list_gene_all:{len(list_gene_all)}\n')
   # examples = {"sequence": list_seq, "density": list_density}
    #print('density: ', list_density)
    return list_seq, list_density


class Split(Enum):
    train = "train"
    dev = "val"
    test = "test"




class GeneSequenceDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_seq_length, mode):
        self.tokenizer = tokenizer
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        sequences, speed = load_data(file_path)
        self.features = convert_examples_to_features(sequences, speed, tokenizer, max_seq_length)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        return features


def convert_examples_to_features(
    sequences, 
    speed, 
    tokenizer, 
    max_seq_length,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True
    ):
    features = []
    for tokens, labels in zip(sequences, speed):

        special_tokens_count = 2
        # Truncate or pad the tokenized sequence and speeds list
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            labels = labels[:(max_seq_length - special_tokens_count)]
        
        tokens += [sep_token]
        labels += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        # print("len label: ", len(labels))
        if cls_token_at_end:
            tokens += [cls_token]
            labels += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            labels = [pad_token_label_id] + labels
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            labels = ([pad_token_label_id] * padding_length) + labels
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            labels += [pad_token_label_id] * padding_length


        valid_sequence = (torch.tensor(labels) != -100).float()
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels) == max_seq_length

        # print('labels:', labels)
        features.append({
            'input_ids': torch.tensor(input_ids), 
            'attention_mask': torch.tensor(input_mask), 
            'token_type_ids': torch.tensor(segment_ids), 
            'density': torch.tensor(labels), 
            'valid_sequence': valid_sequence.clone().detach().requires_grad_(False)
        })
        # print('input_ids: ', type(input_ids))
        # print('attention_mask: ', type(input_mask))
        # print('token_type_ids: ', type(segment_ids))
        # print('density: ', type(labels))
        # print('valid_sequence:', valid_sequence)
    return features
