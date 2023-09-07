import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing
import os, datetime
from Bio import pairwise2
import pickle


def get_prottrans(fasta_file,output_path):
    num_cores = 2
    multiprocessing.set_start_method("forkserver")
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = str, default = '0')
    args = parser.parse_args()
    gpu = args.gpu

    ID_list = []
    seq_list = []
    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
            if line[0] == ">":
                ID_list.append(line[1:-1])
            else:
                seq_list.append(" ".join(list(line.strip())))

    model_path = "./Prot-T5-XL-U50"
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.eval()
    model = model.cuda()
    print(next(model.parameters()).device)
    print('starttime')
    starttime = datetime.datetime.now()
    print(starttime)
    batch_size = 1

    for i in tqdm(range(0, len(ID_list), batch_size)):
        if i + batch_size <= len(ID_list):
            batch_ID_list = ID_list[i:i + batch_size]
            batch_seq_list = seq_list[i:i + batch_size]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq_list[i:]
        

        # Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        # Tokenize, encode sequences and load it into the GPU if possibile
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Extracting sequences' features and load it into the CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            np.save(output_path + "/" + batch_ID_list[seq_num], seq_emd)
            endtime = datetime.datetime.now()
            print('endtime')
            print(endtime)
    

def get_pdb_xyz(pdb_file,ref_seq):
    current_pos = -1000
    X = []
    current_aa = {} # 'N', 'CA', 'C', 'O'
    try:
        for line in pdb_file:
            if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
                if current_aa != {}:
                    X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"]])
                    current_aa = {}
                if line[0:4].strip() != "TER":
                    current_pos = int(line[22:26].strip())

            if line[0:4].strip() == "ATOM":
                atom = line[13:16].strip()
                if atom in ['N', 'CA', 'C', 'O']:
                    xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                    current_aa[atom] = xyz
    except:
        return None
    if len(X) == len(ref_seq):          
        return np.array(X)
    else:
        return None
    
def get_esmfold(fasta_file,output_path):
    os.system('python ./esmfold/esmfold.py -i {fasta} -o {output} --chunk-size 128'.format(fasta=fasta_file,output=output_path))
    pdbfasta = {}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            name = fasta_ori[i].split('>')[1].replace('\n','')
            seq = fasta_ori[i+1].replace('/n','')
            pdbfasta[name] = seq
    for key in pdbfasta.keys():
        coord = get_pdb_xyz(output_path + '/' + key + '.pdb')
        np.save(output_path + '/' + key + '.npy', coord)

    

def get_dssp(fasta_file, pdb_path, dssp_path):
    DSSP = './dssp'
    def process_dssp(dssp_file):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()

        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            aa = lines[i][13]
            if aa == "!" or aa == "*":
                continue
            seq += aa
            SS = lines[i][16]
            if SS == " ":
                SS = "C"
            SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
            SS_vec[SS_type.find(SS)] = 1
            PHI = float(lines[i][103:109].strip())
            PSI = float(lines[i][109:115].strip())
            ACC = float(lines[i][34:38].strip())
            ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
            dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

        return seq, dssp_feature

    def match_dssp(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
        SS_vec[-1] = 1
        padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

        new_dssp = []
        for aa in seq:
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0))

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp

    def transform_dssp(dssp_feature):
        dssp_feature = np.array(dssp_feature)
        angle = dssp_feature[:,0:2]
        ASA_SS = dssp_feature[:,2:]

        radian = angle * (np.pi / 180)
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

        return dssp_feature

    def get_dssp(data_path,dssp_path, ID, ref_seq):
        try:
            os.system("{} -i {}.pdb -o {}.dssp".format(DSSP, data_path + ID, dssp_path + ID))

            dssp_seq, dssp_matrix = process_dssp(dssp_path + ID + ".dssp")
            if dssp_seq != ref_seq:
                dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)
            np.save(dssp_path + ID + "_dssp.npy", transform_dssp(dssp_matrix))
            os.system('rm {}.dssp'.format(dssp_path + ID))
            return 0
        except Exception as e:
            print(e)
            return None
    pdbfasta = {}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            name = fasta_ori[i].split('>')[1].replace('\n','')
            seq = fasta_ori[i+1].replace('\n','')
            pdbfasta[name] = seq

    fault_name = []
    for name in pdbfasta.keys():
        sign = get_dssp(pdb_path,dssp_path, name ,pdbfasta[name])
        if sign == None:
            fault_name.append(name)
    if fault_name != []:
        np.save('../Example/structure_data/dssp_fault.npy',fault_name)

