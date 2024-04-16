from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from moses.metrics.utils import logP, QED, SA, weight, get_mol
from moses.utils import mapper
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import selfies as sf
import moses
from moses.dataset import get_dataset
from tabulate import tabulate
import os
import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser(
        description='Molecules Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name_or_path', default="zjunlp/MolGen-large-opt", type=str,
                        help='model name or path to load checkpoint from HF')
    parser.add_argument('--input_data', default="MOSES", type=str, help='input data to condition and generate for MolGen')
    parser.add_argument('--num_samples', default=30000, type=int, help='number of samples to generate')
    parser.add_argument('--num_return_sequences', default=200, type=int, help='number of return sequence')
    parser.add_argument('--max_length', default=55, type=int, help='max length')
    parser.add_argument('--min_length', default=13, type=int, help='min length')
    parser.add_argument('--top_k', default=30, type=int, help='top_k')
    parser.add_argument('--top_p', default=1.0, type=float, help='top_p')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature')
    parser.add_argument('--prompt', default="CC", type=str, help='input prompt to generate')
    args = parser.parse_args()
    return args

def sf_encode(smile):
    try:
        encode = sf.encoder(smile)
        return encode
    except sf.EncoderError:
        return ''

def dataset_to_selfie(input_data_path="../moldata/finetune/np_test.csv"):
    from pandarallel import pandarallel
    input_data = pd.read_csv(input_data_path)
    pandarallel.initialize(shm_size_mb=60720, nb_workers=20, progress_bar=True)
    if 'selfies' not in input_data.columns.tolist():
        print('convert smiles to selfies ...')
        smiles_name = input_data.keys()[0]
        input_data['selfies'] = input_data[smiles_name].parallel_apply(sf_encode)
        input_data.to_csv(input_data_path, index=None)


def generate(model, tokenizer, num_samples=30000, num_return_sequences=256, no_repeat_ngram_size=2, max_length=64,
             prompt="CC", device=torch.device('cuda'), top_k=50, top_p=0.95, temperature=1.0,):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # skip eos token
    input_ids = input_ids[:, :-1]
    pairs = {'cand_smiles': []}
    for i in tqdm(range(num_samples // num_return_sequences + 1)):
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        pairs['cand_smiles'] += [s.replace(" ", "") for s in output]
    return pairs

def MolGen_conditional_generate(model, tokenizer, test_data, num_samples=30000, num_return_sequences=200, max_length=55,
                                min_length=13, device=torch.device('cuda'), top_k=30, top_p=1, temperature=1.0,):

    pairs = {'input_smiles': [], 'cand_smiles': []}
    for example in tqdm(test_data):
        example_sf = sf_encode(example)
        sf_input = tokenizer(example_sf, return_tensors="pt")
        molecules = model.generate(input_ids=sf_input["input_ids"].to(device),
                                   attention_mask=sf_input["attention_mask"].to(device),
                                   do_sample=True,
                                   top_k=top_k,
                                   top_p=top_p,
                                   max_length=max_length,
                                   min_length=min_length,
                                   temperature=temperature,
                                   num_return_sequences=num_return_sequences)

        cand = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", "") for g
                in
                molecules]
        cand_smiles = [sf.decoder(selfies) for selfies in cand]
        cand_smiles = list(set(cand_smiles))
        input_smiles = [sf.decoder(example_sf) for i in range(len(cand_smiles))]
        pairs['input_smiles'] += input_smiles
        pairs['cand_smiles'] += cand_smiles
        if len(pairs['cand_smiles']) > num_samples:
            break
    return pairs


def compute_MOSES_metrics(generated_smiles, model_name, save_path, n_jobs=1, batch_size=1024, device='cuda'):

    metrics = moses.get_all_metrics(generated_smiles, n_jobs=n_jobs, device=device, batch_size=batch_size)
    metrics_table = [[k, v] for k, v in metrics.items()]
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="pretty"))
    if os.path.exists(save_path):
        result = pd.read_csv(save_path, index_col=0)
    else:
        result = pd.DataFrame()
    df_new_row = pd.DataFrame(metrics, index=[model_name])
    result = pd.concat([result, df_new_row])
    result.to_csv(save_path)

def compute_chemical_prop(generated_smiles, n_jobs=1):
    pool = Pool(n_jobs)
    mols = mapper(pool)(get_mol, generated_smiles)

    generated_logP = []
    generated_QED = []
    generated_SA = []
    generated_weight = []

    for mol in mols:
        if mol is None:
            continue
        generated_logP.append(logP(mol))
        generated_QED.append(QED(mol))
        generated_SA.append(SA(mol))
        generated_weight.append(weight(mol))

    df = pd.DataFrame({
        'logP': generated_logP,
        'QED': generated_QED,
        'SA': generated_SA,
        'weight': generated_weight
    })
    return df


if __name__ == "__main__":
    # 0- load model and data
    args = args_parser()
    model_name = args.model_name_or_path.split('/')[-1]
    if args.input_data == "MOSES":
        test_data = get_dataset('test')
        MOLECULAR_PERFORMANCE_RESULT_PATH = './molecular_performance_MOSES.csv'
    elif args.input_data == "np":
        input_data = pd.read_csv("../moldata/finetune/np_test.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.model_name_or_path.startswith("MolGen/"):
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).cuda()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).cuda()
    model.eval()
    # 1- generate molecules
    print("=============== start generating molecules ===============")
    if args.model_name_or_path.startswith("MolGen/"):
        pairs = generate(model, tokenizer, num_samples=30000, num_return_sequences=256, max_length=64, prompt="CC",
                         top_k=50, top_p=0.95, temperature=1.0, )
    else:
        pairs = MolGen_conditional_generate(model, tokenizer, test_data, num_samples=30000, num_return_sequences=200,
                                            max_length=55, min_length=13, top_k=30, top_p=1, temperature=1.0,)
    # 2- save generated molecules
    print("=============== save generated molecules ===============")
    save_path = f"{model_name}-smiles.csv"
    df = pd.DataFrame(pairs)
    df.to_csv(save_path, index=None)

    # 3- compute MOSES metrics
    print("=============== compute MOSES metrics ===============")
    compute_MOSES_metrics(generated_smiles=df['cand_smiles'], model_name=model_name,
                          save_path=MOLECULAR_PERFORMANCE_RESULT_PATH, n_jobs=1, batch_size=1024, device='cuda')

    # 4- compute chemical property
    print("=============== compute chemical property ===============")
    compute_chemical_prop(generated_smiles=df['cand_smiles'], n_jobs=1)
    # Save the DataFrame to a CSV file
    csv_file_path = f"{model_name}-metrics.csv"  # You can change this to your desired path
    df.to_csv(csv_file_path, index=False)

    print("=============== FINISH ===============")
