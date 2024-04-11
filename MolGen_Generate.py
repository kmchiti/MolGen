from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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


def sf_encode(smile):
    try:
        encode = sf.encoder(smile)
        return encode
    except sf.EncoderError:
        return ''


# 1- generate molecules
num_samples = 30000
MOLECULAR_PERFORMANCE_RESULT_PATH = './molecular_performance.csv'
model_name_or_path = "zjunlp/MolGen-large-opt"

test_data = get_dataset('test')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).cuda()
model.eval()

pairs = {'input_smiles': [], 'cand_smiles': []}
for example in tqdm(test_data):
    example_sf = sf_encode(example)
    sf_input = tokenizer(example_sf, return_tensors="pt")
    molecules = model.generate(input_ids=sf_input["input_ids"].to('cuda'),
                               attention_mask=sf_input["attention_mask"].to('cuda'),
                               do_sample=True,
                               top_k=30,
                               top_p=1,
                               max_length=55,
                               min_length=13,
                               num_return_sequences=100)

    cand = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", "") for g in
            molecules]
    cand_smiles = [sf.decoder(selfies) for selfies in cand]
    cand_smiles = list(set(cand_smiles))
    input_smiles = [sf.decoder(example_sf) for i in range(len(cand_smiles))]
    pairs['input_smiles'] += input_smiles
    pairs['cand_smiles'] += cand_smiles
    if len(pairs['cand_smiles']) > num_samples:
        break
# 2- save generated molecules
save_path = f"{model_name_or_path.split('/')[-1]}-smiles.csv"
df = pd.DataFrame(pairs)
df.to_csv(save_path, index=None)


# 3- compute MOSES metrics
generated_smiles = df['cand_smiles']
metrics = moses.get_all_metrics(generated_smiles, n_jobs=10, device='cuda', batch_size=1024)
metrics_table = [[k, v] for k, v in metrics.items()]
print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="pretty"))
if os.path.exists(MOLECULAR_PERFORMANCE_RESULT_PATH):
    result = pd.read_csv(MOLECULAR_PERFORMANCE_RESULT_PATH, index_col=0)
else:
    result = pd.DataFrame()
df_new_row = pd.DataFrame(metrics, index=[model_name_or_path.split('/')[-1]])
result = pd.concat([result, df_new_row])
result.to_csv(MOLECULAR_PERFORMANCE_RESULT_PATH)

# 4- compute chemical property
n_jobs = 1
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

# Save the DataFrame to a CSV file
csv_file_path = f"{model_name_or_path.split('/')[-1]}-metrics.csv"  # You can change this to your desired path
df.to_csv(csv_file_path, index=False)
