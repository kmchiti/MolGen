# Pretrain GPTs on Chemical SMILES Strings

This project aims to pretrain the GPT family model from Hugging Face's transformers library on chemical SMILES strings.


## Usage

To use the project, follow these steps:

1. Clone the repository: `git clone https://github.com/chandar-lab/MolGen`
2. Install the dependencies: `source scripts/install_requirments.sh`
3. Prepare the data: `python prepare_data.py --config-name=pubchem_smiles`
4. Run the training script: `python train.py --config-name=config_PubChem`

The dependencies require `python 3.10`, `CUDA 11.8`, and `gcc 9.3.0` to be installed .

## License

This project is licensed under the MIT License - see the LICENSE file for details.
