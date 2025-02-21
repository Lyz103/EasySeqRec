# EasySeqRec 🚀

**EasySeqRec** is an efficient toolkit designed for quickly running and benchmarking state-of-the-art sequence recommendation models. This project integrates powerful baselines like **SASRec** (ICDE 2018), **GRU4Rec** (ICLR 2016), **LinRec** (SIGIR 2023), **Mamba4Rec** (RelKD), **LightSans** (SIGIR 2021), and **FMLP4Rec** (WWW 2021). With **EasySeqRec**, you can run all six models on multiple datasets with a single command, eliminating the need for manual execution of each model. Results are automatically saved in an Excel file, which can be further processed with a provided script to convert the data into LaTeX format for direct inclusion in academic papers. 📊

## Features ✨

- **Integrated Baselines**: Run multiple advanced models (SASRec, GRU4Rec, LinRec, Mamba4Rec, LightSans, FMLP4Rec) with minimal configuration. 
- **One-click Execution**: Execute all six models across various datasets simultaneously. 
- **Automated Result Collection**: Results are directly saved to an Excel file, removing the need for manual tracking and computation. 
- **LaTeX Output**: Convert Excel results into LaTeX format with a provided script for seamless paper integration. 
- **Flexible Dataset Handling**: Easily process datasets and run experiments with customized configurations.

## Requirements 📋

To run this project, ensure that the following dependencies are installed:

- `mamba_ssm`: 1.1.1
- `matplotlib`: 3.8.3 or newer
- `numpy`: 1.24.1 or newer
- `torch`: 2.1.1+cu118 or newer
- `tqdm`: 4.66.1 or newer

You can install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Usage 💻

### 1️⃣ Data Preprocessing

For **Amazon** datasets, you can preprocess the data using the `dataprocess.py` script. 

For custom datasets, make sure to organize them in the following format and place them in the `data/` directory:

```txt
uid Seq
1 29 38 28 9 82 92 33 78
```

Where:
- `1` is the `uid` (user identifier),
- `Seq` represents the sequence of items, with the last two items (`33` and `78`) used for validation and testing, respectively.

### 2️⃣ Running the Models

Once your data is ready, you can run all six models across multiple datasets with a single command:

```bash
bash run_finetune.bash
```

You can modify the parameters and experiment with different configurations by editing the `run_finetune.bash` file.

### 3️⃣ Results

After running the models, the results will be saved in an Excel file (`Baseline_Full.xlsx`). For datasets like **Beauty**, running all six models on an NVIDIA RTX 3090 takes approximately **20 minutes**. Once the execution is complete, you can analyze the results directly in the Excel file.

Additionally, we provide a script to convert the Excel results into LaTeX format, making it easy to integrate into academic papers.

## Acknowledgements 🙏

- The code for this project is adapted from **S3Rec**. Special thanks to the original authors for their contributions. 
- The model implementations are based on **RecBole**. Thanks to the RecBole team for providing an excellent framework for sequence recommendation research. However, due to the high coupling in RecBole, we have modified the framework to allow better reuse by the sequence recommendation research community.

## Future Improvements 🚀

We plan to enhance the toolkit further with the following features:

1. **Automatic Dataset Download and Processing**: Simplified handling for datasets like Amazon.
2. **LLM Agent for Result Analysis**: Automate the analysis of model results using large language models.
3. **Automated Result Plotting**: Utilize LLM agents to generate plots from results automatically.
