# RASMALAI

This repository contains the code and experimental pipeline used for **data generation, preprocessing, and secret recovery** experiments on Learning With Errors (LWE) instances with **high-dimensional LWE (N = 1024)** and selected modulus(q) values: 26, 29, and 50.

## Repository Structure Overview

The project is divided into **three main stages**:

1. **Data Generation**
2. **Creation of Modified Training Matrix (`modified_A.npy`)**
3. **Secret Recovery and Analysis**

## 1. Data Generation

We use the official LWE dataset generation framework provided by Facebook Research:

- https://github.com/facebookresearch/LWE-benchmarking

All data generation and preprocessing steps follow the same instructions and commands provided in the official `README` of the LWE-benchmarking repository. We **only download and preprocess** datasets with the following parameters with Dimension: `N = 1024`, Modulus: `q = 26, 29, 50`

Once we have downloaded the data and unzipped it, please follow the exact commands provided below for creating the reduced LWE pairs(A,b) using the providedsecrets and preprocessed data.
Based on the dimension, secrettype, q value, and hamming weights, adjust the values in the command accordingly.

```bash
python3 src/generate/generate_A_b.py --processed_dump_path /path/used/to/store/preprocessed/data/ --secret_path ./n1024_logq26/binary_secrets_h3_40/secret.npy --dump_path /path/to/store/Ab/data/ --N 1024 --min_hamming 3 --max_hamming 40 --secret_type binary --num_secret_seeds 10 --rlwe 0 --actions secrets

python3 src/generate/generate_A_b.py --processed_dump_path /path/used/to/store/preprocessed/data/ --secret_path ./n1024_logq26/ternary_secrets_h3_40/secret.npy --dump_path /path/to/store/Ab/data/ --N 1024 --min_hamming 3 --max_hamming 40 --secret_type ternary --num_secret_seeds 10 --rlwe 0 --actions secrets

```
## 2. **Creation of Modified Training Matrix (`modified_A.npy`)**

Once we get the reduced LWE (A,b) pairs, we modify the matrix A before performing the attack. So next execute the generate_modified_A.py file which will give us a modified_A.npy

Run the following command, to create a modified_A.npy file

```bash
python generate_modified_A.py
```

we have data in three different folders for 3 q values and N=1024. We create the modified data in all of the folders.

## 3. Secret Recovery and Analysis 

Now we recover the secrets in two different settings i.e., binary and ternary. So every data folder has two different folders in it for binary and ternary. All the files that are inside the secret_recovery_codes_for_binary and secret_recovery_codes_for_ternary have to be executed for the secret recovery. 

Make sure all the paths are set to the corresponding results folder, and do the rest of the analysis.

Happy Coding with enough storage space.
