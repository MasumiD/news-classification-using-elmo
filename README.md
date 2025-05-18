# News Classification Inference

This repository contains an inference script for a news classification task. Given a pretrained model (one of six available models) and a news article description, the script predicts the class probabilities for the input article.

## Files

- **inference.py**: The main inference script. It loads a pretrained classifier (ELMo-based or static embedding-based), tokenizes the input description, and outputs class probabilities.
- **model definitions**: The required model classes (ELMo-based and static classifiers) are included in the script. The script assumes that the `ELMoBiLM` class is defined and available in the environment.
- **Tokenizer and utility functions**: Tokenization and preprocessing utilities are provided within the script.

## Requirements

- Python 3.6 or later
- PyTorch
- NLTK (and required resources; e.g. `punkt`)
- Other standard libraries: `sys`, `os`, `re`, `pickle`

## Pretrained Model Files

The inference script requires access to several pretrained model files:
- **For ELMo-based models:**
  - Vocabulary file: e.g. `/kaggle/input/idk/pytorch/default/1/vocab.pkl`
  - BiLSTM weights for the ELMo model: e.g. `/kaggle/input/idk/pytorch/default/1/bilstm.pt`
  - Classifier weights: Filename should include keywords such as `frozen`, `trainable`, or `learnable` (e.g. `classifier_learnable.pt`).

- **For Static embedding models:**
  - Static embedding model files (e.g. CBOW, Skipgram, SVD): e.g. `/kaggle/input/a3_models/pytorch/default/1/cbow.pt`, etc.
  - Classifier weights: Filename should include keywords such as `cbow`, `skipgram`, or `svd` (e.g. `classifier_skipgram.pt`).

Make sure the file paths in the script match your file structure.

## Usage

Run the inference script from the command line using:

```bash
python inference.py <saved_model_path> "<news description>"
```

For example:

```bash
python inference.py /path/to/classifier_learnable.pt "Breaking news: A major event has just occurred in the city center..."
```

The script will:
1. Determine the model type (ELMo-based or static) based on the filename.
2. Load the corresponding pretrained weights and support files (vocabulary, BiLSTM or static embedding model).
3. Tokenize and preprocess the provided news article description.
4. Output the predicted probabilities for each class in the following format:

```
class-1 0.6000
class-2 0.2000
class-3 0.1000
class-4 0.1000
```

## Implementation Assumptions

- **Model Consistency:**  
  The script assumes that the model architecture defined in the code is identical to the architecture used when saving the model weights. Any changes between saving and inference may result in errors.

- **File Paths:**  
  The file paths for the pretrained ELMo model, vocabulary, and static embedding models are hardcoded in the script. Ensure these paths match your environment (e.g., Kaggle paths or local file system paths).

- **ELMoBiLM Availability:**  
  The `ELMoBiLM` class is assumed to be available in the runtime environment (either defined elsewhere or imported). Without this, the ELMo-based models will not work.

- **Security Warning:**  
  The script uses `torch.load` with the default settings. If using untrusted model files, please consider setting `weights_only=True` to avoid potential security risks as indicated in the PyTorch warnings.

- **Environment:**  
  The code is developed and tested in an environment where CUDA is available (if applicable). Otherwise, the CPU will be used.

## Pretrained Models can be found here:
[Pretrained Models](https://drive.google.com/drive/folders/1tQ4_ZICrcfy2tGOiLSZ0aMwYwJp5TP1k?usp=drive_link)