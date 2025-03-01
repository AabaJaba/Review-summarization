# Review Summarization using GPT-2  

## Overview  
This project implements review summarization using the GPT-2 model, fine-tuned on the **Amazon Fine Food Reviews** dataset. The goal is to generate concise summaries of user reviews while maintaining semantic accuracy. The model is trained using the **Hugging Face Transformers** library and evaluated using **ROUGE scores**.  

## Dataset  
- **Amazon Fine Food Reviews Dataset** from Kaggle  
- Used the **Text** and **Summary** columns for training  
- Dataset link: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  

## Features  
- Data preprocessing using **NLTK** and **BeautifulSoup**  
- Tokenization with **GPT-2 tokenizer**  
- Fine-tuning of **GPT-2** using **PyTorch and Transformers**  
- Model evaluation using **ROUGE metrics**  

## Installation  
To run the project, install the required dependencies:  
```bash
pip install transformers datasets torch nltk rouge-score
```

## Usage  

### 1. Preprocessing  
Run the preprocessing script to clean and tokenize the dataset:  
```python
python preprocess.py
```

### 2. Training  
Fine-tune GPT-2 on the dataset using:  
```python
python train.py
```

### 3. Evaluation  
Evaluate the model on the test set and compute ROUGE scores:  
```python
python evaluate.py
```

## Results  
The model is evaluated using **ROUGE-1, ROUGE-2, and ROUGE-L** scores. Example output:  

| Metric  | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| ROUGE-1 | 0.75      | 0.80   | 0.77     |
| ROUGE-2 | 0.50      | 0.67   | 0.57     |
| ROUGE-L | 0.67      | 0.75   | 0.71     |

## References  
- [Hugging Face GPT-2](https://huggingface.co/openai-community/gpt2)  
- [Fine-tuning GPT-2 Tutorial](https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners)  
- [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  
