# Report

## 1. Overview

I evaluated six classifier models trained on a news classification dataset. Three models used contextualized ELMo embeddings (with different ways of combining the three ELMo outputs), while three models relied on static word embeddings generated via CBOW, Skipgram, and SVD techniques. For each model, I reported training and test accuracy, precision, recall, F1 scores, and the corresponding confusion matrices. The goal is to determine which embedding method performs best and to understand the impact of hyperparameter settings on overall performance.

## 2. Experimental Results

### 2.1. ELMo-based Classifiers

The three ELMo models were defined as follows:
- **FrozenLambdaElmoClassifier**  
- **TrainableLambdaElmoClassifier**  
- **LearnableFunctionElmoClassifier**

**Metrics Summary (Test Set):**

| Model      | Test Accuracy | Precision | Recall | F1 Score |
|------------|---------------|-----------|--------|----------|
| Frozen     | 84.50%        | 84.77%    | 84.50% | 84.55%   |
| Trainable  | 85.01%        | 84.99%    | 85.01% | 84.99%   |
| Learnable  | 85.82%        | 85.87%    | 85.82% | 85.78%   |

**Confusion Matrices (Test Set):**

- **Frozen:**  
  ```
  [[1600   80   84  136]
   [  38 1760   24   78]
   [  72   34 1490  304]
   [  65   65  198 1572]]
  ```
- **Trainable:**  
  ```
  [[1615   83  107   95]
   [  64 1756   40   40]
   [  98   41 1525  236]
   [  96   64  175 1565]]
  ```
- **Learnable:**  
  ```
  [[1639   79   80  102]
   [  38 1798   22   42]
   [  76   46 1501  277]
   [  85   62  169 1584]]
  ```

### 2.2. Static Embedding Classifiers

The static embedding models were built using different pretrained representations:
- **CBOWClassifier**  
- **SkipgramClassifier**  
- **SVDClassifier**

**Metrics Summary (Test Set):**

| Model    | Test Accuracy | Precision | Recall | F1 Score |
|----------|---------------|-----------|--------|----------|
| CBOW     | 78.75%        | 78.82%    | 78.75% | 78.56%   |
| Skipgram | 83.33%        | 83.38%    | 83.33% | 83.35%   |
| SVD      | 64.38%        | 65.88%    | 64.38% | 64.27%   |

**Confusion Matrices (Test Set):**

- **CBOW:**  
  ```
  [[1442  174  163  121]
   [  69 1739   31   61]
   [  99   73 1525  203]
   [ 123  146  352 1279]]
  ```
- **Skipgram:**  
  ```
  [[1598   88  106  108]
   [  80 1704   55   61]
   [  106   31 1540  223]
   [  100   53  256 1491]]
  ```
- **SVD:**  
  ```
  [[1116  239  153  392]
   [ 169 1494   15  222]
   [ 236   87  940  637]
   [  176  147  234 1343]]
  ```

### Results
![alt text](<Screenshot 2025-03-31 at 2.00.35 AM.png>) <br>
![alt text](<Screenshot 2025-03-31 at 2.01.27 AM.png>)<br>
![alt text](<Screenshot 2025-03-31 at 2.01.43 AM.png>) <br>
![alt text](<Screenshot 2025-03-31 at 2.01.55 AM.png>)<br>

## 3. Ranking and Performance Analysis

### 3.1. Overall Ranking (Test Accuracy)

1. **LearnableFunctionElmoClassifier (85.82%)**  
2. **TrainableLambdaElmoClassifier (85.01%)**  
3. **FrozenLambdaElmoClassifier (84.50%)**  
4. **SkipgramClassifier (83.33%)**  
5. **CBOWClassifier (78.75%)**  
6. **SVDClassifier (64.38%)**

### 3.2. Why Contextualized ELMo Models Outperform Static Methods

- **Context Sensitivity:**  
  ELMo embeddings are generated by a deep bidirectional LSTM and adapt based on surrounding context. This leads to richer semantic representations compared to static embeddings, which assign the same vector to a word regardless of context.

- **Dynamic Combination Strategies:**  
  - **LearnableFunctionElmoClassifier:**  
    The MLP-based combiner allows the model to learn complex, nonlinear interactions among the three ELMo representations at a token level. This flexibility enables the network to adjust the contribution of each representation depending on the input context, leading to the highest overall performance.
  
  - **TrainableLambdaElmoClassifier:**  
    This model learns scalar weights (via a softmax) to combine the outputs, offering a moderate degree of adaptability. Although less flexible than the MLP approach, it still improves over fixed combination.
  
  - **FrozenLambdaElmoClassifier:**  
    Uses fixed randomly initialised weights for all three outputs. This static approach does not allow for dynamic adaptation and is slightly less effective than the trainable alternatives.

- **Static Embeddings:**  
  While Skipgram achieves relatively high performance among static methods, both CBOW and SVD lag behind. Their inability to capture context nuances (and, in the case of SVD, possible oversimplifications due to linear approximations) results in lower performance.

## 4. Best Hyperparameter Settings and Their Impact

**Hyperparameters:**
- **Combiner:** Learnable Function
- **Embedding Dimension:** 100  
- **Hidden Dimension:** 256  
- **Number of RNN Layers:** 2  
- **Batch Size:** 32  
- **Learning Rate:** 1e-3  
- **Epochs:** 10

**Justification & Impact:**

- **Embedding and Hidden Dimensions (100 & 256):**  
  These dimensions strike a balance between representational power and computational efficiency. In ELMo-based models, the chosen dimensions allow sufficient capacity to capture context-dependent features without overfitting. For static embeddings, these dimensions are standard and yield competitive performance when using methods like Skipgram.

- **Number of RNN Layers (2):**  
  Two layers provide additional depth to capture complex sequential patterns while still being shallow enough to train effectively. For the learnable ELMo classifier, this depth allows the MLP combiner to integrate information efficiently before feeding into the LSTM.

- **Batch Size (32) and Learning Rate (1e-3):**  
  A batch size of 32 offers stable gradient updates and a good balance between convergence speed and stability. The learning rate of 1e-3 (with the Adam optimizer) has proven effective for both ELMo-based and static models, leading to robust convergence.

- **Epochs (10):**  
  Training for 10 epochs provided enough iterations for the models to learn the task while preventing overfitting. The confusion matrices indicate that the models generalize well, with relatively low misclassification rates—especially for the learnable ELMo variant.

**Why the Learnable Model Works Best:**  
The learnable function approach employs an MLP to combine the three ELMo outputs. This design allows the network to learn a more sophisticated representation of the input by emphasizing or de-emphasizing different aspects of the context, which results in improved performance. The trainable model, while still learning weights, is less flexible since it only adjusts scalar multipliers. The frozen model, using fixed weights, does not adapt to the nuances in the data, leading to slightly lower performance.

**Confusion Matrix Impact:**  
- **ELMo Models:**  
  The confusion matrices for the ELMo classifiers show fewer misclassifications along the off-diagonals, particularly in the learnable model. This indicates that dynamic combination (via the MLP) helps the classifier better distinguish between similar classes.
  
- **Static Models:**  
  The Skipgram model has a relatively balanced confusion matrix, but CBOW and especially SVD show more pronounced misclassifications, suggesting that the static methods’ limitations in context sensitivity directly affect class separation.

## 5. Conclusion

The experiments demonstrate that contextualized ELMo embeddings, especially when combined via a learnable MLP function, outperform static embedding methods in news classification. The dynamic combination offered by the learnable model enables it to adapt to nuanced contextual cues in the input text, resulting in superior performance compared to the trainable (scalar) and frozen (fixed) methods. Among the static embeddings, Skipgram is the best, yet still lags behind the ELMo approaches. The chosen hyperparameter settings—embedding dimension 100, hidden dimension 256, 2 LSTM layers, batch size 32, learning rate 1e-3, and 10 epochs—provided an effective trade-off between model complexity and training stability, as evidenced by both the quantitative metrics and the analysis of the confusion matrices.

[Pretrained Models](https://drive.google.com/drive/folders/1tQ4_ZICrcfy2tGOiLSZ0aMwYwJp5TP1k?usp=drive_link)