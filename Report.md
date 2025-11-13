# Project Report:

## 1. Dataset Summary

**Dataset:** IMDb Movie Reviews (50,000 labeled reviews for sentiment analysis)

### Preprocessing Steps
- Conversion to lowercase  
- Removal of non-alphanumeric characters  
- Tokenization using whitespace separation  
- Padding and truncation applied to fixed sequence lengths of 25, 50, and 100 tokens  

### Statistics
| Metric | Value |
|---------|--------|
| Average Review Length | 230.20 words |
| Vocabulary Size | 180,586 unique tokens (restricted to 10,000 most frequent words for modeling) |

---

## 2. Model Configuration

| Parameter | Value |
|------------|--------|
| Embedding Dimension | 100 |
| Hidden Size | 64 |
| Number of Layers | 2 |
| Dropout | 0.3 |
| Batch Size | 32 |
| Epochs | 10 |
| Random Seed | 42 |
| Activation Functions | ReLU|
| Optimizers | Adam (lr=0.001), SGD (lr=0.01)|
| Sequence Lengths | 25, 50, 100 |
| Gradient Clipping | Tested both ON and OFF (max_norm = 1.0) |

---

## 3. Comparative Analysis

### Metrics Used
- **Accuracy:** Fraction of correctly classified samples  
- **F1-Score:** Harmonic mean of precision and recall  
- **Precision:** TP / (TP + FP)  
- **Recall:** TP / (TP + FN)  
- **Training Time:** Average time per epoch (in seconds)  

### Sample Results

| Model | Activation | Optimizer | Seq Length | Grad Clipping | Accuracy | F1 | Epoch Time (s) | Final Loss | Loss History |
|--------|-------------|------------|-------------|----------------|-----------|------|----------------|-------------|--------------|
| bidirectional_lstm | relu | adam | 25 | No | 0.7297 | 0.7297 | 60.15 | 0.3708 | [0.6810, 0.6044,..., 0.3708] |
| bidirectional_lstm | relu | adam | 25 | Yes | 0.7256 | 0.7251 | 50.81 | 0.3672 | [0.6787, 0.6116, ,..., 0.3672] |
| bidirectional_lstm | relu | adam | 50 | No | 0.7808 | 0.7808 | 85.50 | 0.3184 | [0.6937, 0.6637 ,...,  0.3184] |
| bidirectional_lstm | relu | adam | 50 | Yes | 0.7803 | 0.7803 | 78.19 | 0.3107 | [0.6890, 0.6151,..., 0.3107] |
| bidirectional_lstm | relu | adam | 100 | No | 0.8339 | 0.8339 | 147.39 | 0.2628 | [0.6925, 0.6457,...,  0.2628] |
| bidirectional_lstm | relu | adam | 100 | Yes | 0.8328 | 0.8327 | 150.33 | 0.2620 | [0.6923, 0.6647,...,  0.2620] |
| bidirectional_lstm | relu | sgd | 25 | No | 0.5116 | 0.5014 | 45.14 | 0.6935 | [0.6955, 0.6946,..., 0.6935] |
| bidirectional_lstm | relu | sgd | 25 | Yes | 0.5033 | 0.4908 | 45.36 | 0.6936 | [0.6950, 0.6944,..., 0.6936] |
| bidirectional_lstm | relu | sgd | 50 | No | 0.5006 | 0.5937 | 82.39 | 0.6935 | [0.6953, 0.6943,..., 0.6935] |
| bidirectional_lstm | relu | sgd | 50 | Yes | 0.4998 | 0.5982 | 81.83 | 0.6932 | [0.6956, 0.6943,..., 0.6932] |
| bidirectional_lstm | relu | sgd | 100 | No | 0.4993 | 0.6330 | 133.50 | 0.6934 | [0.6945, 0.6946,..., 0.6934] |
| bidirectional_lstm | relu | sgd | 100 | Yes | 0.4993 | 0.6330 | 129.83 | 0.6931 | [0.6946, 0.6939,..., 0.6931] |


*Full results are available in* `results/experiments_summary.xslx`.

### Charts
- **Accuracy and F1 vs. Sequence Length**
- <img width="796" height="500" alt="image" src="https://github.com/user-attachments/assets/ea46be05-3989-4c15-b1c5-4436fd657b02" />

- **Training Loss (Best vs. Worst Configurations)**  
<img width="889" height="555" alt="image" src="https://github.com/user-attachments/assets/9d3298fa-d360-4f04-a74f-cc2849ad4a3c" />

---

## 4. Discussion

### Best Configuration
The top-performing configuration achieved both the highest Accuracy and F1-Score:

| **Parameter**         | **Value**          |
| --------------------- | ------------------ |
| **Model**             | bidirectional_lstm |
| **Activation**        | ReLU               |
| **Optimizer**         | Adam               |
| **Sequence Length**   | 100                |
| **Gradient Clipping** | **No**             |
| **Accuracy**          | **0.8339**         |
| **F1 Score**          | **0.8339**         |
| **Final Loss**        | 0.2628             |
| **Epoch Time**        | 147.39 s           |


---

### Effect of Sequence Length
Longer sequence lengths (100 tokens) yielded superior performance as they captured richer contextual information from reviews.
However, this improvement came at the cost of increased training time.
Gradient clipping had minimal effect on performance in this setup.

### Effect of Optimizer
Adam consistently outperformed SGD in both accuracy and F1-score.
While Adam converged faster and more stably, SGD failed to effectively optimize the model under these settings.

### Effect of Gradient Clipping
Gradient clipping had negligible impact on performance in this study.
The Adam optimizer provided stable training even without clipping, suggesting that gradient explosion was not a major issue in these configurations.

---

## 5. Conclusion

### Optimal Configuration (Under CPU Constraints)

| **Parameter**         | **Value**          |
| --------------------- | ------------------ |
| **Model**             | bidirectional_lstm |
| **Activation**        | ReLU               |
| **Optimizer**         | Adam               |
| **Sequence Length**   | 100                |
| **Gradient Clipping** | **No**             |
| **Accuracy**          | **0.8339**         |
| **F1 Score**          | **0.8339**         |
| **Final Loss**        | 0.2628             |
| **Epoch Time**        | 147.39 s           |


### Justification
This configuration achieved the highest accuracy (0.8339) and F1-score (0.8339), with a training time of approximately 147 seconds per epoch.
It demonstrates the optimal balance between predictive performance and computational efficiency under CPU-based conditions, indicating stable convergence and effective learning behavior across epochs.

---
