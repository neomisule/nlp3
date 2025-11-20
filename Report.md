# Project Report:
## 1. Dataset Summary
Dataset: IMDb Movie Reviews (50,000 labeled reviews for sentiment analysis)

Preprocessing Steps
-	Conversion to lowercase
-	Removal of non-alphanumeric characters
-	Tokenization using whitespace separation
-	Padding and truncation applied to fixed sequence lengths of 25, 50, and 100 tokens

## Statistics

| Metric | Value |
|--------|-------|
| Average Review Length | 230.20 words |
| Vocabulary Size | 180,586 unique tokens (restricted to 10,000 most frequent words for modeling) |

## 2. Model Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 100 |
| Hidden Size | 64 |
| Number of Layers | 2 |
| Dropout | 0.3 |
| Batch Size | 32 |
| Epochs | 10 |
| Random Seed | 42 |
| Activation Functions | ReLU |
| Optimizers | Adam (lr=0.001), SGD (lr=0.01) |
| Sequence Lengths | 25, 50, 100 |
| Gradient Clipping | Tested both ON and OFF (max_norm = 1.0) |

## 3. Comparative Analysis
Metrics Used
-	Accuracy: Fraction of correctly classified samples
-	F1-Score: Harmonic mean of precision and recall
-	Precision: TP / (TP + FP)
-	Recall: TP / (TP + FN)
-	Training Time: Average time per epoch (in seconds)

Sample Results

| Model | Activation | Optimizer | Seq_Length | Grad_Clip | Accuracy | F1 | Epoch_Time | Final_Loss | History_Train_Loss | History_Val_Loss | Checkpoint |
|-------|------------|-----------|------------|-----------|----------|-------|------------|------------|-------------------|------------------|------------|
| bidirectional | relu | adam | 25 | No | 0.7297 | 0.7297 | 60.15 | 0.3708 | [0.6810159786125584, 0.6...] | [0.6334604713922877, 0.5745707...] | models\best_20251120_135836.pth |
| bidirectional | relu | adam | 25 | Yes | 0.7256 | 0.7251 | 50.81 | 0.3672 | [0.6787419174333362, 0.6...] | [0.608665300566522, 0.56448310...] | models\best_20251120_140129.pth |
| bidirectional | relu | adam | 50 | No | 0.7808 | 0.7808 | 85.5 | 0.3184 | [0.6936756113301152, 0.6...] | [0.6503922570773097, 0.5745214...] | models\best_20251120_140631.pth |
| bidirectional | relu | adam | 50 | Yes | 0.7803 | 0.7803 | 78.19 | 0.3107 | [0.6890190253629709, 0.6...] | [0.6271090149296989, 0.5575814...] | models\best_20251120_141053.pth |
| bidirectional | relu | adam | 100 | No | 0.8339 | 0.8339 | 147.39 | 0.2628 | [0.6925153268115295, 0.6...] | [0.6922784123731696, 0.6912066...] | models\best_20251120_142129.pth |
| bidirectional | relu | adam | 100 | Yes | 0.8328 | 0.8327 | 150.33 | 0.262 | [0.6922965548989718, 0.6...] | [0.6223766079644109, 0.5205591...] | models\best_20251120_143127.pth |
| bidirectional | relu | sgd | 25 | No | 0.5116 | 0.5014 | 45.14 | 0.6935 | [0.6955411792411219, 0.6...] | [0.6932571393906918, 0.6931903...] | models\best_20251120_143626.pth |
| bidirectional | relu | sgd | 25 | Yes | 0.5033 | 0.4908 | 45.36 | 0.6936 | [0.6949710169106799, 0.6...] | [0.6932853111220748, 0.6932335...] | models\best_20251120_143907.pth |
| bidirectional | relu | sgd | 50 | No | 0.5006 | 0.5937 | 82.39 | 0.6935 | [0.6953026182816157, 0.6...] | [0.693277728679954, 0.693225478...] | models\best_20251120_144337.pth |
| bidirectional | relu | sgd | 50 | Yes | 0.4998 | 0.5982 | 81.83 | 0.6932 | [0.6956035690691177, 0.69...] | [0.6934727455496483, 0.6933727...] | models\best_20251120_144804.pth |
| bidirectional | relu | sgd | 100 | No | 0.4993 | 0.633 | 133.5 | 0.6934 | [0.6944846388933908, 0.6...] | [0.6933967817165053, 0.6932923...] | models\best_20251120_145546.pth |
| bidirectional | relu | sgd | 100 | Yes | 0.4993 | 0.633 | 129.83 | 0.6931 | [0.6945810824861307, 0.6...] | [0.6931494059007796, 0.6930770...] | models\best_20251120_150522.pth |

Full results are available in results/experiments_summary.csv

Charts
- Accuracy and F1 vs. Sequence Length
- <img width="940" height="591" alt="image" src="https://github.com/user-attachments/assets/b15d9db6-3500-431b-8ee8-1a45b0ca881b" />

- Training Loss (Best vs. Worst Configurations)
- <img width="940" height="587" alt="image" src="https://github.com/user-attachments/assets/5c2e6df3-bfc1-49ab-ad9b-04ef323728e9" />


## 4. Discussion

Best Configuration

The top-performing configuration achieved both the highest Accuracy and F1-Score:

| Parameter | Value |
|-----------|-------|
| Model | bidirectional_lstm |
| Activation | ReLU |
| Optimizer | Adam |
| Sequence Length | 100 |
| Gradient Clipping | No |
| Accuracy | 0.8339 |
| F1 Score | 0.8339 |
| Final Loss | 0.2628 |
| Epoch Time | 147.39 s |

Effect of Sequence Length

Longer sequence lengths (100 tokens) yielded superior performance as they captured richer contextual information from reviews. However, this improvement came at the cost of increased training time. Gradient clipping had minimal effect on performance in this setup.

Effect of Optimizer

Adam consistently outperformed SGD in both accuracy and F1-score. While Adam converged faster and more stably, SGD failed to effectively optimize the model under these settings.

Effect of Gradient Clipping

Gradient clipping had negligible impact on performance in this study. The Adam optimizer provided stable training even without clipping, suggesting that gradient explosion was not a major issue in these configurations.

## 5. Conclusion
Optimal Configuration (Under CPU Constraints)

| Parameter | Value |
|-----------|-------|
| Model | bidirectional_lstm |
| Activation | ReLU |
| Optimizer | Adam |
| Sequence Length | 100 |
| Gradient Clipping | No |
| Accuracy | 0.8339 |
| F1 Score | 0.8339 |
| Final Loss | 0.2628 |
| Epoch Time | 147.39 s |

Justification

This configuration achieved the highest accuracy (0.8339) and F1-score (0.8339), with a training time of approximately 147 seconds per epoch. It demonstrates the optimal balance between predictive performance and computational efficiency under CPU-based conditions, indicating stable convergence and effective learning behavior across epochs.
