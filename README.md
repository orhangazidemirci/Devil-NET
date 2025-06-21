# Devil-NET

**Devil-NET** is a three-stage debiasing framework designed to mitigate spurious correlations in image classification tasks by leveraging multiple overfitted "bias models" and a final contrastive learning step.

## 🔧 Project Structure

The pipeline consists of three main Python scripts corresponding to each stage of the framework:

### Stage 1: Train Bias Models
**File:** `bias_models_new.py`  
- Partitions the dataset into *N* disjoint shards.  
- Trains *N* separate bias models, each on one partition.

### Stage 2: Generate Unbiased Logits
**File:** `bias_data_combined.py`  
- Passes each partition through the *N−1* bias models that were not trained on it.  
- Aggregates the logits and reconstructs soft pseudo-labels for the full dataset.

### Stage 3: Train Final Unbiased Model
**File:** `Devil-NET_train.py`  
- Trains the final “unbiased” model using the reconstructed dataset.  
- Includes baseline training for comparison (e.g., ResNet-18, EfficientNet-B0).

## ▶️ Running the Code

Only **Stage 1** and **Stage 3** scripts need to be run sequentially for the full training process:

```bash
# Step 1: Train bias models
python bias_models_new.py

# Step 2: Train final unbiased model
python Devil-NET_train.py
