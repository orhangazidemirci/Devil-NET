# -*- coding: utf-8 -*-
"""
Created on Tue May  6 21:55:17 2025

@author: Orhan
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets,models

import numpy as np
import pickle
import os
import random
import torch.nn.functional as F

dataset="cifar10" #cifar10, imagenet
model_type="efficientnet" #resnet, efficientnet

# --- Configuration (should match settings used when indices were saved) ---
num_bias_models = 5 # Or 10, etc., matching the saved indices file
BATCH_SIZE = 128
NEW_VALIDATION_SPLIT_RATIO = 0.1 # 20% of the *combined* data for new validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For reproducibility of the NEW train/val split
NEW_SPLIT_MANUAL_SEED = 42  
random.seed(NEW_SPLIT_MANUAL_SEED)
np.random.seed(NEW_SPLIT_MANUAL_SEED)
torch.manual_seed(NEW_SPLIT_MANUAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(NEW_SPLIT_MANUAL_SEED)


print(f"Using device: {DEVICE}")
print(f"Combining data originally for {num_bias_models} bias models.")

# --- 1. Define Transforms (must be consistent) ---
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([ # For the final test set
     transforms.Resize(32),  # ensure consistent size
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# --- 2. Specify and Load Indices File ---
indices_file_name_template = f'{dataset}_train_val_indices_split_{num_bias_models}_v2.pkl' 

# indices_file_name_template = 'cifar10_train_val_indices_split_{}_v2.pkl'
indices_file = indices_file_name_template.format(num_bias_models)

# --- 3. Load the Full Original Training Dataset ---
if dataset=="cifar10":
    print("\nLoading the full CIFAR10 training dataset...")
    try:
        full_train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform_train)
    except RuntimeError:
        full_train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        
elif dataset=="imagenet":


    full_train_dataset = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform=transform_train)
    # test_dataset = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=transform_test)
    
    # train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_dir = './data/tiny-imagenet-200/val'

    test_dataset = datasets.ImageFolder(os.path.join(val_dir, 'organized'), transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Full training dataset loaded with {len(full_train_dataset)} samples.")

# --- 4. Load the Saved Indices ---
loaded_global_val_indices = []      # From the previous setup
loaded_all_model_train_indices = [] # List of lists of indices for each bias model

if os.path.exists(indices_file):
    print(f"\nLoading pre-split indices from {indices_file}...")
    with open(indices_file, 'rb') as f:
        saved_indices_data = pickle.load(f)
        loaded_global_val_indices = saved_indices_data.get('val_indices', []) # Use .get for safety
        loaded_all_model_train_indices = saved_indices_data.get('train_indices_per_model', [])
    print("Indices loaded.")
    if not loaded_all_model_train_indices:
        print("ERROR: 'train_indices_per_model' not found or empty in the loaded file. Exiting.")
        exit()
    if len(loaded_all_model_train_indices) != num_bias_models:
         print(f"Warning: Number of models in loaded indices ({len(loaded_all_model_train_indices)}) "
              f"does not match current num_bias_models ({num_bias_models}). Using loaded count.")
         # Adjust num_bias_models if you want to be dynamic, or ensure they match
         # num_bias_models = len(loaded_all_model_train_indices)
else:
    print(f"Error: Indices file '{indices_file}' not found! Exiting.")
    exit()

# --- 5. Re-create Global Validation Set and DataLoader ---
if loaded_global_val_indices:
    global_val_subset_reloaded = Subset(full_train_dataset, loaded_global_val_indices)
    global_val_loader_reloaded = DataLoader(global_val_subset_reloaded, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"\nReloaded global validation set created with {len(global_val_subset_reloaded)} samples.")
    print(f"  First 5 indices for global validation: {loaded_global_val_indices[:5]}")
else:
    print("\nNo global validation indices found in the loaded file.")
    global_val_loader_reloaded = None


# --- 5. Prepare Data for the Combined Dataset ---
# We will combine all partitions from 'loaded_all_model_train_indices'.
# We could also choose to include 'loaded_global_val_indices' if desired.
# For now, let's assume we only combine the N model partitions.

combined_data_indices = []    # Stores the actual CIFAR10 index
combined_data_bias_source = [] # Stores the ID of the original bias model (0 to N-1)

for bias_model_id, partition_indices in enumerate(loaded_all_model_train_indices):
    combined_data_indices.extend(partition_indices)
    combined_data_bias_source.extend([bias_model_id] * len(partition_indices))
    print(f"  Added {len(partition_indices)} samples from original bias model partition {bias_model_id}.")

print(f"\nTotal samples in the combined pool (from {len(loaded_all_model_train_indices)} partitions): {len(combined_data_indices)}")
if len(combined_data_indices) != len(combined_data_bias_source):
    print("Error: Mismatch in lengths of combined_data_indices and combined_data_bias_source. This should not happen.")
    exit()

# Optional: Include the old global validation set into this combined pool
# If you want to do this, uncomment the following:
# if loaded_global_val_indices:
#     print(f"  Adding {len(loaded_global_val_indices)} samples from the old global validation set to the pool.")
#     combined_data_indices.extend(loaded_global_val_indices)
#     # Assign a special bias_source_id for these, e.g., -1 or num_bias_models
#     OLD_GLOBAL_VAL_SOURCE_ID = -1 # Or num_bias_models
#     combined_data_bias_source.extend([OLD_GLOBAL_VAL_SOURCE_ID] * len(loaded_global_val_indices))
#     print(f"Total samples in the combined pool now: {len(combined_data_indices)}")


# --- 6. Create a Custom Dataset to include bias source ---
class CIFAR10WithBiasSource(Dataset):
    def __init__(self, cifar_dataset, data_indices, bias_sources):
        """
        Args:
            cifar_dataset (Dataset): The original full CIFAR10 dataset (e.g., full_train_dataset).
            data_indices (list): List of indices into cifar_dataset that form this combined dataset.
            bias_sources (list): List of bias source IDs corresponding to each index in data_indices.
        """
        self.cifar_dataset = cifar_dataset
        self.data_indices = data_indices
        self.bias_sources = bias_sources
        
        if len(self.data_indices) != len(self.bias_sources):
            raise ValueError("data_indices and bias_sources must have the same length.")

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        # idx is an index for *this* CIFAR10WithBiasSource dataset (0 to len-1)
        original_cifar_idx = self.data_indices[idx]
        bias_source_id = self.bias_sources[idx]
        
        # Get image and original label from the base CIFAR10 dataset
        # The transform (transform_train) is already applied by full_train_dataset
        image, original_label = self.cifar_dataset[original_cifar_idx]
        
        return image, original_label, torch.tensor(bias_source_id, dtype=torch.long)

# Instantiate the combined dataset
combined_tagged_dataset = CIFAR10WithBiasSource(
    full_train_dataset,
    combined_data_indices,
    combined_data_bias_source
)
print(f"\nCreated `combined_tagged_dataset` with {len(combined_tagged_dataset)} samples.")

# --- 7. Split the Combined Tagged Dataset into New Train and Validation ---
num_combined_samples = len(combined_tagged_dataset)
indices_for_new_split = list(range(num_combined_samples))
np.random.shuffle(indices_for_new_split) # Shuffle for random split

split_point = int(np.floor(NEW_VALIDATION_SPLIT_RATIO * num_combined_samples))
new_val_indices_in_combined = indices_for_new_split[:split_point]
new_train_indices_in_combined = indices_for_new_split[split_point:]

new_train_subset = Subset(combined_tagged_dataset, new_train_indices_in_combined)
new_val_subset = Subset(combined_tagged_dataset, new_val_indices_in_combined)

print(f"Split combined dataset into:")
print(f"  - New Training Set: {len(new_train_subset)} samples")
print(f"  - New Validation Set: {len(new_val_subset)} samples")

# --- 8. Create DataLoaders ---
new_train_loader = DataLoader(new_train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
new_val_loader = DataLoader(new_val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Original Test set remains unchanged
original_test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
original_test_loader = DataLoader(original_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


# print(f"\nCreated DataLoaders:")
# print(f"  - `new_train_loader` (for the combined data, now split for training)")
# print(f"  - `new_val_loader` (for the combined data, now split for validation)")
# print(f"  - `original_test_loader` (standard CIFAR10 test set)")

# # --- 9. Verify the output of a DataLoader batch ---
# print("\nVerifying a batch from `new_train_loader`:")
# try:
#     images, original_labels, bias_source_ids = next(iter(new_train_loader))
#     print(f"  Images batch shape: {images.shape}")
#     print(f"  Original labels batch shape: {original_labels.shape}")
#     print(f"  Bias source IDs batch shape: {bias_source_ids.shape}")
#     print(f"  Sample original labels: {original_labels[:5]}")
#     print(f"  Sample bias source IDs: {bias_source_ids[:5]} (these indicate original partition index)")
# except Exception as e:
#     print(f"  Error fetching batch: {e}")

# print("\n--- Setup Complete ---")
# print("You can now train a model using `new_train_loader` and validate with `new_val_loader`.")
# print("Each sample will provide (image, original_cifar_label, bias_source_id).")



# def compute_statistics_dataset(
#     loaded_bias_models: list,  # List of pre-trained nn.Module models
#     dataloader: torch.utils.data.DataLoader, # DataLoader yielding (images, labels, possibly_other_info)
#     device: torch.device
# ) -> tuple:
#     """
#     Computes accuracy statistics for a list of bias models over a given dataset.

#     Parameters:
#       loaded_bias_models: A list of bias models (nn.Module instances).
#                           Each model must output logits of shape (batch_size, num_classes).
#       dataloader: A torch DataLoader (e.g., new_train_loader from previous steps).
#                   It should yield at least (images, labels, ...).
#       device: Device on which models and data are (e.g., "cuda" or "cpu").

#     Returns:
#       g_acc_list (list): List of float accuracies, one for each model in loaded_bias_models.
#       best_g_acc (float): Accuracy achieved by selecting the "best" bias model per sample
#                           (best = highest logit for the true class).
#       # The following are returned as 0.0 as their logic is not implemented for bias models alone
#       b_acc (float): Placeholder, returns 0.0.
#       combined_acc (float): Placeholder, returns 0.0.
#     """
#     num_bias_models = len(loaded_bias_models)
#     if num_bias_models == 0:
#         print("Warning: No bias models provided.")
#         return [], 0.0, 0.0, 0.0

#     # Ensure all models are on the correct device and in eval mode
#     for model in loaded_bias_models:
#         model.to(device)
#         model.eval()

#     # total_samples = 0
#     # # Correct counts for each individual bias model
#     # g_correct_counts = [0 for _ in range(num_bias_models)]
#     # # Correct counts when using the "best" bias model per sample
#     # best_g_correct_count = 0

#     # Placeholders, as their calculation logic is not tied to bias models alone in this func
#     # b_correct_count = 0
#     # combined_correct_count = 0

#     num_bias_models = len(loaded_bias_models)
#     if num_bias_models == 0: return [], 0.0, 0.0, 0.0
#     if num_bias_models == 1:
#         print("Warning: Only one bias model. 'g_acc_list_exclusive' and 'best_g_acc_exclusive' "
#               "will be 0 if all samples originate from this single bias source.")

#     for model in loaded_bias_models:
#         model.to(device)
#         model.eval()

#     # For g_acc_list_exclusive:
#     g_exclusive_correct_counts = [0 for _ in range(num_bias_models)]
#     g_exclusive_total_samples_evaluated = [0 for _ in range(num_bias_models)] # Denominators

#     # For best_g_acc_exclusive:
#     best_g_exclusive_correct_count = 0
#     total_samples_for_best_g = 0 # Denominator for best_g_acc_exclusive

#     # Placeholders
#     # b_correct_count = 0
#     # combined_correct_count = 0

#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(dataloader):
#             if len(batch_data) < 3:
#                 raise ValueError("DataLoader must yield (images, labels, bias_source_ids)")
            
#             images, labels, bias_source_ids = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)

#             batch_size_curr = labels.size(0)
#             if batch_size_curr == 0: continue
            
#             total_samples_for_best_g += batch_size_curr # All samples contribute to best_g evaluation pool

#             g_logits_list = [model(images) for model in loaded_bias_models]

#             # 1. Compute g_acc_list_exclusive[i]
#             #    Accuracy of model_i on samples where bias_source_id != i
#             for i, model_logits in enumerate(g_logits_list):
#                 # Identify samples in the current batch where bias_source_id is NOT i
#                 # These are the samples model_i should be evaluated on for g_acc_list_exclusive[i]
#                 relevant_samples_mask = (bias_source_ids != i)
                
#                 if relevant_samples_mask.sum().item() == 0: # No relevant samples in this batch for model i
#                     continue

#                 relevant_images = images[relevant_samples_mask]
#                 relevant_labels = labels[relevant_samples_mask]
#                 # No need to re-run model, just use relevant portion of model_logits
#                 relevant_logits = model_logits[relevant_samples_mask]

#                 g_pred_for_relevant = relevant_logits.argmax(dim=1)
#                 g_exclusive_correct_counts[i] += (g_pred_for_relevant == relevant_labels).sum().item()
#                 g_exclusive_total_samples_evaluated[i] += relevant_labels.size(0)


#             # 2. Compute "best" bias model accuracy (per sample) WITH EXCLUSION
#             #    (This part is the same as in compute_statistics_dataset_with_source_exclusion)
#             if num_bias_models > 1:
#                 true_class_labels = labels.unsqueeze(1)
#                 g_true_class_logits = [
#                     logits.gather(1, true_class_labels).squeeze(1) for logits in g_logits_list
#                 ]
#                 g_true_class_logits_stack = torch.stack(g_true_class_logits, dim=1)

#                 masked_g_true_class_logits_stack = g_true_class_logits_stack.clone()
#                 sample_idx_range = torch.arange(batch_size_curr, device=device)
#                 try:
#                     masked_g_true_class_logits_stack[sample_idx_range, bias_source_ids] = -torch.inf
#                 except IndexError as e:
#                     # Handle cases where bias_source_ids might be out of bounds for num_bias_models
#                     # This can happen if bias_source_ids were generated with a different N
#                     # or if there's a data integrity issue.
#                     valid_mask = (bias_source_ids >= 0) & (bias_source_ids < num_bias_models)
#                     if not valid_mask.all():
#                         print(f"Warning: Invalid bias_source_ids found in batch {batch_idx}. "
#                               f"Values: {torch.unique(bias_source_ids[~valid_mask])}. "
#                               f"Expected range [0, {num_bias_models-1}]. Skipping these for masking.")
#                     # Apply masking only for valid bias_source_ids
#                     masked_g_true_class_logits_stack[sample_idx_range[valid_mask], bias_source_ids[valid_mask]] = -torch.inf


#                 best_g_exclusive_indices_per_sample = torch.argmax(masked_g_true_class_logits_stack, dim=1)
#                 all_g_logits_stack = torch.stack(g_logits_list, dim=1)
#                 selected_logits_from_best_g_exclusive = all_g_logits_stack[
#                     sample_idx_range, best_g_exclusive_indices_per_sample, :
#                 ]
#                 best_g_exclusive_pred = selected_logits_from_best_g_exclusive.argmax(dim=1)
#                 best_g_exclusive_correct_count += (best_g_exclusive_pred == labels).sum().item()
#             # If num_bias_models == 1, best_g_exclusive_correct_count remains 0.


#     g_acc_list_exclusive = [
#         (g_exclusive_correct_counts[i] / g_exclusive_total_samples_evaluated[i])
#         if g_exclusive_total_samples_evaluated[i] > 0 else 0.0
#         for i in range(num_bias_models)
#     ]
    
#     best_g_acc_exclusive = (best_g_exclusive_correct_count / total_samples_for_best_g) \
#                            if total_samples_for_best_g > 0 else 0.0
    
#     b_acc = 0.0 # Placeholder
#     combined_acc = 0.0 # Placeholder

#     return g_acc_list_exclusive, best_g_acc_exclusive, b_acc, combined_acc

import torch
import torch.nn as nn
import torchvision # Only needed if you were also loading datasets, but good practice
import os




class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        

        out = self.fc(out)
        
        out = F.normalize(out, p=2, dim=1)

        return out
  
    
def ResNet18():
    if dataset=="cifar10":
        return ResNet(ResidualBlock).to(DEVICE)
    elif dataset=="imagenet":
        return ResNet(ResidualBlock,200).to(DEVICE)


# ###Dataset with biasses


def compute_statistics_dataset(
    loaded_bias_models: list,  # List of pre-trained nn.Module models
    dataloader: torch.utils.data.DataLoader, # DataLoader yielding (images, labels, possibly_other_info)
    device: torch.device
) -> tuple:
    """
    Computes accuracy statistics for a list of bias models over a given dataset.

    Parameters:
      loaded_bias_models: A list of bias models (nn.Module instances).
                          Each model must output logits of shape (batch_size, num_classes).
      dataloader: A torch DataLoader (e.g., new_train_loader from previous steps).
                  It should yield at least (images, labels, ...).
      device: Device on which models and data are (e.g., "cuda" or "cpu").

    Returns:
      g_acc_list (list): List of float accuracies, one for each model in loaded_bias_models.
      best_g_acc (float): Accuracy achieved by selecting the "best" bias model per sample
                          (best = highest logit for the true class).
      # The following are returned as 0.0 as their logic is not implemented for bias models alone
      b_acc (float): Placeholder, returns 0.0.
      combined_acc (float): Placeholder, returns 0.0.
    """
    num_bias_models = len(loaded_bias_models)
    if num_bias_models == 0:
        print("Warning: No bias models provided.")
        return [], 0.0, 0.0, 0.0

    # Ensure all models are on the correct device and in eval mode
    for model in loaded_bias_models:
        model.to(device)
        model.eval()

    # total_samples = 0
    # # Correct counts for each individual bias model
    # g_correct_counts = [0 for _ in range(num_bias_models)]
    # # Correct counts when using the "best" bias model per sample
    # best_g_correct_count = 0

    # Placeholders, as their calculation logic is not tied to bias models alone in this func
    # b_correct_count = 0
    # combined_correct_count = 0

    num_bias_models = len(loaded_bias_models)
    if num_bias_models == 0: return [], 0.0, 0.0, 0.0
    if num_bias_models == 1:
        print("Warning: Only one bias model. 'g_acc_list_exclusive' and 'best_g_acc_exclusive' "
              "will be 0 if all samples originate from this single bias source.")

    for model in loaded_bias_models:
        model.to(device)
        model.eval()

    # For g_acc_list_exclusive:
    g_exclusive_correct_counts = [0 for _ in range(num_bias_models)]
    g_exclusive_total_samples_evaluated = [0 for _ in range(num_bias_models)] # Denominators

    # For best_g_acc_exclusive:
    best_g_exclusive_correct_count = 0
    total_samples_for_best_g = 0 # Denominator for best_g_acc_exclusive

    # Placeholders
    # b_correct_count = 0
    # combined_correct_count = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) < 3:
                raise ValueError("DataLoader must yield (images, labels, bias_source_ids)")
            
            images, labels, bias_source_ids = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)

            batch_size_curr = labels.size(0)
            if batch_size_curr == 0: continue
            
            total_samples_for_best_g += batch_size_curr # All samples contribute to best_g evaluation pool

            g_logits_list = [model(images) for model in loaded_bias_models]

            # 1. Compute g_acc_list_exclusive[i]
            #    Accuracy of model_i on samples where bias_source_id != i
            for i, model_logits in enumerate(g_logits_list):
                # Identify samples in the current batch where bias_source_id is NOT i
                # These are the samples model_i should be evaluated on for g_acc_list_exclusive[i]
                relevant_samples_mask = (bias_source_ids != i)
                
                if relevant_samples_mask.sum().item() == 0: # No relevant samples in this batch for model i
                    continue

                relevant_images = images[relevant_samples_mask]
                relevant_labels = labels[relevant_samples_mask]
                # No need to re-run model, just use relevant portion of model_logits
                relevant_logits = model_logits[relevant_samples_mask]

                g_pred_for_relevant = relevant_logits.argmax(dim=1)
                g_exclusive_correct_counts[i] += (g_pred_for_relevant == relevant_labels).sum().item()
                g_exclusive_total_samples_evaluated[i] += relevant_labels.size(0)


            # 2. Compute "best" bias model accuracy (per sample) WITH EXCLUSION
            #    (This part is the same as in compute_statistics_dataset_with_source_exclusion)
            if num_bias_models > 1:
                true_class_labels = labels.unsqueeze(1)
                g_true_class_logits = [
                    logits.gather(1, true_class_labels).squeeze(1) for logits in g_logits_list
                ]
                g_true_class_logits_stack = torch.stack(g_true_class_logits, dim=1)

                masked_g_true_class_logits_stack = g_true_class_logits_stack.clone()
                sample_idx_range = torch.arange(batch_size_curr, device=device)
                try:
                    masked_g_true_class_logits_stack[sample_idx_range, bias_source_ids] = -torch.inf
                except IndexError as e:
                    # Handle cases where bias_source_ids might be out of bounds for num_bias_models
                    # This can happen if bias_source_ids were generated with a different N
                    # or if there's a data integrity issue.
                    valid_mask = (bias_source_ids >= 0) & (bias_source_ids < num_bias_models)
                    if not valid_mask.all():
                        print(f"Warning: Invalid bias_source_ids found in batch {batch_idx}. "
                              f"Values: {torch.unique(bias_source_ids[~valid_mask])}. "
                              f"Expected range [0, {num_bias_models-1}]. Skipping these for masking.")
                    # Apply masking only for valid bias_source_ids
                    masked_g_true_class_logits_stack[sample_idx_range[valid_mask], bias_source_ids[valid_mask]] = -torch.inf


                best_g_exclusive_indices_per_sample = torch.argmax(masked_g_true_class_logits_stack, dim=1)
                all_g_logits_stack = torch.stack(g_logits_list, dim=1)
                selected_logits_from_best_g_exclusive = all_g_logits_stack[
                    sample_idx_range, best_g_exclusive_indices_per_sample, :
                ]
                best_g_exclusive_pred = selected_logits_from_best_g_exclusive.argmax(dim=1)
                best_g_exclusive_correct_count += (best_g_exclusive_pred == labels).sum().item()
            # If num_bias_models == 1, best_g_exclusive_correct_count remains 0.


    g_acc_list_exclusive = [
        (g_exclusive_correct_counts[i] / g_exclusive_total_samples_evaluated[i])
        if g_exclusive_total_samples_evaluated[i] > 0 else 0.0
        for i in range(num_bias_models)
    ]
    
    best_g_acc_exclusive = (best_g_exclusive_correct_count / total_samples_for_best_g) \
                            if total_samples_for_best_g > 0 else 0.0
    
    b_acc = 0.0 # Placeholder
    combined_acc = 0.0 # Placeholder

    return g_acc_list_exclusive, best_g_acc_exclusive, b_acc, combined_acc



# # --- Configuration ---
# # num_bias_models = 10 # Or 10, etc. MUST match the number of models you trained and saved
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SAVED_MODELS_DIR = './saved_bias_models_v2' # Directory where model weights are saved
SAVED_MODELS_DIR = f'./{dataset}/{model_type}/saved_bias_models_v2' # New dir for this version

# print(f"Using device: {DEVICE}")
# print(f"Attempting to load {num_bias_models} bias models from: {SAVED_MODELS_DIR}")


# # # --- 2. Initialize Model Instances ---
# # loaded_bias_models = []
# # for i in range(num_bias_models):
# #     model = ResNet18().cuda() # Create a new instance of the model architecture
# #     loaded_bias_models.append(model)

# # print(f"\nInitialized {len(loaded_bias_models)} fresh ResNet18 model instances.")
if __name__ == '__main__':
    
    # --- 2. Load Pre-trained Bias Models ---
    loaded_bias_models = []
    for i in range(num_bias_models):
        if model_type=="resnet":
            model=ResNet18().to(DEVICE)
        elif model_type=="efficientnet":
            # Load pretrained EfficientNet-B0
            model = models.efficientnet_b0(pretrained=False)
            # Replace the classifier head
            if dataset=="cifar10":
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    
            elif dataset=="imagenet":
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
        # model = ResNet18()
        model_filename = f"bias_model_{i}_best.pth"
        model_path = os.path.join(SAVED_MODELS_DIR, model_filename)
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.to(DEVICE)
                model.eval()
                loaded_bias_models.append(model)
                print(f"Successfully loaded model {i} from {model_path}")
            except Exception as e:
                print(f"Error loading model {i}: {e}. Skipping this model.")
                loaded_bias_models.append(None) # Placeholder for missing model
        else:
            print(f"Weight file not found for model {i} at {model_path}. Skipping this model.")
            loaded_bias_models.append(None) # Placeh
            
    g_accuracies, best_g_accuracy, _, _ = compute_statistics_dataset(
        loaded_bias_models=loaded_bias_models,
        dataloader=new_train_loader,
        device=DEVICE
    )
    
    print("\n--- Results from `compute_statistics_dataset` (mock data) ---")
    for i, acc in enumerate(g_accuracies):
        print(f"  Accuracy of Bias Model {i}: {acc*100:.2f}%")
    print(f"  Accuracy of 'Best Bias Model (per sample)': {best_g_accuracy*100:.2f}%")

# import torch.optim as optim



# # --- Training Function ---
# def train_model_with_validation(
#     model_name: str,
#     model: nn.Module,
#     train_dataloader: DataLoader,
#     val_dataloader: DataLoader,
#     epochs: int,
#     lr: float,
#     device: torch.device,
#     loss_type: str = "cross_entropy",
#     temp: float = 1.0
# ):
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     BEST_MODEL_SAVE_DIR="./hotmodels"
#     best_val_metric = float('inf') if loss_type != "cross_entropy" else 0.0 # Loss or Accuracy
#     best_model_path = os.path.join(BEST_MODEL_SAVE_DIR, f"{model_name}_best.pth")

#     print(f"\nTraining {model_name} with {loss_type} loss...")

#     for epoch in range(epochs):
#         model.train()
#         running_train_loss = 0.0
#         for batch_idx, (inputs, targets, bias_source_id) in enumerate(train_dataloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)

#             if loss_type == "cross_entropy":
#                 loss = F.cross_entropy(outputs, targets)
#             elif loss_type == "kl_divergence":
#                 student_log_probs = F.log_softmax(outputs / temp, dim=1)
#                 teacher_probs = F.softmax(targets / temp, dim=1)
#                 loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temp * temp)
#             elif loss_type == "mse_logits":
#                 loss = F.mse_loss(outputs, targets)
#             else:
#                 raise ValueError("Unsupported loss_type")
            
#             loss.backward()
#             optimizer.step()
#             running_train_loss += loss.item()
        
#         avg_train_loss = running_train_loss / len(train_dataloader)

#         # Validation step
#         model.eval()
#         running_val_loss = 0.0
#         correct_val = 0
#         total_val = 0
#         with torch.no_grad():
#             for inputs, targets in val_dataloader:
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)

#                 if loss_type == "cross_entropy":
#                     val_loss_item = F.cross_entropy(outputs, targets, reduction='sum').item() # Sum for avg later
#                     _, predicted = outputs.max(1)
#                     total_val += targets.size(0)
#                     correct_val += predicted.eq(targets).sum().item()
#                 elif loss_type == "kl_divergence":
#                     student_log_probs = F.log_softmax(outputs / temp, dim=1)
#                     teacher_probs = F.softmax(targets / temp, dim=1) # targets are logits here
#                     val_loss_item = F.kl_div(student_log_probs, teacher_probs, reduction='sum').item() * (temp * temp)
#                     total_val += inputs.size(0) # KLDiv doesn't have 'correct' in the same way
#                 elif loss_type == "mse_logits":
#                     val_loss_item = F.mse_loss(outputs, targets, reduction='sum').item() # targets are logits
#                     total_val += inputs.size(0) # MSE doesn't have 'correct' in the same way
                
#                 running_val_loss += val_loss_item
        
#         avg_val_loss = running_val_loss / total_val if total_val > 0 else float('inf')
#         val_accuracy = 100. * correct_val / total_val if loss_type == "cross_entropy" and total_val > 0 else 0.0

#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", end="")
#         if loss_type == "cross_entropy":
#             print(f" | Val Acc: {val_accuracy:.2f}%")
#         else:
#             print() # Newline

#         # Save best model
#         current_metric_is_better = False
#         if loss_type == "cross_entropy": # Higher accuracy is better
#             if val_accuracy > best_val_metric:
#                 best_val_metric = val_accuracy
#                 current_metric_is_better = True
#         else: # Lower loss is better for KLDiv, MSE
#             if avg_val_loss < best_val_metric:
#                 best_val_metric = avg_val_loss
#                 current_metric_is_better = True
        
#         if current_metric_is_better:
#             torch.save(model.state_dict(), best_model_path)
#             print(f"  New best validation metric: {best_val_metric:.4f}. Model saved to {best_model_path}")
            
#     print(f"Finished training {model_name}. Best validation metric: {best_val_metric:.4f} achieved.")
#     return best_model_path

# TEMPERATURE=1.0
# NEW_MODEL_EPOCHS=20
# NEW_MODEL_LR=0.001
# # --- Train Model A (Soft Targets - KL Divergence is common) ---
# model_A_fresh = ResNet18() # Fresh instance
# best_model_A_path = train_model_with_validation(
#     model_name="unbiased",
#     model=model_A_fresh,
#     train_dataloader=new_train_loader,
#     val_dataloader=new_val_loader,
#     epochs=NEW_MODEL_EPOCHS,
#     lr=NEW_MODEL_LR,
#     device=DEVICE,
#     loss_type="kl_divergence",
#     temp=TEMPERATURE
# )

# # --- Train Model B (Hard Targets - Cross Entropy) ---
# model_B_fresh = ResNet18() # Fresh instance
# best_model_B_path = train_model_with_validation(
#     model_name="comparison",
#     model=model_B_fresh,
#     train_dataloader=new_train_loader,
#     val_dataloader=new_val_loader,
#     epochs=NEW_MODEL_EPOCHS,
#     lr=NEW_MODEL_LR,
#     device=DEVICE,
#     loss_type="cross_entropy"
# )
# # You can now use these models, for example:
# # if loaded_bias_models:
# #   first_loaded_model = loaded_bias_models[0]
# #   # dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE) # Example CIFAR10 input
# #   # with torch.no_grad():
# #   #   output = first_loaded_model(dummy_input)
# #   #   print(f"Output shape from first loaded model: {output.shape}")