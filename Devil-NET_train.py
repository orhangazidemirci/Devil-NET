import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset # For example usage and return type
import os # Still useful for creating output_dir if other things are saved there
import torch.nn.functional as F
# Assume DEVICE is defined globally or passed as an argument
# Assume ResNet18 or MockResNet class is defined
import matplotlib.pyplot as plt
from torchvision import datasets,models

import random
import numpy as np
MANUAL_SEED = 42
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

dataset="imagenet" #cifar10, imagenet
model_type="resnet" #resnet, efficientnet


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(MANUAL_SEED)

def create_dataset_with_best_g_logits_exclusive(
    bias_models_list: list,
    source_dataloader: DataLoader, # Expected to yield (images, labels, bias_source_ids)
    device: torch.device
) -> TensorDataset | None: # Return a TensorDataset or None on failure
    """
    Processes data from source_dataloader to create a new TensorDataset.
    The new dataset yields: (original_image, original_label, best_g_logit_exclusive).
    "best_g_logit_exclusive": For each sample, these are the full logits from the
       bias model that gives the highest true-class logit, EXCLUDING the model whose ID
       matches the sample's bias_source_id.

    Parameters:
      bias_models_list: List of pre-trained bias models (nn.Module instances).
      source_dataloader: DataLoader yielding (images, labels, bias_source_ids).
      device: Device for computation.

    Returns:
      A TensorDataset containing (images, original_labels, best_g_logits_exclusive),
      or None if processing fails or no data is generated.
    """
    num_bias_models = len(bias_models_list)
    if num_bias_models == 0:
        print("Error: No bias models provided.")
        return None
    if num_bias_models == 1:
        print("Warning: Only one bias model provided. 'Best G Logits with Exclusion' might not be "
              "meaningful as there's no 'other' model to choose if all samples originate from this "
              "single source. Logits from this single model will effectively be used.")

    print(f"Creating dataset with 'best_g_logits_exclusive' for distillation...")

    all_selected_best_g_logits_exclusive_cpu = []
    all_original_images_cpu = []
    all_original_labels_cpu = []

    for model in bias_models_list:
        model.to(device)
        model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(source_dataloader):
            if len(batch_data) < 3:
                # If your source_dataloader might not always have bias_source_ids, handle this.
                # For this function's explicit purpose, we expect it.
                print(f"Warning: Batch {batch_idx} did not yield 3 components (expected images, labels, bias_source_ids). Skipping batch.")
                # Or raise ValueError("DataLoader must yield (images, labels, bias_source_ids)")
                continue
            
            images, labels, bias_source_ids = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)

            batch_size_curr = images.size(0)
            if batch_size_curr == 0: continue

            g_logits_list = [model(images) for model in bias_models_list] 

            true_class_labels = labels.unsqueeze(1)
            g_true_class_logits = [
                logits.gather(1, true_class_labels).squeeze(1) for logits in g_logits_list
            ]
            g_true_class_logits_stack = torch.stack(g_true_class_logits, dim=1)

            masked_g_true_class_logits_stack = g_true_class_logits_stack.clone()
            sample_idx_range = torch.arange(batch_size_curr, device=device)
            
            valid_bias_source_indices_mask = (bias_source_ids >= 0) & (bias_source_ids < num_bias_models)
            if valid_bias_source_indices_mask.any():
                 masked_g_true_class_logits_stack[
                     sample_idx_range[valid_bias_source_indices_mask], 
                     bias_source_ids[valid_bias_source_indices_mask]
                 ] = -torch.inf
            
            best_g_exclusive_indices_per_sample = torch.argmax(masked_g_true_class_logits_stack, dim=1)
            all_g_logits_stack = torch.stack(g_logits_list, dim=1)
            selected_logits = all_g_logits_stack[
                sample_idx_range, best_g_exclusive_indices_per_sample, :
            ]
            
            all_selected_best_g_logits_exclusive_cpu.append(selected_logits.cpu())
            all_original_images_cpu.append(images.cpu())
            all_original_labels_cpu.append(labels.cpu())
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(source_dataloader):
                print(f"  Processed batch {batch_idx+1}/{len(source_dataloader)}")

    if not all_selected_best_g_logits_exclusive_cpu:
        print("No data processed or generated to create the dataset.")
        return None

    # Concatenate all collected batch tensors
    final_best_g_logits_exclusive = torch.cat(all_selected_best_g_logits_exclusive_cpu, dim=0)
    final_original_images = torch.cat(all_original_images_cpu, dim=0)
    final_original_labels = torch.cat(all_original_labels_cpu, dim=0)

    print(f"\nGenerated 'best_g_logits_exclusive' tensor shape: {final_best_g_logits_exclusive.shape}")
    print(f"Collected original images tensor shape: {final_original_images.shape}")
    print(f"Collected original labels tensor shape: {final_original_labels.shape}")

    # Create and return the TensorDataset
    distillation_dataset = TensorDataset(
        final_original_images,
        final_original_labels, # For potential combined loss or just for reference
        final_best_g_logits_exclusive # These are the soft targets
    )
    print(f"Created TensorDataset for distillation with {len(distillation_dataset)} samples.")
    return distillation_dataset

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
  
    
# def ResNet18():
#     return ResNet(ResidualBlock).to(DEVICE)
def ResNet18():
    if dataset=="cifar10":
        return ResNet(ResidualBlock).to(DEVICE)
    elif dataset=="imagenet":
        return ResNet(ResidualBlock,200).to(DEVICE)

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Assume ResNet18 definition is available
# Assume DEVICE is defined

BEST_MODEL_SAVE_DIR = f"./hotmodels/{dataset}/{model_type}/" # Centralized definition
os.makedirs(BEST_MODEL_SAVE_DIR, exist_ok=True)

def train_model_with_validation(
    model_name: str,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    learning_rate: float, # Renamed for clarity
    device: torch.device,
    loss_type: str, # e.g., "kl_divergence" or "cross_entropy"
    temperature: float = 2.0, # Default temperature for KLDiv
    epochs_switch: int = 30,
    sharpening_factor: float = 1.5,
    noise_level: float = 0.1

):
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Using Adam as requested
    best_optimizer=optimizer
    # Determine what metric to optimize for validation
    # For KLDiv or MSE, we minimize loss.
    # For CrossEntropy, we typically maximize accuracy (or minimize loss).
    # Let's consistently minimize validation loss for simplicity in choosing the best model,
    # but we'll still report accuracy for cross_entropy.
    best_val_loss = float('inf')
    best_model_path = os.path.join(BEST_MODEL_SAVE_DIR, f"{model_name}_best.pth")
    # Reduce LR on plateau of validation loss
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, verbose=True)
    print(f"\n--- Training {model_name} ---")
    print(f"  Loss Type: {loss_type}, Optimizer: Adam, LR: {learning_rate}, Epochs: {epochs}")
    if loss_type == "kl_divergence":
        print(f"  Temperature for KLDiv: {temperature}")
    epoch_switch=epochs_switch
    
    val_losses=[]
    training_losses=[]
   
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        
        # --- Training Loop ---
        for batch_idx, batch_data in enumerate(train_dataloader):
            inputs = batch_data[0].to(device)
            # `actual_targets` will be class indices for cross_entropy
            # or teacher logits for kl_divergence/mse_logits
            
            if loss_type == "cross_entropy" or epoch>epoch_switch:
                # Dataloader for Comparison Model yields (images, original_labels)
                actual_targets = batch_data[1].to(device) # These are the original_labels
            elif loss_type == "kl_divergence" or loss_type == "mse_logits":
                # Dataloader for Unbiased Model yields (images, original_labels, best_g_logits)
                # We need the best_g_logits as targets for distillation
                actual_targets = batch_data[2].to(device) # These are best_g_logits_exclusive
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            optimizer.zero_grad()
            outputs = model(inputs) # Student model's raw logits

            if loss_type == "cross_entropy" or epoch>epoch_switch:
                if epoch==epoch_switch+1 and loss_type == "kl_divergence":
                #     optimizer = optim.Adam(model.parameters(), lr=learning_rate/2) # Using Adam as requested
                    new_lr = learning_rate/10  # your desired learning rate
                    
                    if model_type=="resnet":
                        model=ResNet18()
                    elif model_type=="efficientnet":
                        # Load pretrained EfficientNet-B0
                        model = models.efficientnet_b0(pretrained=False)
                        # Replace the classifier head
                        if dataset=="cifar10":
                            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
                    
                        elif dataset=="imagenet":
                            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
                    # model=ResNet18()
                    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
                    
                    model=model.to(DEVICE)
                    model.train()
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
                    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=new_lr)
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, verbose=True)

                    # optimizer=best_optimizer
                    # for param_group in optimizer.param_groups:
                    #     param_group['lr'] = new_lr
                    # optimizer.zero_grad()

                loss = F.cross_entropy(outputs, actual_targets)
            elif loss_type == "kl_divergence":
                # sharpening_factor = sharpening_factor # Or an additive constant, e.g., +2.0. Needs tuning.
                # sharpened_logits = actual_targets.clone() # B, C
                
                # # For each sample in the batch, increase the logit of its true class
                # batch_indices = torch.arange(sharpened_logits.size(0), device=device)
                # true_class_original_logits = sharpened_logits[batch_indices, batch_data[1].to(device)]
                
                # # Method 1: Multiplicative sharpening
                # sharpened_logits[batch_indices, batch_data[1].to(device)] = true_class_original_logits * sharpening_factor

                # student_log_probs = F.log_softmax(outputs / temperature, dim=1)
                # # actual_targets are teacher_logits here
                # teacher_probs = F.softmax(sharpened_logits / temperature, dim=1)
                
                
                # loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature * temperature)
                noise_level = noise_level  # Can try values like 0.05 - 0.2

                
                
                                
                labels = batch_data[1].to(device) # These are best_g_logits_exclusive

                sharpening_factor = sharpening_factor # Or an additive constant, e.g., +2.0. Needs tuning.
                sharpened_logits = actual_targets.clone() # B, C
                
                # For each sample in the batch, increase the logit of its true class
                batch_indices = torch.arange(sharpened_logits.size(0), device=device)
                true_class_original_logits = sharpened_logits[batch_indices, batch_data[1].to(device)]
                
                # Create random noise for each class (shape: [B, C])
                noise = torch.randn_like(sharpened_logits) * noise_level
                
                # Compute sum of noise per sample
                noise_sum = noise.sum(dim=1, keepdim=True)  # Shape: [B, 1]
                
                # Set true class noise to half the total noise
                noise[batch_indices, labels] = (noise_sum / 2).squeeze()
                
                # Add noise to full logits
                sharpened_logits += noise

                sharpened_logits[batch_indices, labels] = true_class_original_logits * sharpening_factor

                student_log_probs = F.log_softmax(outputs / temperature, dim=1)
                # actual_targets are teacher_logits here
                teacher_probs = F.softmax(sharpened_logits / temperature, dim=1)
                
                correct_teacher = teacher_probs.argmax(dim=1)
                teacher_mask = (correct_teacher == labels)
                neg_mask = ~teacher_mask

                # if teacher_mask.any():
                #     if neg_mask.any():
                #         loss_ce = F.cross_entropy(outputs[neg_mask], labels[neg_mask])
                #         loss_kl = F.kl_div(student_log_probs[teacher_mask], teacher_probs[teacher_mask], reduction='batchmean') * (temperature * temperature)
                #         loss=loss_ce+loss_kl
                #     else:
                #         loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature * temperature)
                # else:
                #     loss = F.cross_entropy(outputs, labels)
                
                # Initialize final_logits with teacher_logits
                final_logits = teacher_probs.clone()
                
                # For weak teacher predictions, replace logits with corrected ones
                if neg_mask.any():
                    # Manually sharpen the true class
                    true_logits = teacher_probs[neg_mask].clone()
                    true_labels = labels[neg_mask]
                
                    # Boost the true class logit
                    idx = torch.arange(true_logits.size(0), device=outputs.device)
                    true_logits[idx, true_labels] *= sharpening_factor**2
                
                    # Replace in final_logits
                    final_logits[neg_mask] = true_logits
                loss = F.kl_div(student_log_probs, final_logits, reduction='batchmean') * (temperature * temperature)
            elif loss_type == "mse_logits":
                # actual_targets are teacher_logits here
                loss = F.mse_loss(outputs, actual_targets)
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)

        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0.0
        val_correct_if_ce = 0
        val_total_if_ce = 0 # Only relevant for cross_entropy

        with torch.no_grad():
            for batch_data_val in val_dataloader:
                inputs_val = batch_data_val[0].to(device)
                
     
                actual_targets_val = batch_data_val[1].to(device) # original_labels
                outputs_val = model(inputs_val)
                loss_item_val = F.cross_entropy(outputs_val, actual_targets_val, reduction='sum').item()
                
                _, predicted_val = outputs_val.max(1)
                val_total_if_ce += actual_targets_val.size(0)
                val_correct_if_ce += predicted_val.eq(actual_targets_val).sum().item()

                running_val_loss += loss_item_val
        
        # total_samples_in_val_epoch is val_total_if_ce (which is correctly populated for all loss types)
        avg_val_loss = running_val_loss / val_total_if_ce if val_total_if_ce > 0 else float('inf')
        
        val_losses.append(avg_val_loss)


        scheduler.step(avg_val_loss)
        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']}")
        val_accuracy_if_ce = (100. * val_correct_if_ce / val_total_if_ce) \
                             if  val_total_if_ce > 0 else 0.0
                             
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", end="")
        # if loss_type == "cross_entropy":
        print(f" | Val Acc: {val_accuracy_if_ce:.2f}%")
        # else:
        #     # For KLDiv/MSE, we don't have a direct "accuracy" against teacher logits.
        #     # We could calculate accuracy against original_labels (batch_data_val[1]) if desired for monitoring.
        #     # Let's do that for KLDiv/MSE validation for better intuition:
        #     val_accuracy_against_hard_labels = 0.0
        #     if loss_type != "cross_entropy" and val_total_if_ce > 0:
        #          # Need to re-calculate accuracy against original_labels if they are available
        #          temp_correct = 0
        #          temp_total = 0
        #          with torch.no_grad(): # Re-iterate val_dataloader to get original labels
        #             for b_data_val_for_acc in val_dataloader:
        #                 imgs_for_acc = b_data_val_for_acc[0].to(device)
        #                 orig_lbls_for_acc = b_data_val_for_acc[1].to(device) # original_labels are at index 1
                        
        #                 outs_for_acc = model(imgs_for_acc)
        #                 _, preds_for_acc = outs_for_acc.max(1)
        #                 temp_total += orig_lbls_for_acc.size(0)
        #                 temp_correct += preds_for_acc.eq(orig_lbls_for_acc).sum().item()
        #          val_accuracy_against_hard_labels = (100. * temp_correct / temp_total) if temp_total > 0 else 0.0
        #          print(f" | Val Acc (vs Hard Labels): {val_accuracy_against_hard_labels:.2f}%")
        #     else: # just a newline for cross_entropy already printed
        #         print()


        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            best_optimizer=optimizer
            # best_model=model
            print(f"  New best Val Loss: {best_val_loss:.4f}. Model saved to {best_model_path}")
            
    print(f"Finished training {model_name}. Best validation loss: {best_val_loss:.4f} achieved.")
    return training_losses,val_losses


#####Evalutation Loop
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import copy

def evaluate_model_extended(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int # Required for some sklearn metrics like confusion matrix if not all classes appear
):
    model.to(device)
    model.eval()
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Assuming dataloader can yield (inputs, labels) or (inputs, labels, other_stuff)
            inputs = batch_data[0].to(device)
            labels = batch_data[1].to(device) # Assuming labels are at index 1
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    if not all_labels: # Dataloader was empty
        print("Warning: No samples found in the dataloader for evaluation.")
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int),
            "per_class_metrics": {}
        }

    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)

    # Overall Accuracy
    accuracy = accuracy_score(all_labels_np, all_predictions_np) * 100.0

    # Precision, Recall, F1-score (Macro and Weighted)
    # `zero_division=0` handles cases where a class has no predictions or no true samples.
    # `labels=np.arange(num_classes)` ensures metrics are reported for all classes, even if some are missing in the batch.
    # However, if some classes are completely absent from all_labels_np, sklearn might warn or error depending on version
    # We can ensure all_labels_np and all_predictions_np are treated with the full set of possible labels
    # For precision_recall_fscore_support, this is less critical if `average` is macro/weighted
    
    unique_labels_present = np.unique(np.concatenate((all_labels_np, all_predictions_np)))
    target_names = [f"Class {i}" for i in range(num_classes)]
    
    # Ensure labels for metrics cover all classes if some are missing in the current eval set
    # This affects per-class reporting mainly. For macro/weighted, it's less of an issue.
    report_labels = np.arange(num_classes)


    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels_np, all_predictions_np, average='macro', zero_division=0, labels=report_labels
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels_np, all_predictions_np, average='weighted', zero_division=0, labels=report_labels
    )
    
    # Per-class metrics
    # Setting labels=report_labels ensures we get a metric for each class, even if it's 0 due to no presence.
    p_class, r_class, f1_class, s_class = precision_recall_fscore_support(
        all_labels_np, all_predictions_np, average=None, zero_division=0, labels=report_labels
    )
    
    per_class_metrics_dict = {}
    for i in range(num_classes):
        per_class_metrics_dict[target_names[i]] = {
            "precision": p_class[i],
            "recall": r_class[i],
            "f1-score": f1_class[i],
            "support": s_class[i] # Number of true instances for class i in this dataset
        }

    # Confusion Matrix
    # `labels=report_labels` ensures the matrix is num_classes x num_classes
    cm = confusion_matrix(all_labels_np, all_predictions_np, labels=report_labels)

    results = {
        "accuracy": accuracy,
        "precision_macro": precision_macro * 100.0,
        "recall_macro": recall_macro * 100.0,
        "f1_macro": f1_macro * 100.0,
        "precision_weighted": precision_weighted * 100.0,
        "recall_weighted": recall_weighted * 100.0,
        "f1_weighted": f1_weighted * 100.0,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics_dict
    }
    
    return results

import bias_data_combined
import copy
from collections import defaultdict

# --- Example of how to use it (assuming mock setup or your actual setup) ---
if __name__ == '__main__':
    import bias_dataload
    print("Setting up mock data for `create_dataset_with_best_g_logits_exclusive`...")


    # --- Configuration ---
    num_bias_models = 5 # Or 10, etc. MUST match the number of models you trained and saved
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SAVED_MODELS_DIR = './saved_bias_models_v2' # Directory where model weights are saved
    SAVED_MODELS_DIR = f'./{dataset}/{model_type}/saved_bias_models_v2' # New dir for this version

    BATCH_SIZE = 128
    NEW_VALIDATION_SPLIT_RATIO = 0.1 # 20% of the *combined* data for new validation
    
    
    
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
            
            
    new_train_loader=bias_data_combined.new_train_loader
    original_test_loader=bias_data_combined.original_test_loader
    validation_loader= bias_data_combined.global_val_loader_reloaded
    # Generate the dataset in memory
    distillation_dataset_in_memory = create_dataset_with_best_g_logits_exclusive(
        bias_models_list=loaded_bias_models,
        source_dataloader=new_train_loader,
        device=DEVICE
    )
    
    # # Generate the dataset in memory
    # distillation_dataset_for_lr = create_dataset_with_best_g_logits_exclusive(
    #     bias_models_list=loaded_bias_models,
    #     source_dataloader=new_train_loader,
    #     device=DEVICE
    # )
    # distillation_dataset_for_lr= DataLoader(distillation_dataset_for_lr, batch_size=BATCH_SIZE, shuffle=True)
    distillation_train_loader = DataLoader(distillation_dataset_in_memory, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader= bias_dataload.global_val_loader_reloaded

    #     # Verify a batch
    # try:
    #     img_batch, lbl_batch, logits_batch = next(iter(distillation_train_loader))
    #     print(f"  Sample batch from distillation_train_loader:")
    #     print(f"    Images shape: {img_batch.shape}")
    #     print(f"    Original Labels shape: {lbl_batch.shape}")
    #     print(f"    Best G Exclusive Logits shape: {logits_batch.shape}")
    # except Exception as e:
    #     print(f"  Error fetching batch: {e}")
    # else:
    #     print("Failed to create distillation dataset in memory.")

    # This `distillation_dataset_in_memory` (or a DataLoader made from it)
    # would then be used in Stage V to train your "Unbiased Model".
    # The "Comparison Model" would still be trained using a dataset of
    # (final_original_images, final_original_labels).
    
    ####APPLY Label sharpening, get distribution of logits to use it for the failure ones.
    # TEMPERATURES=[1.0,3.0,5.0]
    mode="Train" # Train, Eval
    TEMPERATURES=[1.0]

    NEW_MODEL_EPOCHS=100
    NEW_MODEL_LR=0.003390804841860312
    # --- Train Model A (Soft Targets - KL Divergence is common) ---
    # --- Train Unbiased Model (Model A) ---
    # unbiased_model_fresh = ResNet18() # Your actual ResNet18
    if model_type=="resnet":
        unbiased_model_fresh=ResNet18().to(DEVICE)
    elif model_type=="efficientnet":
        # Load pretrained EfficientNet-B0
        unbiased_model_fresh = models.efficientnet_b0(pretrained=False)
        # Replace the classifier head
        if dataset=="cifar10":
            unbiased_model_fresh.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

        elif dataset=="imagenet":
            unbiased_model_fresh.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
        
    # # === choose an init ===
    # learning_rates=[0.001,0.002,0.003]
    # # learning_rates=[0.002]
    
    # noise_levels=[0.1,0.3,0.5]
    # epochs_switches=[50,100]
    # sharpening_factors=[1.5,2]
 
    # === choose an init ===
    learning_rates=[0.001]
    # learning_rates=[0.002]
    
    noise_levels=[0.1]
    epochs_switches=[50]
    sharpening_factors=[2]
    
    loss_type="kl_divergence" #kl_divergence
    losses = defaultdict(lambda: {'train': [], 'val': []})
    # accuracies = defaultdict(lambda: {'train': [], 'val': []})
    if mode=="Train":
        for temprature in TEMPERATURES:
            for l_rates in learning_rates:
                for epoch_switch in epochs_switches:
                    for sharpening_factor in sharpening_factors:
                        for noise_level in noise_levels:
                            # model_name=f"Unbiased_Model_{loss_type}_lr{l_rates}_temp{temprature}"            
                            # model_name=f"Unbiased_Model_{loss_type}_lr{l_rates}_temp{temprature}_sharp{sharpening_factor}_switch{epoch_switch}_noise{noise_level}"            
                            model_name=f"Unbiased_Model_{dataset}_{model_type}_lr{l_rates}_sharp{sharpening_factor}_switch{epoch_switch}_noise{noise_level}"            

                            # soup_state_dict=soup_models(loaded_bias_models)
                    
                            # unbiased_model_fresh.load_state_dict(soup_state_dict)
                        
                            training_losses,val_losses = train_model_with_validation(
                                model_name=model_name,
                                model=copy.deepcopy(unbiased_model_fresh),  #Initialize each model the same
                                train_dataloader=distillation_train_loader, # Yields (img, orig_lbl, best_g_logits_excl)
                                val_dataloader=validation_loader,     # Yields (img, orig_lbl, best_g_logits_excl)
                                epochs=NEW_MODEL_EPOCHS,
                                learning_rate=l_rates, # e.g., 1e-3 or 1e-4 for Adam
                                device=DEVICE,
                                loss_type=loss_type,
                                temperature=temprature,
                                epochs_switch=epoch_switch,
                                sharpening_factor=sharpening_factor,
                                noise_level=noise_level
    
                            )
                            losses[model_name]['train'] = training_losses
                            losses[model_name]['val'] = val_losses

    import pickle
    path_for_losses=f"{dataset}_{model_type}_losses_devil_final.pkl"
    # Save as regular dict
    
    if mode=="Train":
        with open(path_for_losses, 'wb') as f:
            pickle.dump(dict(losses), f)
            
    # Load losses from file
    with open(path_for_losses, 'rb') as f:
        losses_load = pickle.load(f)
    #Plotting the training and validation loss values under different setups

#     plt.figure(figsize=(10,6))
#     for temprature in TEMPERATURES:
#         for l_rates in learning_rates:
#             for epoch_switch in epochs_switches:
#                 for sharpening_factor in sharpening_factors:
#                     for noise_level in noise_levels:
#                         model_name=f"Unbiased_Model_{loss_type}_lr{l_rates}_temp{temprature}_sharp{sharpening_factor}_switch{epoch_switch}_noise{noise_level}"            
#                         if epoch_switch==100:
#                             # epochs_list = range(1, len(losses_load[model_name]["train"],) + 1)
    
#                             # # model_name=f"Unbiased_Model_{loss_type}_lr{l_rates}_temp{temprature}"            
#                             # plt.plot(epochs_list, losses_load[model_name]["train"], label=f"Train (lr={l_rates},Sf={sharpening_factor},Nl={noise_level})")
#                             # plt.plot(epochs_list, losses_load[model_name]["val"], '--', label=f"Val   (lr={l_rates},Sf={sharpening_factor},Nl={noise_level})")
#                             train_losses = losses_load[model_name].get("train")
#                             val_losses = losses_load[model_name].get("val")
        
#                             if not train_losses or not val_losses:
#                                 print(f"[WARN] Missing loss values for '{model_name}'")
#                                 continue
        
#                             epochs_list = range(1, len(train_losses) + 1)
        
#                             # Plot with labels
#                             plt.plot(epochs_list, train_losses, label=f"Train (lr={l_rates}, Sf={sharpening_factor}, Nl={noise_level})")
#                             plt.plot(epochs_list, val_losses, '--', label=f"Val   (lr={l_rates}, Sf={sharpening_factor}, Nl={noise_level})")

# #                             plt.plot(epochs_list, results[key][trn], label=f"Train (lr={key[1]},{key[0]})")
# # plt.plot(epochs_list, results[key]['val'], '--', label=f"Val   (lr={key[1]},{key[0]})")
#     for key, (trn, val) in losses_load.items():
    
#         if "switch100" in key:
#             plt.plot(epochs_list, losses_load[key][trn], label=f"Train (lr={key[1]},{key[0]})")
#             plt.plot(epochs_list, losses_load[key][val], '--', label=f"Val   (lr={key[1]},{key[0]})")
        
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training & Validation Loss for Two Learning Rates under Two different model size")
#     plt.legend()
#     plt.grid(True)
#     plt.show()    

    plt.figure(figsize=(10,6))

    for key, (trn, val) in losses_load.items():
        epochs_list = range(1, len(losses_load[key][val]) + 1)
        # if "switch100" in key:
        keys=key.split("_")
        # plt.plot(epochs_list, losses_load[key][trn], label=f"Train (lr={keys[4]},Sf={keys[6]},Nl={keys[7]})")
        plt.plot(epochs_list, losses_load[key][val], label=f"Val   (lr={keys[4]},Sf={keys[6]},Nl={keys[7]})")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Trends of Our Model (ResNet-18 Backbone) Across Hyperparameter Variations")
    plt.legend()
    plt.grid(True)
    plt.show()  
    import pandas as pd
    results_table = []
    for temprature in TEMPERATURES:
        for l_rates in learning_rates:
            for epoch_switch in epochs_switches:
                for sharpening_factor in sharpening_factors:
                    for noise_level in noise_levels:
                        try:
                            # model_name=f"Unbiased_Model_{loss_type}_lr{l_rates}_temp{temprature}"            
                            # model_name=f"Unbiased_Model_{loss_type}_lr{l_rates}_temp{temprature}_sharp{sharpening_factor}_switch{epoch_switch}_noise{noise_level}"            
                            model_name=f"Unbiased_Model_{dataset}_{model_type}_lr{l_rates}_sharp{sharpening_factor}_switch{epoch_switch}_noise{noise_level}"            

                            
                    #         # model_name=f"Unbiased_Model_{loss_type}_lr{l_rates}_temp{temprature}"            
                    #         model_name=f"Unbiased_Model_{loss_type}_lr{l_rates}_temp{temprature}_sharp{sharpening_factor}_switch{epoch_switch}"            
                            # model_name="Unbiased_Model_kl_divergence_lr0.002_temp1.0_sharp2_switch30"
                    # # model_name=f"Unbiased_Model_{loss_type}_lr{l_rates}_temp{temprature}"            
                    #     #######Evaluation
                            
                            
                            # # Load and evaluate Model A
                            # best_model_A = ResNet18()

                            if model_type=="resnet":
                                best_model_A=ResNet18().to(DEVICE)
                            elif model_type=="efficientnet":
                                # Load pretrained EfficientNet-B0
                                best_model_A = models.efficientnet_b0(pretrained=False)
                                # Replace the classifier head
                                if dataset=="cifar10":
                                    best_model_A.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
                        
                                elif dataset=="imagenet":
                                    best_model_A.classifier[1] = nn.Linear(best_model_A.classifier[1].in_features, 200)
                            best_model_A_path = os.path.join(BEST_MODEL_SAVE_DIR, f"{model_name}_best.pth")
                        
                            best_model_A.load_state_dict(torch.load(best_model_A_path, map_location=DEVICE))
                            print("Evaluating Best Model A (trained on soft targets):")
                            # acc_A_best = evaluate_model(best_model_A, original_test_loader, DEVICE)
                            # print(f"  Test Accuracy (Model A): {acc_A_best:.2f}%")
                            
                        
                        
                            evaluation_results_devil = evaluate_model_extended(
                            model=best_model_A,
                            dataloader=original_test_loader,
                            device=DEVICE,
                            num_classes=5
                            )
                            
                            print(f"\n--- Evaluation Results: Our Model {model_name}---")
                            print(f"  Accuracy: {evaluation_results_devil['accuracy']:.2f}%")
                            print(f"  Macro Precision: {evaluation_results_devil['precision_macro']:.2f}%")
                            print(f"  Macro Recall: {evaluation_results_devil['recall_macro']:.2f}%")
                            print(f"  Macro F1-Score: {evaluation_results_devil['f1_macro']:.2f}%")
                            # print(f"  Weighted Precision: {evaluation_results_devil['precision_weighted']:.2f}%")
                            # print(f"  Weighted Recall: {evaluation_results_devil['recall_weighted']:.2f}%")
                            # print(f"  Weighted F1-Score: {evaluation_results_devil['f1_weighted']:.2f}%")
                            
                            results_table.append({
                            "Model Name": model_name,
                            "Temperature": temprature,
                            "Learning Rate": l_rates,
                            "Epoch Switch": epoch_switch,
                            "Sharpening Factor": sharpening_factor,
                            "Noise Level": noise_level,
                            "Accuracy (%)": evaluation_results_devil['accuracy'],
                            "Macro Precision (%)": evaluation_results_devil['precision_macro'],
                            "Macro Recall (%)": evaluation_results_devil['recall_macro'],
                            "Macro F1-Score (%)": evaluation_results_devil['f1_macro'],
                            # You can also log weighted metrics if you want
                        })
                        except:
                            pass
                            
    # Convert to DataFrame
    results_df = pd.DataFrame(results_table)
    # Drop one or more columns
    results_df = results_df.drop(columns=["Model Name", "Temperature"])

    # Save to CSV
    results_df.to_csv("model_evaluation_results.csv", index=False)
    
    # (Optional) Display in notebook/script
    print(results_df)
    # from torch_lr_finder import LRFinder
    # # 4. Run LR Finder
    # criterion = nn.CrossEntropyLoss()
    # temperature = 1.0  # example value

    # def distillation_loss(student_logits, teacher_logits):
    #     # sharpening_factor = 1.5 # Or an additive constant, e.g., +2.0. Needs tuning.
    #     # sharpened_logits = teacher_logits.clone() # B, C
        
    #     # # For each sample in the batch, increase the logit of its true class
    #     # batch_indices = torch.arange(sharpened_logits.size(0))
    #     # true_class_original_logits = sharpened_logits[batch_indices, labels]
        
    #     # # Method 1: Multiplicative sharpening
    #     # sharpened_logits[batch_indices, labels] = true_class_original_logits * sharpening_factor

    #     student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    #     teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    #     # KL divergence between student and teacher outputs
    #     loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
    #     # Scale as per Hinton et al. (2015)
    #     return loss * (temperature ** 2)
    # # optimizer = optim.Adam(best_model_A.parameters(), lr=0.0001) # Using Adam as requested

    # # lr_finder = LRFinder(best_model_A, optimizer, criterion, device="cuda" if torch.cuda.is_available() else "cpu")
    # # lr_finder.range_test(model_train_loader_reloaded, end_lr=10, num_iter=100)



    # # # lr_finder = LRFinder(best_model_A, optimizer, distillation_loss, device="cuda" if torch.cuda.is_available() else "cpu")
    # # # lr_finder.range_test(distillation_dataset_for_lr, end_lr=10, num_iter=100)
    # # # 5. Plot
    # # lr_finder.plot()  # Matplotlib plot
    # # lr_finder.reset()  # Reset the model and optimizer to initial state

    # from ignite.handlers import FastaiLRFinder
    # from ignite.engine import create_supervised_trainer, create_supervised_evaluator
    # unbiased_model_fresh = ResNet18() # Your actual ResNet18

    # trainer = create_supervised_trainer(unbiased_model_fresh, optimizer, criterion, device=DEVICE)
    # optimizer = optim.Adam(unbiased_model_fresh.parameters(), lr=0.001) # Using Adam as requested

    # lr_finder = FastaiLRFinder()
    # to_save = {"model": unbiased_model_fresh, "optimizer": optimizer}
    
    # with lr_finder.attach(trainer, to_save=to_save, start_lr=1e-06,end_lr=1e-02, num_iter=1000) as trainer_with_lr_finder:
    #     trainer_with_lr_finder.run(model_train_loader_reloaded)
    
    # # Get lr_finder results
    # lr_finder.get_results()
    
    # # Plot lr_finder results (requires matplotlib)
    # lr_finder.plot()
    
    # # get lr_finder suggestion for lr
    # lr_finder.lr_suggestion()
    
    # === choose an init ===
    # learning_rates=[0.001,0.002] 
    learning_rates=[0.001] 

    loss_type="cross_entropy" #kl_divergence, cross_entropy
    losses_comparison = defaultdict(lambda: {'train': [], 'val': []})
    # accuracies = defaultdict(lambda: {'train': [], 'val': []})
    if mode=="Train":
        for l_rates in learning_rates:
            # model_name=f"Unbiased_Model_{dataset}_{model_type}_lr{l_rates}_sharp{sharpening_factor}_switch{epoch_switch}_noise{noise_level}"            

            model_name=f"Comparison_Model_{dataset}_{model_type}_{loss_type}_lr{l_rates}"            
            # soup_state_dict=soup_models(loaded_bias_models)
    
            # # unbiased_model_fresh.load_state_dict(soup_state_dict)
            # comparison_model_fresh = ResNet18() # Your actual ResNet18
    
            training_losses,val_losses = train_model_with_validation(
                model_name=model_name,
                model=copy.deepcopy(unbiased_model_fresh),  #Initialize each model the same
                train_dataloader=distillation_train_loader, # Yields (img, orig_lbl)
                val_dataloader=validation_loader,     # Yields (img, orig_lbl)
                epochs=NEW_MODEL_EPOCHS,
                learning_rate=l_rates, # e.g., 1e-3 or 1e-4 for Adam
                device=DEVICE,
                loss_type=loss_type
                # temperature argument is not used for cross_entropy
            )
            losses_comparison[model_name]['train'] = training_losses
            losses_comparison[model_name]['val'] = val_losses
            

    results_table = []
    for l_rates in learning_rates:
        try:
            # model_name=f"Comparison_Model_{loss_type}_lr{l_rates}"         
            model_name=f"Comparison_Model_{dataset}_{model_type}_{loss_type}_lr{l_rates}"            

            # model_name=f"Comparison_Model_cross_entropy_lr0.001"            
    
            # Load and evaluate Model B
            # best_model_B = ResNet18()
            if model_type=="resnet":
                best_model_B=ResNet18().to(DEVICE)
            elif model_type=="efficientnet":
                # Load pretrained EfficientNet-B0
                best_model_B = models.efficientnet_b0(pretrained=False)
                # Replace the classifier head
                if dataset=="cifar10":
                    best_model_B.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        
                elif dataset=="imagenet":
                    best_model_B.classifier[1] = nn.Linear(best_model_B.classifier[1].in_features, 200)    
            best_model_B_path = os.path.join(BEST_MODEL_SAVE_DIR, f"{model_name}_best.pth")
        
            best_model_B.load_state_dict(torch.load(best_model_B_path, map_location=DEVICE))
            print("\nEvaluating Best Model B (trained on hard targets):")
            # acc_B_best = evaluate_model(best_model_B, original_test_loader, DEVICE)
            # print(f"  Test Accuracy (Model B): {acc_B_best:.2f}%")
            
            
            # print(f"\nSummary of Test Accuracies (Best Models):")
            # print(f"  Best Model A (Soft Targets - KL Div): {acc_A_best:.2f}% (saved at {best_model_A_path})")
            # print(f"  Best Model B (Hard Targets - Cross Entropy): {acc_B_best:.2f}% (saved at {best_model_B_path})")
            
            evaluation_results_comparison = evaluate_model_extended(
            model=best_model_B,
            dataloader=original_test_loader,
            device=DEVICE,
            num_classes=5
            )
    
    
            print("\n--- Evaluation Results: Comparison ---")
            print(f"  Accuracy: {evaluation_results_comparison['accuracy']:.2f}%")
            print(f"  Macro Precision: {evaluation_results_comparison['precision_macro']:.2f}%")
            print(f"  Macro Recall: {evaluation_results_comparison['recall_macro']:.2f}%")
            print(f"  Macro F1-Score: {evaluation_results_comparison['f1_macro']:.2f}%")
            print(f"  Weighted Precision: {evaluation_results_comparison['precision_weighted']:.2f}%")
            print(f"  Weighted Recall: {evaluation_results_comparison['recall_weighted']:.2f}%")
            print(f"  Weighted F1-Score: {evaluation_results_comparison['f1_weighted']:.2f}%")
            
            results_table.append({
            "Learning Rate": l_rates,
            "Accuracy (%)": evaluation_results_devil['accuracy'],
            "Macro Precision (%)": evaluation_results_devil['precision_macro'],
            "Macro Recall (%)": evaluation_results_devil['recall_macro'],
            "Macro F1-Score (%)": evaluation_results_devil['f1_macro'],
            # You can also log weighted metrics if you want
        })
        except:
            pass
        
        
    # Convert to DataFrame
    results_df = pd.DataFrame(results_table)
    # Drop one or more columns
    # results_df = results_df.drop(columns=["Model Name", "Temperature"])

    # Save to CSV
    results_df.to_csv("comparison_model_evaluation_results.csv", index=False)
    
    # (Optional) Display in notebook/script
    print(results_df)
    
    
    import pickle
    path_for_losses_c=f"{dataset}_{model_type}_losses_comparison.pkl"
    # Save as regular dict
    if mode=="Train":
        with open(path_for_losses_c, 'wb') as f:
            pickle.dump(dict(losses_comparison), f)
            
        # Load losses from file
    with open(path_for_losses_c, 'rb') as f:
        losses_load = pickle.load(f)
    
    # # --- Train Comparison Model (Model B) ---
    # comparison_model_fresh = ResNet18() # Your actual ResNet18
    # best_comparison_model_path = train_model_with_validation(
    #     model_name="Comparison_Model_CrossEntropy",
    #     model=comparison_model_fresh,
    #     train_dataloader=distillation_train_loader, # Yields (img, orig_lbl)
    #     val_dataloader=validation_loader,     # Yields (img, orig_lbl)
    #     epochs=NEW_MODEL_EPOCHS,
    #     learning_rate=NEW_MODEL_LR, # e.g., 1e-3 or 1e-4 for Adam
    #     device=DEVICE,
    #     loss_type="cross_entropy"
    #     # temperature argument is not used for cross_entropy
    # )
    
    
    
    # #######Evaluation
    
    
    # # Load and evaluate Model A
    # best_model_A = ResNet18()
    # model_name="Unbiased_Model_KLDiv_Exclusive"
    # best_model_A_path = os.path.join(BEST_MODEL_SAVE_DIR, f"{model_name}_best.pth")

    # best_model_A.load_state_dict(torch.load(best_model_A_path, map_location=DEVICE))
    # print("Evaluating Best Model A (trained on soft targets):")
    # # acc_A_best = evaluate_model(best_model_A, original_test_loader, DEVICE)
    # # print(f"  Test Accuracy (Model A): {acc_A_best:.2f}%")
    
    # evaluation_results_devil = evaluate_model_extended(
    # model=best_model_A,
    # dataloader=original_test_loader,
    # device=DEVICE,
    # num_classes=10
    # )
    
    # print("\n--- Evaluation Results: Our Model ---")
    # print(f"  Accuracy: {evaluation_results_devil['accuracy']:.2f}%")
    # print(f"  Macro Precision: {evaluation_results_devil['precision_macro']:.2f}%")
    # print(f"  Macro Recall: {evaluation_results_devil['recall_macro']:.2f}%")
    # print(f"  Macro F1-Score: {evaluation_results_devil['f1_macro']:.2f}%")
    # print(f"  Weighted Precision: {evaluation_results_devil['precision_weighted']:.2f}%")
    # print(f"  Weighted Recall: {evaluation_results_devil['recall_weighted']:.2f}%")
    # print(f"  Weighted F1-Score: {evaluation_results_devil['f1_weighted']:.2f}%")
    
    # print("\n  Confusion Matrix:")
    # print(evaluation_results_devil['confusion_matrix'])
    
    # print("\n  Per-Class Metrics:")
    # for class_name, metrics in evaluation_results_devil['per_class_metrics'].items():
    #     print(f"    {class_name}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}, Support={metrics['support']}")

    # # Load and evaluate Model B
    # best_model_B = ResNet18()
    # model_name="Comparison_Model_CrossEntropy"
    # best_model_B_path = os.path.join(BEST_MODEL_SAVE_DIR, f"{model_name}_best.pth")

    # best_model_B.load_state_dict(torch.load(best_model_B_path, map_location=DEVICE))
    # print("\nEvaluating Best Model B (trained on hard targets):")
    # # acc_B_best = evaluate_model(best_model_B, original_test_loader, DEVICE)
    # # print(f"  Test Accuracy (Model B): {acc_B_best:.2f}%")
    
    
    # # print(f"\nSummary of Test Accuracies (Best Models):")
    # # print(f"  Best Model A (Soft Targets - KL Div): {acc_A_best:.2f}% (saved at {best_model_A_path})")
    # # print(f"  Best Model B (Hard Targets - Cross Entropy): {acc_B_best:.2f}% (saved at {best_model_B_path})")
    
    # evaluation_results_comparison = evaluate_model_extended(
    # model=best_model_B,
    # dataloader=original_test_loader,
    # device=DEVICE,
    # num_classes=10
    # )

    # print("\n--- Evaluation Results: Comparison ---")
    # print(f"  Accuracy: {evaluation_results_comparison['accuracy']:.2f}%")
    # print(f"  Macro Precision: {evaluation_results_comparison['precision_macro']:.2f}%")
    # print(f"  Macro Recall: {evaluation_results_comparison['recall_macro']:.2f}%")
    # print(f"  Macro F1-Score: {evaluation_results_comparison['f1_macro']:.2f}%")
    # print(f"  Weighted Precision: {evaluation_results_comparison['precision_weighted']:.2f}%")
    # print(f"  Weighted Recall: {evaluation_results_comparison['recall_weighted']:.2f}%")
    # print(f"  Weighted F1-Score: {evaluation_results_comparison['f1_weighted']:.2f}%")
    
    # print("\n  Confusion Matrix:")
    # print(evaluation_results_comparison['confusion_matrix'])
    
    # print("\n  Per-Class Metrics:")
    # for class_name, metrics in evaluation_results_comparison['per_class_metrics'].items():
    #     print(f"    {class_name}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}, Support={metrics['support']}")
