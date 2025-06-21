import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets,models
import numpy as np
import pickle # For saving/loading indices
import os
import random
import torch.nn.functional as F
import shutil


dataset="cifar10" #cifar10, imagenet
model_type="efficientnet" #resnet, efficientnet

# --- Configuration ---
num_bias_models = 5 # Reduced for faster demonstration of this specific feature
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS_PER_MODEL = 100 # Reduced for faster demo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALIDATION_SPLIT_RATIO = 0.1
SAVED_MODELS_DIR = f'./{dataset}/{model_type}/saved_bias_models_v2' # New dir for this version
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# import os
# def print_folder_tree(startpath, prefix=''):
#     folders = [f for f in sorted(os.listdir(startpath)) if os.path.isdir(os.path.join(startpath, f))]
#     for idx, folder in enumerate(folders):
#         path = os.path.join(startpath, folder)
#         is_last = idx == len(folders) - 1
#         connector = '└── ' if is_last else '├── '
#         print(prefix + connector + folder)
#         extension = '    ' if is_last else '│   '
#         print_folder_tree(path, prefix + extension)
# # Usage:
# print_folder_tree('./data/tiny-imagenet-200/')


print(f"Using device: {DEVICE}")

MANUAL_SEED = 42
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(MANUAL_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- 1. ResNet Model Definition (Same as before) ---
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.linear = nn.Linear(512, num_classes)
#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride_val in strides:
#             layers.append(block(self.in_channels, out_channels, stride_val))
#             self.in_channels = out_channels
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


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
        return ResNet(ResidualBlock).to(device)
    elif dataset=="imagenet":
        return ResNet(ResidualBlock,200).to(device)

# --- 2. Data Loading and Preprocessing ---
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.Resize(32),  # ensure consistent size
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if dataset=="cifar10":
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    
elif dataset=="imagenet":

    full_train_dataset = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform=transform_train)
    # test_dataset = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=transform_test)
    
    # train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_dir = './data/tiny-imagenet-200/val'
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    images_dir = os.path.join(val_dir, 'images')
    new_base_dir = os.path.join(val_dir, 'organized')
    
    if not os.path.exists(new_base_dir):
        os.makedirs(new_base_dir)
        # Step 1: Read annotations
        with open(annotations_file, 'r') as f:
            for line in f.readlines():
                img_file, class_id = line.strip().split('\t')[:2]
                class_dir = os.path.join(new_base_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                # Move or copy image into its class folder
                src = os.path.join(images_dir, img_file)
                dst = os.path.join(class_dir, img_file)
                shutil.copy(src, dst)  # or shutil.move(src, dst)
                
    test_dataset = datasets.ImageFolder(os.path.join(val_dir, 'organized'), transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# --- 3. Prepare and Split Indices (Same as before) ---

indices_file = f'{dataset}_train_val_indices_split_{num_bias_models}_v2.pkl' 

all_model_train_indices = []
global_val_indices = []

if os.path.exists(indices_file):
    print(f"Loading pre-split indices from {indices_file}")
    with open(indices_file, 'rb') as f:
        saved_indices_data = pickle.load(f)
        global_val_indices = saved_indices_data['val_indices']
        all_model_train_indices = saved_indices_data['train_indices_per_model']
else:
    print("Generating and splitting indices...")
    num_total_train_samples = len(full_train_dataset)
    original_train_indices = list(range(num_total_train_samples))
    np.random.shuffle(original_train_indices)
    val_split_point = int(np.floor(VALIDATION_SPLIT_RATIO * num_total_train_samples))
    global_val_indices = original_train_indices[:val_split_point]
    remaining_train_indices_for_models = original_train_indices[val_split_point:]
    print(f"Total original train samples: {num_total_train_samples}")
    print(f"Global validation samples: {len(global_val_indices)}")
    print(f"Remaining samples for bias model training: {len(remaining_train_indices_for_models)}")
    split_indices_np = np.array_split(np.array(remaining_train_indices_for_models), num_bias_models)
    all_model_train_indices = [list(arr) for arr in split_indices_np]
    indices_to_save = {
        'val_indices': global_val_indices,
        'train_indices_per_model': all_model_train_indices
    }
    with open(indices_file, 'wb') as f:
        pickle.dump(indices_to_save, f)
    print(f"Indices saved to {indices_file}")

# --- 4. Create Global Validation Set and DataLoader (Same as before) ---
global_val_subset = Subset(full_train_dataset, global_val_indices)
global_val_loader = DataLoader(global_val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"\nGlobal validation set created with {len(global_val_subset)} samples.")

# --- 5. Create Training Subsets and DataLoaders for each model (Same as before) ---
train_loaders_for_bias_models = []
for i in range(num_bias_models):
    model_indices = all_model_train_indices[i]
    # print(f"Model {i}: {len(model_indices)} training samples. First 5 indices: {model_indices[:5]}")
    model_subset = Subset(full_train_dataset, model_indices)
    # For the "other models' training data" evaluation, we want shuffle=False to get a consistent view,
    # but for actual training of model_i, its loader should have shuffle=True.
    # So, we create two types of loaders for these partitions if needed, or just use the shuffled one.
    # For simplicity here, we'll use the existing train_loaders_for_bias_models which have shuffle=True.
    # This means the "validation" on other model's data will be on shuffled batches.
    # A more rigorous approach would be to create separate DataLoaders for evaluation with shuffle=False.
    model_train_loader = DataLoader(model_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    train_loaders_for_bias_models.append(model_train_loader)
print(f"Created {len(train_loaders_for_bias_models)} specific training DataLoaders (shuffle=True).")

# --- 6. Initialize Bias Models (Same as before) ---


    
# bias_models = [ResNet18().to(DEVICE) for _ in range(num_bias_models)]
bias_models=[]
for i in range(num_bias_models):
    # MANUAL_SEED = 42
    MANUAL_SEED = random.randint(0, 2**32 - 1)  # Generate a random 32-bit seed

    random.seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)
    torch.manual_seed(MANUAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MANUAL_SEED)
    if model_type=="resnet":
        bias_models.append(ResNet18().to(DEVICE))
    elif model_type=="efficientnet":
        # Load pretrained EfficientNet-B0
        model = models.efficientnet_b0(pretrained=False)
        # Replace the classifier head
        if dataset=="cifar10":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

        elif dataset=="imagenet":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
        bias_models.append(model.to(DEVICE))
        
if model_type=="efficientnet":
    del model

import random

def random_exclude(n, exclude):
    candidates = [i for i in range(n + 1) if i != exclude]
    return random.choice(candidates)


# --- 7. Training and Validation Loop ---
criterion = nn.CrossEntropyLoss()
from torch.optim.lr_scheduler import ReduceLROnPlateau


for model_idx, model in enumerate(bias_models):
    # if model_idx>0:
    print(f"\n--- Training Bias Model {model_idx} ---")
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Using Adam as requested

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    model_train_loader = train_loaders_for_bias_models[model_idx]
    # best_val_accuracy_on_global_val = 0.0 # For saving the best version of this model
    best_val_loss_on_global_val = float('inf')

    model_save_path = os.path.join(SAVED_MODELS_DIR, f"bias_model_{model_idx}_best.pth")

    for epoch in range(EPOCHS_PER_MODEL):
        # --- Training Phase ---
        model.train()
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, (inputs, targets) in enumerate(model_train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
        avg_train_loss = running_loss_train / len(model_train_loader)
        train_accuracy = 100. * correct_train / total_train
        print(f"Model {model_idx} | Epoch {epoch+1}/{EPOCHS_PER_MODEL} | Train Loss: {avg_train_loss:.3f} | Train Acc: {train_accuracy:.3f}%")

        # --- Primary Validation Phase (on global_val_loader) ---
        model.eval()
        val_loss_global = 0
        correct_val_global = 0
        total_val_global = 0
        with torch.no_grad():
            for inputs, targets in global_val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss_global += loss.item()
                _, predicted = outputs.max(1)
                total_val_global += targets.size(0)
                correct_val_global += predicted.eq(targets).sum().item()
        avg_val_loss_global = val_loss_global / len(global_val_loader)
        val_accuracy_global = 100. * correct_val_global / total_val_global
        print(f"Model {model_idx} | Epoch {epoch+1} | Global Val Loss: {avg_val_loss_global:.3f} | Global Val Acc: {val_accuracy_global:.3f}%")


        # --- Additional Evaluation: Current model_idx on OTHER models' training data ---
        print(f"  Model {model_idx} | Epoch {epoch+1} | Evaluating on other models' training data partitions:")
        model.eval() # Ensure model is in eval mode
        # val_idx=random_exclude(num_bias_models-1, model_idx)  # Random number from 0 to 5, excluding 3

        # # for other_model_data_idx in range(num_bias_models):
        # #     if other_model_data_idx == model_idx:
        # #         continue # Skip evaluating on its own training data for this specific check

        # other_model_train_loader = train_loaders_for_bias_models[val_idx]
        # # Note: other_model_train_loader has shuffle=True and train_transforms.
        # # For a strict "validation" feel, you might want shuffle=False and test_transforms
        # # on these partitions, requiring separate DataLoaders.
        # # Here, we're checking performance on the *actual data distribution* another model trains on.
        
        cross_val_loss = 0
        cross_val_correct = 0
        cross_val_total = 0
        
        len_val=0
        with torch.no_grad():
            for val_idx in range(num_bias_models):
                if model_idx!=val_idx:
                    other_model_train_loader = train_loaders_for_bias_models[val_idx]
                    len_val+=len(other_model_train_loader)
                    for inputs, targets in other_model_train_loader: # Iterates once through this data
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        cross_val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        cross_val_total += targets.size(0)
                        cross_val_correct += predicted.eq(targets).sum().item()
                    
        avg_val_loss_global = cross_val_loss / len_val
        val_accuracy_global = 100. * correct_val_global / total_val_global

        if len(other_model_train_loader) > 0 : # Check if the loader is not empty
            avg_cross_val_loss = cross_val_loss / len(other_model_train_loader)
            cross_val_accuracy = 100. * cross_val_correct / cross_val_total
            print(f"    Model {model_idx} on Model {val_idx}'s data: Loss: {avg_cross_val_loss:.3f}, Acc: {cross_val_accuracy:.3f}%")
        else:
            print(f"    Model {model_idx} on Model {val_idx}'s data: Loader is empty (no samples).")


        scheduler.step(avg_val_loss_global)

        # Save the best model based on global validation set
        if avg_val_loss_global < best_val_loss_on_global_val:
            print(f"  Global Val accuracy improved for model {model_idx} {val_accuracy_global:.3f}%). Saving model to {model_save_path}")
            best_val_loss_on_global_val = avg_val_loss_global
            torch.save(model.state_dict(), model_save_path)

    print(f"Finished training model {model_idx}. Best global validation accuracy: {best_val_loss_on_global_val:.3f}% (saved at {model_save_path})")

    # Optional: Evaluate the best saved model on the official test set
    if os.path.exists(model_save_path): # Check if a model was saved
        print(f"Evaluating best saved model {model_idx} on the official test set...")
        best_model_state = torch.load(model_save_path)
        model.load_state_dict(best_model_state)
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()
                
        print(f"Best Model {model_idx} Test Set: Avg loss: {test_loss/len(test_loader):.4f}, Accuracy: {100.*correct_test/total_test:.3f}%")
    else:
        print(f"No model saved for model {model_idx} as validation accuracy might not have improved.")


print("\n--- Process Complete ---")


import torch
import torch.nn as nn
import torchvision # Only needed if you were also loading datasets, but good practice
import os

# --- Configuration ---
# num_bias_models = 10 # Or 10, etc. MUST match the number of models you trained and saved
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SAVED_MODELS_DIR = './saved_bias_models_v2' # Directory where model weights are saved
# SAVED_MODELS_DIR = f'./{dataset}/saved_bias_models_v2' # New dir for this version
SAVED_MODELS_DIR = f'./{dataset}/{model_type}/saved_bias_models_v2' # New dir for this version

print(f"Using device: {DEVICE}")
print(f"Attempting to load {num_bias_models} bias models from: {SAVED_MODELS_DIR}")


# --- 2. Initialize Model Instances ---
loaded_bias_models = []
for i in range(num_bias_models):
    # model = ResNet18() # Create a new instance of the model architecture
    # loaded_bias_models.append(model)
    if model_type=="resnet":
        loaded_bias_models.append(ResNet18().to(DEVICE))
    elif model_type=="efficientnet":
        # Load pretrained EfficientNet-B0
        model = models.efficientnet_b0(pretrained=False)
        # Replace the classifier head
        if dataset=="cifar10":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

        elif dataset=="imagenet":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
        loaded_bias_models.append(model.to(DEVICE))

print(f"\nInitialized {len(loaded_bias_models)} fresh ResNet18 model instances.")

# --- 3. Load Saved Weights for Each Model ---
successfully_loaded_count = 0
for model_idx, model in enumerate(loaded_bias_models):
    model_filename = f"bias_model_{model_idx}_best.pth"
    model_path = os.path.join(SAVED_MODELS_DIR, model_filename)

    if os.path.exists(model_path):
        print(f"Loading weights for model {model_idx} from: {model_path}")
        try:
            # Load the state dictionary. Map location to ensure compatibility
            # if the model was saved on a different device (e.g., GPU vs CPU).
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE) # Move model to the target device
            model.eval()     # Set to evaluation mode by default after loading
            print(f"  Successfully loaded weights for model {model_idx} and moved to {DEVICE}.")
            successfully_loaded_count += 1
        except Exception as e:
            print(f"  Error loading weights for model {model_idx}: {e}")
            print(f"  Model {model_idx} will have its initial (random) weights.")
    else:
        print(f"Warning: Weight file not found for model {model_idx} at {model_path}.")
        print(f"  Model {model_idx} will have its initial (random) weights.")
        # Even if not found, move the randomly initialized model to device
        model.to(DEVICE)
        model.eval()


print(f"\n--- Model Loading Complete ---")
print(f"Successfully loaded weights for {successfully_loaded_count} out of {num_bias_models} models.")
print(f"All {len(loaded_bias_models)} models are in the `loaded_bias_models` list and on device '{DEVICE}'.")
print("They are set to evaluation mode (.eval()).")

# You can now use these models, for example:
# if loaded_bias_models:
#   first_loaded_model = loaded_bias_models[0]
#   # dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE) # Example CIFAR10 input
#   # with torch.no_grad():
#   #   output = first_loaded_model(dummy_input)
#   #   print(f"Output shape from first loaded model: {output.shape}")


