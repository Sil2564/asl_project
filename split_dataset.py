import os
import shutil
import random

# Percorsi
original_data_dir = r'C:\Users\busti\Desktop\asl_project\asl_alphabet_train'
output_base_dir = "dataset"

# Percentuali di divisione
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Crea le directory
for split in ['train', 'val', 'test']:
    for label in os.listdir(original_data_dir):
        os.makedirs(os.path.join(output_base_dir, split, label), exist_ok=True)

# Per ogni lettera (A-Z)
for label in os.listdir(original_data_dir):
    label_path = os.path.join(original_data_dir, label)
    if not os.path.isdir(label_path):
        continue

    images = [img for img in os.listdir(label_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    total = len(images)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    for i, img in enumerate(images):
        src = os.path.join(label_path, img)
        if not os.path.isfile(src):
            continue  # Salta se non è un file
        if i < train_count:
            dst = os.path.join(output_base_dir, 'train', label, img)
        elif i < train_count + val_count:
            dst = os.path.join(output_base_dir, 'val', label, img)
        else:
            dst = os.path.join(output_base_dir, 'test', label, img)
        shutil.copy(src, dst)

print("✅ Divisione completata con successo!")
