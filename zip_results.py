import zipfile
import os

files_to_zip = [
    'app.py',
    'predict.py',
    'requirements.txt',
    'README.md',
    'training_results.png',
    'train_model.py',
    'models/flower_cnn_model.h5'
]

zip_name = 'flower_classification_model.zip'

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(file)
            print(f"Added {file} to zip.")
        else:
            print(f"Warning: {file} not found.")

print(f"\nAll files packaged into {zip_name}")
