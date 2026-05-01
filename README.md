# Object Detection on Raspberry Pi 5 with Google Coral USB

This repository provides a complete guide and all necessary scripts to run and train a custom object detection model on Raspberry Pi 5 using Google Coral USB Accelerator and TensorFlow Lite.

Tutorial video: https://youtu.be/AE6fcQHJ_lE

---

## Credits

This project is based on the original tutorial by: https://www.youtube.com/watch?v=fVmAeK-GLXA&t=837s

Commands and scripts have been modified to work correctly on Raspberry Pi 5.

---

## Overview

This project walks you through:
- Setting up Python 3.9.12 on Raspberry Pi 5 using pyenv
- Installing and configuring Google Coral USB EdgeTPU
- Running real-time object detection with TensorFlow Lite
- Annotating custom images with LabelImg
- Training a custom EfficientDet Lite model on Google Colab
- Deploying the trained model back to Raspberry Pi 5

---

## Requirements

**Hardware:**
- Raspberry Pi 5
- Google Coral USB Accelerator
- USB Camera

**Software:**
- Raspberry Pi OS (64-bit recommended)
- Python 3.9.12 (via pyenv)
- Google Coral EdgeTPU runtime
- TensorFlow Lite runtime
- LabelImg (for annotation)

---

## Repository Structure

```
.
├── detect.py                          # Main object detection script (runs inference on camera)
├── train.py                           # Training script for Google Colab
├── test.py                            # Copies metadata from best.tflite to best_edgetpu.tflite
├── labelImg.py                        # Patched LabelImg source file
├── canvas.py                          # Patched canvas source file for LabelImg
├── labelimg.sh                        # Shell script to install and patch LabelImg
├── tflite_custom_model_edgetpu.ipynb  # Google Colab notebook for training
├── tensorflow-lite-bullseye-main.zip  # TensorFlow Lite setup package
└── rpi5.txt                           # Additional notes for Raspberry Pi 5 setup
```

---

## Step 1: Install Python 3.9.12 with pyenv

Open a terminal on your Raspberry Pi and run each command one at a time.

**1.1 Update and upgrade the system:**

```bash
sudo apt-get update
```

```bash
sudo apt-get upgrade
```

**1.2 Create the project folder and enter it:**

```bash
mkdir freedomtech
```

```bash
cd freedomtech
```

**1.3 Install pyenv:**

```bash
curl https://pyenv.run | bash
```

**1.4 Add pyenv to your shell configuration (run each line separately):**

```bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
```

```bash
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
```

```bash
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

```bash
exec "$SHELL"
```

**1.5 Install required build dependencies:**

```bash
sudo apt-get install --yes libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses-dev xz-utils tk-dev libgdbm-dev lzma tcl-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev wget curl make build-essential openssl
```

**1.6 Install Python 3.9.12:**

```bash
pyenv install 3.9.12
```

**1.7 Set Python 3.9.12 as the local version:**

```bash
pyenv local 3.9.12
```

**1.8 Verify the Python version:**

```bash
python --version
```

---

## Step 2: Install Google Coral USB EdgeTPU

**2.1 Create a Python virtual environment:**

```bash
python3 -m venv .venv
```

**2.2 Activate the virtual environment:**

```bash
source .venv/bin/activate
```

**2.3 Create the keyrings directory:**

```bash
sudo mkdir -p /etc/apt/keyrings
```

**2.4 Download and add the Coral GPG key:**

```bash
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/coral-edgetpu.gpg
```

**2.5 Add the Coral package repository:**

```bash
echo "deb [signed-by=/etc/apt/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
```

**2.6 Update the package list:**

```bash
sudo apt-get update
```

**2.7 Install the EdgeTPU runtime:**

```bash
sudo apt-get install libedgetpu1-std
```

Reboot the Raspberry Pi after installation completes.

**2.8 After reboot, re-enter the project folder and activate the virtual environment:**

```bash
cd freedomtech
```

```bash
source .venv/bin/activate
```

---

## Step 3: Install TensorFlow Lite

Extract `tensorflow-lite-bullseye-main.zip` into the `freedomtech` folder, then run:

```bash
cd tensorflow-lite-bullseye-main
chmod 775 tensorflow-lite.sh
bash tensorflow-lite.sh
```

The `tensorflow-lite.sh` script will:
- Upgrade pip
- Install tflite-runtime
- Clone the TensorFlow examples repository

After the script finishes, navigate to the object detection example and run setup:

```bash
cd examples/lite/examples/object_detection/raspberry_pi/
bash setup.sh
```

Check the OpenCV version installed and update `requirements.txt` if needed before running `setup.sh`.

Copy `detect.py` from this repository into the `freedomtech` folder. Make sure the Python version inside `detect.py` matches 3.9.12.

Run a quick test with the pre-built model:

```bash
python detect.py --model efficientdet_lite0_edgetpu.tflite --enableEdgeTPU
```

---

## Step 4: Install and Configure LabelImg

Open a new terminal that is NOT inside the virtual environment, then run:

```bash
sudo rm /usr/lib/python3.13/EXTERNALLY-MANAGED

cd tflite-coral-usb-custom-main/
```

Check the Python version inside `labelimg.sh` and update it to match your system, then install:

```bash
chmod 775 labelimg.sh
bash labelimg.sh
```

If you see this warning after launching LabelImg:

```
QStandardPaths: wrong permissions on runtime directory
```

Fix it with:

```bash
chmod 0700 /run/user/1000
```

Launch LabelImg:

```bash
labelImg
```

---

## Step 5: Annotate Images

1. Create a folder named `images` and copy your raw images into it.
2. Open LabelImg, set the image directory to your `images` folder.
3. Draw bounding boxes around objects and assign class labels.
4. Save annotations in Pascal VOC (XML) format.

After annotation, organize your dataset:

```
freedomtech/
├── train/
│   ├── image1.jpg
│   ├── image1.xml
│   └── ...
└── validate/
    ├── image2.jpg
    ├── image2.xml
    └── ...
```

A general rule of thumb is to use 80% of images for training and 20% for validation.

---

## Step 6: Train the Model on Google Colab

Compress the dataset folder:

```bash
sudo zip -r freedomtech.zip freedomtech/*
```

Upload `freedomtech.zip` to Google Drive.

Open `tflite_custom_model_edgetpu.ipynb` in Google Colab:
- Go to Runtime > Change runtime type
- Set Hardware accelerator to GPU (T4)
- Run each cell sequentially

Before running cell 8, upload `train.py` to the Colab session. Open `train.py` and update the class names to match your annotation labels:

```python
train_data = object_detector.DataLoader.from_pascal_voc(
    'freedomtech/train',
    'freedomtech/train',
    ['your_class_1', 'your_class_2']   # Replace with your actual class names
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'freedomtech/validate',
    'freedomtech/validate',
    ['your_class_1', 'your_class_2']   # Replace with your actual class names
)
```

The training script uses `EfficientDet Lite0` with the following default configuration:
- Batch size: 4
- Epochs: 100
- Full model fine-tuning: enabled

After training completes, the output file `best.tflite` will be exported.

Before running cell 12, upload `test.py` to the Colab session. This script copies metadata from `best.tflite` into `best_edgetpu.tflite` so the model is compatible with the Coral EdgeTPU compiler.

If you encounter errors or need to modify the code, always reset the environment fully:
1. Runtime > Change runtime type (re-select GPU)
2. Edit > Clear all outputs
3. Runtime > Disconnect and delete runtime
4. Restart session

---

## Step 7: Deploy the Trained Model to Raspberry Pi 5

Download `best_edgetpu.tflite` from Google Colab and copy it to the `freedomtech` folder on your Raspberry Pi.

Activate the virtual environment and run object detection with your custom model:

```bash
source .venv/bin/activate
python detect.py --model best_edgetpu.tflite --enableEdgeTPU
```

---

## Notes

- Always remember to rename class labels in `train.py` to match your dataset before training.
- The `labelimg.sh` script must use the correct Python version matching your system installation.
- Using `libedgetpu1-std` (standard speed) is recommended for stable operation. The `libedgetpu1-max` package runs the TPU at maximum clock speed but may cause the device to run hot.
- The `test.py` script requires both `best.tflite` and `best_edgetpu.tflite` to exist in the same directory before running.

---

## License

This project is based on the TensorFlow Examples repository, licensed under the Apache License 2.0.
