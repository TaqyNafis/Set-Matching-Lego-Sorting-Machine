# Set-Matching-Lego-Sorting-Machine
A LEGO sorting machine prototype that detects moving LEGO parts using MOG2 background subtraction, classifies them with a CNN model, and uses the Rebrickable API to match parts against a selected LEGO set inventory.  Developed for my graduation Final Year Project.

## Project Overview

This project is an automated LEGO sorting machine that uses computer vision, deep learning, set matching, and a servo-based mechanical gate to classify and sort LEGO parts.

The system detects a LEGO brick on a conveyor using OpenCV MOG2 background subtraction, captures multiple images of the brick, classifies the part using a trained ResNet-50 model, checks whether the predicted part belongs to the selected LEGO set, and moves a servo gate to direct the brick to either the accepted or rejected path.

## Demo Video

<p align="center">
  <a href="https://youtu.be/YiQE_hScU2o">
    <img src="https://img.youtube.com/vi/YiQE_hScU2o/maxresdefault.jpg" alt="Demo Video" width="800">
  </a>
</p>

<p align="center">
  <a href="https://youtu.be/YiQE_hScU2o">Watch the demo video on YouTube</a>
</p>

## Main System Flow

```text
Camera input
→ MOG2 background subtraction
→ Object detection and burst image capture
→ ResNet-50 classification
→ Majority voting
→ LEGO set matching
→ Quantity limit checking
→ Servo gate sorting
```

## Main Files

| File | Description |
|---|---|
| `main.py` | Main entry point for running the complete sorting system. |
| `detection.py` | Handles camera input, MOG2 background subtraction, object detection, and burst image capture. |
| `sorting_pipeline.py` | Handles classification, majority voting, LEGO set matching, quantity checking, and servo sorting command. |
| `Classify.py` | Contains model loading, image preprocessing, class mapping, and prediction functions. |
| `Classification_train.py` | Training script used to train and fine-tune the ResNet-50 LEGO part classifier. |
| `Set_data_read/Dataset_read.py` | Generates LEGO set JSON files using the local Rebrickable CSV database. |
| `Set_data_read/Dataset_read_api.py` | Generates LEGO set JSON files using the Rebrickable API. |
| `gate_control2.py` | Servo control script used on the Raspberry Pi. This file should be placed on the Raspberry Pi at the script path configured in `sorting_pipeline.py`. |

## Folders

| Folder | Description |
|---|---|
| `checkpoints/` | Contains the trained model checkpoint. Required file: `best_model.pt`. |
| `artifacts/` | Contains model class mapping and training/evaluation results. Required file: `class_to_idx.json`. Optional files: `training_history.json`, `final_metrics.json`. |
| `Set_data_read/` | Contains scripts and data used for LEGO set information. |
| `Set_data_read/Database/` | Contains local Rebrickable CSV database files. |
| `Set_data_read/Set_data/` | Contains generated LEGO set JSON files used by the sorting pipeline. |
| `captures/` | Stores temporary captured brick images during detection. |
| `Dataset/` | Contains the training dataset. Expected structure: `Dataset/renders/` and `Dataset/photos/`. |
| `images/` | Contains additional project images, such as hardware setup and Raspberry Pi servo pin setup images. |

## Expected File Structure

```text
FYP_code/
│
├── main.py
├── detection.py
├── sorting_pipeline.py
├── Classify.py
├── Classification_train.py
│
├── checkpoints/
│   └── best_model.pt
│
├── artifacts/
│   ├── class_to_idx.json
│   ├── training_history.json
│   └── final_metrics.json
│
├── captures/
│
├── Dataset/
│   ├── renders/
│   │   ├── [part_class_folder]/
│   │   └── ...
│   │
│   └── photos/
│       ├── [part_class_folder]/
│       └── ...
│
├── images/
│   ├── Hardware_setup_front.jpg
│   ├── Hardware_setup_top.jpg
│   └── Pi_servo_pin_setup.jpg
│
└── Set_data_read/
    ├── __init__.py
    ├── Dataset_read.py
    ├── Dataset_read_api.py
    │
    ├── Database/
    │   └── .gitkeep
    │
    └── Set_data/
        ├── test.json
        └── 10696-1.json
```

> Note: The `Dataset/` folder is not included in this repository due to file size. The `Set_data_read/Database/` folder is also empty in this repository because the Rebrickable CSV database files are not included.

## Dataset and Database Files

### Training Dataset

The dataset used in this project is a combination of the following LEGO brick image datasets:

- [LDRAW based renders of LEGO bricks moving on a conveyor belt with extracted models](https://mostwiedzy.pl/en/open-research-data/ldraw-based-renders-of-lego-bricks-moving-on-a-conveyor-belt-with-extracted-models,202106131552249793221-0?_share=1e566d05eb6db0dd)  
  T. M. Boiński, K. Zawora, S. Zaraziński, and B. Śledź, Version 3.0, Gdańsk University of Technology, 2021, doi: `10.34808/xfgk-6f77`.

- [Images of LEGO bricks](https://mostwiedzy.pl/en/open-research-data/images-of-lego-bricks,202309140837142278781-0)  
  T. M. Boiński, Version 1.1, Gdańsk University of Technology, 2021, doi: `10.34808/arsb-4268`.

- [LEGO bricks for training classification network](https://www.mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,618104539639776-0)  
  T. M. Boiński, S. Zaraziński, and B. Śledź, Version 1.0, Gdańsk University of Technology, 2021, doi: `10.34808/3qfs-rt94`.

After downloading or preparing the dataset, place it in the main project directory using this structure:

```text
FYP_code/
└── Dataset/
    ├── renders/
    └── photos/
```

The `Dataset/` folder is only required for training or retraining the model. It is not required for running the trained sorting system if `checkpoints/best_model.pt` and `artifacts/class_to_idx.json` are already provided.

### Rebrickable CSV Database

The `Set_data_read/Database/` folder is kept in the repository using `.gitkeep`, but the CSV database files are not included because they are large and can be downloaded separately.

The Rebrickable LEGO catalog database can be downloaded from:

[Rebrickable CSV File Downloads](https://rebrickable.com/downloads/)

The required CSV files for local set matching are:

```text
sets.csv
inventories.csv
inventory_parts.csv
parts.csv
```

After downloading and extracting the files, place them inside:

```text
FYP_code/
└── Set_data_read/
    └── Database/
        ├── sets.csv
        ├── inventories.csv
        ├── inventory_parts.csv
        └── parts.csv
```

Alternatively, the system can use the Rebrickable API instead of the local CSV database by setting `SET_SOURCE = "api"` in `sorting_pipeline.py`.


## Raspberry Pi File

`gate_control2.py` is included in the submission as the Raspberry Pi servo control script.

This file is used for controlling the servo gate on the Raspberry Pi. It should be placed on the Raspberry Pi at the script path configured in `sorting_pipeline.py`.

## Selecting the LEGO Set

The LEGO set used for sorting is selected inside `sorting_pipeline.py` using:

```python
SET_NUMBER = "test"
```

In this project, `"test"` refers to a custom set JSON file created for demonstration purposes:

```text
Set_data_read/Set_data/test.json
```

The value can be changed to any other LEGO set number or custom set name, as long as a matching JSON file exists in `Set_data_read/Set_data/`.

Example for an official LEGO set:

```python
SET_NUMBER = "10696-1"
```

This would use:

```text
Set_data_read/Set_data/10696-1.json
```

If the JSON file for the selected set does not exist, the system can generate it using either the local CSV database or the Rebrickable API, depending on the selected `SET_SOURCE` value.

## Selecting Set Data Source

Inside `sorting_pipeline.py`:

```python
SET_SOURCE = "local"
```

Available options:

```text
"local" = use local Rebrickable CSV files
"api"   = use the Rebrickable online API
```

For normal use and submission, `"local"` is recommended because it does not require an internet connection or API key.

## Using the Rebrickable API

If using:

```python
SET_SOURCE = "api"
```

the Rebrickable API key must be stored as an environment variable.

On Windows PowerShell:

```powershell
setx REBRICKABLE_API_KEY "your_api_key_here"
```

After setting the environment variable, close and reopen VS Code or PowerShell.

## Raspberry Pi Setup

The Raspberry Pi is used to control the servo gate through its GPIO pins.

The servo control script, `gate_control2.py`, should be placed on the Raspberry Pi at the designated script path used in `sorting_pipeline.py`.

The main computer sends an SSH command to the Raspberry Pi to run this script with the required servo angle.

## Raspberry Pi Requirements

The Raspberry Pi should have Python 3 installed.

Install the required Python package on the Raspberry Pi:

```bash
sudo apt update
sudo apt install python3-gpiozero
```

If `gpiozero` is not available through `apt`, install it using pip:

```bash
pip3 install gpiozero
```

SSH should also be enabled on the Raspberry Pi so that the main computer can send servo control commands.

## SSH Access Note

The main computer must be able to connect to the Raspberry Pi through SSH.

The Raspberry Pi username, hostname, and servo script path should be configured in `sorting_pipeline.py`.

The Raspberry Pi password is not stored in the source code. Login details are provided separately for assessment purposes.

If password login is used, the password may be requested each time a servo command is sent. For assessment purposes, any required temporary Raspberry Pi login details are provided separately from the source code.

Alternatively, SSH key-based login can be configured to allow the main computer to send servo commands without repeatedly asking for a password. This is recommended for smoother operation during live sorting.

## Servo Pin Setup

The servo signal wire is connected to GPIO18 on the Raspberry Pi.

Servo wiring:

```text
Servo signal wire (yellow) -> GPIO18, physical pin 12
Servo power wire (red)     -> 5V, physical pin 2
Servo ground wire (black)  -> GND, physical pin 6
```

## Servo Control

The file `gate_control2.py` controls the servo gate.

It accepts an angle from the command line:

```bash
python3 gate_control2.py 145
python3 gate_control2.py 170
```

Default servo angles:

```text
145 degrees = accepted path
170 degrees = rejected path
```

The SSH settings in `sorting_pipeline.py` should match the Raspberry Pi username, hostname, and the location of `gate_control2.py` on the Raspberry Pi.

Example settings in `sorting_pipeline.py`:

```python
PI_USER = "ananta"
PI_HOST = "lego-pi.local"
PI_SCRIPT_PATH = "designated/path/to/gate_control2.py"
```

Before running the full sorting system, test the servo directly on the Raspberry Pi using:

```bash
python3 gate_control2.py 145
python3 gate_control2.py 170
```

## Python Requirements

The project uses the following main Python libraries:

```text
torch
torchvision
opencv-python
pillow
pandas
numpy
scikit-learn
requests
```

Example installation command:

```bash
pip install torch torchvision opencv-python pillow pandas numpy scikit-learn requests
```

## Before Running Checklist

Before running the system, check that:

- The camera is connected.
- The Raspberry Pi is powered on.
- The Raspberry Pi is connected to the same network as the main computer.
- SSH access to the Raspberry Pi is working.
- `gate_control2.py` is placed on the Raspberry Pi at the configured script path.
- `checkpoints/best_model.pt` exists.
- `artifacts/class_to_idx.json` exists.
- The selected set JSON file exists in `Set_data_read/Set_data/`.

## How to Run the System

1. Open a terminal in the main project folder:

```bash
cd FYP_code
```

2. Run the main system:

```bash
python main.py
```

3. Press `ESC` to stop the camera window.

## Important Notes

- The trained model file `best_model.pt` is required for classification.
- The `class_to_idx.json` file is required to map model outputs to LEGO part IDs.
- The system should be run from the main project folder.
- The local Rebrickable CSV database is used to generate LEGO set part lists.
- The `Dataset/` folder is only required for training or retraining the model.
- The `captures/` folder stores temporary images captured during detection.
- The sorting accuracy may vary depending on camera placement, lighting, conveyor speed, and brick position.

## Additional Images

Hardware setup images:

```text
images/Hardware_setup_front.jpg
images/Hardware_setup_top.jpg
```

Raspberry Pi servo pin setup image:

```text
images/Pi_servo_pin_setup.jpg
```

## Author

Ananta Taqy Nafis

Final Year Project

Automated LEGO Sorting Machine
