Project Overview
----------------
This project is an automated LEGO sorting machine that uses computer vision, deep learning, set matching, and a servo-based mechanical gate to classify and sort LEGO parts.

The system detects a LEGO brick on a conveyor using OpenCV MOG2 background subtraction, captures multiple images of the brick, classifies the part using a trained ResNet-50 model, checks whether the predicted part belongs to the selected LEGO set, and moves a servo gate to direct the brick to either the accepted or rejected path.

Main System Flow
----------------
Camera input
→ MOG2 background subtraction
→ Object detection and burst image capture
→ ResNet-50 classification
→ Majority voting
→ LEGO set matching
→ Quantity limit checking
→ Servo gate sorting

Main Files
----------
main.py
    Main entry point for running the complete sorting system.

detection.py
    Handles camera input, MOG2 background subtraction, object detection, and burst image capture.

sorting_pipeline.py
    Handles classification, majority voting, LEGO set matching, quantity checking, and servo sorting command.

Classify.py
    Contains model loading, image preprocessing, class mapping, and prediction functions.

Classification_train.py
    Training script used to train and fine-tune the ResNet-50 LEGO part classifier.

Set_data_read/Dataset_read.py
    Generates LEGO set JSON files using the local Rebrickable CSV database.

Set_data_read/Dataset_read_api.py
    Generates LEGO set JSON files using the Rebrickable API.

gate_control2.py
    Servo control script used on the Raspberry Pi.
    This file should be placed on the Raspberry Pi at the script path configured in sorting_pipeline.py.

Folders
-------
checkpoints/
    Contains the trained model checkpoint.
    Required file:
    - best_model.pt

artifacts/
    Contains model class mapping and training/evaluation results.
    Required file:
    - class_to_idx.json

    Optional files:
    - training_history.json
    - final_metrics.json

Set_data_read/
    Contains scripts and data used for LEGO set information.

Set_data_read/Database/
    Contains local Rebrickable CSV database files.

Set_data_read/Set_data/
    Contains generated LEGO set JSON files used by the sorting pipeline.

captures/
    Stores temporary captured brick images during detection.

Dataset/
    Contains the training dataset.
    Expected structure:
    - Dataset/renders/
    - Dataset/photos/

images/
    Contains additional project images, such as hardware setup and Raspberry Pi servo pin setup images.

Expected File Structure
-----------------------
The project folder should follow this structure:

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
    │   ├── sets.csv
    │   ├── inventories.csv
    │   ├── inventory_parts.csv
    │   └── parts.csv
    │
    └── Set_data/
        ├── test.json
        └── 10696-1.json
    
    Note: The Dataset/ folder is not included directly in the software ZIP file due to file size. It can be downloaded using the dataset link below.

Raspberry Pi File
-----------------
gate_control2.py
    This file is included in the submission as the Raspberry Pi servo control script.
    This file is used for controlling the servo gate on the Raspberry Pi.
    It should be placed on the Raspberry Pi at the script path configured in sorting_pipeline.py.

Dataset Download
----------------
The full training dataset is provided separately because it is too large to include directly in the software ZIP file. The dataset is approximately 12.6 GB when zipped and 14.6 GB when extracted.

The dataset can be downloaded separately from the following Google Drive link:

    https://drive.google.com/file/d/1nu0tGKkl-7D9QcM4DcleFhq9gf2CN9f5/view?usp=sharing

After downloading, extract or place the dataset folder in the main project directory with the following structure:

    FYP_code/
    └── Dataset/
        ├── renders/
        └── photos/

The Dataset/ folder is only required for training or retraining the model. It is not required for running the trained sorting system if checkpoints/best_model.pt and artifacts/class_to_idx.json are already provided.

Selecting the LEGO Set
----------------------
The LEGO set used for sorting is selected inside sorting_pipeline.py using:

    SET_NUMBER = "test"

In this project, "test" refers to a custom set JSON file, created for demonstration purposes:

    Set_data_read/Set_data/test.json

The value can be changed to any other LEGO set number or custom set name, as long as a matching JSON file exists in Set_data_read/Set_data/.

Example for an official LEGO set:

    SET_NUMBER = "10696-1"

This would use:

    Set_data_read/Set_data/10696-1.json

If the JSON file for the selected set does not exist, the system can generate it using either the local CSV database or the Rebrickable API, depending on the selected SET_SOURCE value.

Selecting Set Data Source
-------------------------
Inside sorting_pipeline.py:

    SET_SOURCE = "local"

Available options:

    "local" = use local Rebrickable CSV files
    "api"   = use the Rebrickable online API

For normal use and submission, "local" is recommended because it does not require an internet connection or API key.

Using the Rebrickable API
-------------------------
If using SET_SOURCE = "api", the Rebrickable API key must be stored as an environment variable.

On Windows PowerShell:

    setx REBRICKABLE_API_KEY "your_api_key_here"

After setting the environment variable, close and reopen VS Code or PowerShell.

Raspberry Pi Setup
------------------
The Raspberry Pi is used to control the servo gate through its GPIO pins.

The servo control script, gate_control2.py, should be placed on the Raspberry Pi at the designated script path used in sorting_pipeline.py.

The main computer sends an SSH command to the Raspberry Pi to run this script with the required servo angle.

Raspberry Pi Requirements
-------------------------
The Raspberry Pi should have Python 3 installed.

Install the required Python package on the Raspberry Pi:

    sudo apt update
    sudo apt install python3-gpiozero

If gpiozero is not available through apt, install it using pip:

    pip3 install gpiozero

SSH should also be enabled on the Raspberry Pi so that the main computer can send servo control commands.

SSH Access Note
---------------
The main computer must be able to connect to the Raspberry Pi through SSH.

The Raspberry Pi username, hostname, and servo script path should be configured in sorting_pipeline.py.

The Raspberry Pi password is not stored in the source code. Login details are provided separately for assessment purposes.

If password login is used, the password may be requested each time a servo command is sent. For assessment purposes, any required temporary Raspberry Pi login details are provided separately from the source code.

Alternatively, SSH key-based login can be configured to allow the main computer to send servo commands without repeatedly asking for a password. This is recommended for smoother operation during live sorting.

Servo Pin Setup
---------------
The servo signal wire is connected to GPIO18 on the Raspberry Pi.

Servo wiring:

    Servo signal wire (yellow) -> GPIO18, physical pin 12
    Servo power wire (red)     -> 5V, physical pin 2
    Servo ground wire (black)  -> GND, physical pin 6

Servo Control
-------------
The file gate_control2.py controls the servo gate.

It accepts an angle from the command line:

    python3 gate_control2.py 145
    python3 gate_control2.py 170

Default servo angles:

    145 degrees = accepted path
    170 degrees = rejected path


The SSH settings in sorting_pipeline.py should match the Raspberry Pi username, hostname, and the location of gate_control2.py on the Raspberry Pi.

Example settings in sorting_pipeline.py:

    PI_USER = "ananta"
    PI_HOST = "lego-pi.local"
    PI_SCRIPT_PATH = "designated/path/to/gate_control2.py"

Before running the full sorting system, test the servo directly on the Raspberry Pi using:

    python3 gate_control2.py 145
    python3 gate_control2.py 170

Python Requirements
-------------------
The project uses the following main Python libraries:

    torch
    torchvision
    opencv-python   (imported as cv2)
    pillow          (imported as PIL)
    pandas
    numpy
    scikit-learn
    requests

Example installation command:

    pip install torch torchvision opencv-python pillow pandas numpy scikit-learn requests

Before Running Checklist
------------------------
Before running the system, check that:

    - The camera is connected.
    - The Raspberry Pi is powered on.
    - The Raspberry Pi is connected to the same network as the main computer.
    - SSH access to the Raspberry Pi is working.
    - gate_control2.py is placed on the Raspberry Pi at the configured script path.
    - checkpoints/best_model.pt exists.
    - artifacts/class_to_idx.json exists.
    - The selected set JSON file exists in Set_data_read/Set_data/.

How to Run the System
---------------------
1. Open a terminal in the main project folder:

   cd FYP_code

2. Run the main system:

   python main.py

3. Press ESC to stop the camera window.

Important Notes
---------------
- The trained model file best_model.pt is required for classification.
- The class_to_idx.json file is required to map model outputs to LEGO part IDs.
- The system should be run from the main project folder.
- The local Rebrickable CSV database is used to generate LEGO set part lists.
- The Dataset/ folder is only required for training or retraining the model.
- The captures/ folder stores temporary images captured during detection.
- The sorting accuracy may vary depending on camera placement, lighting, conveyor speed, and brick position.

Additional Images
-----------------
Hardware setup images:
    images/Hardware_setup_front.jpg
    images/Hardware_setup_top.jpg

Raspberry Pi servo pin setup image:
    images/Pi_servo_pin_setup.jpg
