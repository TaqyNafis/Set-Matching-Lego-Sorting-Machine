from pathlib import Path
from collections import Counter, defaultdict
import torch
import os
import json

from Classify import load_model, load_mapping, get_preprocess, predict_image
from Set_data_read.Dataset_read import save_set_to_json
from Set_data_read.Dataset_read_api import save_set_to_json_api

# Settings
CKPT_PATH = Path("checkpoints/best_model.pt")
CLASS_MAP_PATH = Path("artifacts/class_to_idx.json")
IMG_SIZE = 224
TOPK = 1

# Servo angle settings
ACCEPT_ANGLE = 145
REJECT_ANGLE = 170

# Servo SSH settings
PI_USER = "ananta"
PI_HOST = "lego-pi.local"
PI_SCRIPT_PATH = "/home/ananta/gate_control2.py"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SET_NUMBER = "test"

# Choose where to get the LEGO set data from:
# "local" = use local Rebrickable CSV files
# "api"   = download from Rebrickable API
SET_SOURCE = "local"

# Json Handling
SET_FOLDER = Path("Set_data_read") / "Set_data"
SET_FOLDER.mkdir(exist_ok=True)

json_path = SET_FOLDER / f"{SET_NUMBER}.json"

# If JSON does not exist, create it using the selected source
if not json_path.exists():
    print(f"{json_path} not found. Creating JSON...")

    if SET_SOURCE == "local":
        print("Using local Rebrickable CSV database...")
        save_set_to_json(SET_NUMBER)

    elif SET_SOURCE == "api":
        print("Using Rebrickable API...")
        save_set_to_json_api(SET_NUMBER)

    else:
        raise ValueError("Invalid SET_SOURCE. Use 'local' or 'api'.")

else:
    print(f"Using existing JSON file: {json_path}")

# load Json
with open(json_path, "r", encoding="utf-8") as f:
    set_data = json.load(f)

# Build lookup tables
PART_LIMITS = {
    part["part_num"]: int(part["quantity"])
    for part in set_data["parts"]
}

VALID_PARTS = set(PART_LIMITS.keys())

PART_COUNTS = defaultdict(int)

# Keep track of last servo angle sent
CURRENT_SERVO_ANGLE = None

print(f"Loaded {len(VALID_PARTS)} valid parts from {json_path}")

# Load model
_, idx_to_class = load_mapping(CLASS_MAP_PATH)

model = load_model(
    ckpt_path=CKPT_PATH,
    num_classes=len(idx_to_class),
    device=device,
)

preprocess = get_preprocess(IMG_SIZE)


# Servo helpers
def move_servo(angle):
    global CURRENT_SERVO_ANGLE

    if CURRENT_SERVO_ANGLE == angle:
        print(f"Servo already at {angle}, skipping command")
        return

    print(f"Moving servo from {CURRENT_SERVO_ANGLE} to {angle}")

    command = f'ssh {PI_USER}@{PI_HOST} "python3 {PI_SCRIPT_PATH} {angle}"'
    os.system(command)

    CURRENT_SERVO_ANGLE = angle


# Initial servo position
move_servo(REJECT_ANGLE)


# Classify and vote
def classify_and_vote(image_paths):
    predictions = []

    for img_path in image_paths:
        results = predict_image(
            image_path=Path(img_path),
            model=model,
            preprocess=preprocess,
            idx_to_class=idx_to_class,
            device=device,
            topk=TOPK,
        )

        predicted_class = results[0][0]
        predictions.append(predicted_class)
        print(f"{img_path} -> {predicted_class}")

    counter = Counter(predictions)
    final_class, vote_count = counter.most_common(1)[0]

    print("\nVote results:")
    for cls, cnt in counter.items():
        print(f"{cls}: {cnt}")

    print(f"\nFinal prediction: {final_class}")

   #Check againts set and quantity
    if final_class in VALID_PARTS:
        allowed_qty = PART_LIMITS[final_class]
        current_qty = PART_COUNTS[final_class]

        print(f"{final_class}: accepted {current_qty}/{allowed_qty}")

        #if parts is in set
        if current_qty < allowed_qty:
            PART_COUNTS[final_class] += 1
            print(f"{final_class} (in set, accepted)")
            print(f"New count: {PART_COUNTS[final_class]}/{allowed_qty}")
            move_servo(ACCEPT_ANGLE)
        else: #if part is in set but quantity limit reached
            print(f"{final_class} (in set but quantity limit reached, treated as not valid)")
            move_servo(REJECT_ANGLE)
    else:#if part is not in set
        print(f"{final_class} (not in set)")
        move_servo(REJECT_ANGLE)

    return final_class