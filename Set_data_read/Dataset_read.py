from pathlib import Path
import pandas as pd
import json

# paths
BASE_FOLDER = Path(__file__).resolve().parent

DATA_PATH = BASE_FOLDER / "Database"
SET_FOLDER = BASE_FOLDER / "Set_data"
SET_FOLDER.mkdir(exist_ok=True)

# Load data
sets = pd.read_csv(DATA_PATH / "sets.csv")
inventories = pd.read_csv(DATA_PATH / "inventories.csv")
inventory_parts = pd.read_csv(DATA_PATH / "inventory_parts.csv")
parts = pd.read_csv(DATA_PATH / "parts.csv")


# Functions
def save_set_to_json(set_number):

    # Get set info
    set_info = sets[sets["set_num"] == set_number]
    if set_info.empty:
        print("Set not found")
        return

    set_name = set_info.iloc[0]["name"]

    # Get invetory ID
    inv = inventories[inventories["set_num"] == set_number]
    if inv.empty:
        print("Inventory not found")
        return

    inventory_id = inv.iloc[0]["id"]

    # Get parts
    parts_list = inventory_parts[inventory_parts["inventory_id"] == inventory_id]
    parts_list = parts_list.merge(parts, on="part_num", how="left")

    result = parts_list[["part_num", "name", "quantity"]]
    result.columns = ["part_num", "part_name", "quantity"]

    # Combine same parts
    combined = {}

    for _, row in result.iterrows():
        key = row["part_num"]

        if key not in combined:
            combined[key] = {
                "part_num": row["part_num"],
                "part_name": row["part_name"],
                "quantity": int(row["quantity"])
            }
        else:
            combined[key]["quantity"] += int(row["quantity"])

    result = list(combined.values())

    result = sorted(result, key=lambda x: x["part_num"])

    # Save JSON
    data = {
        "set_number": set_number,
        "set_name": set_name,
        "parts": result
    }

    output_path = SET_FOLDER / f"{set_number}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Saved to {output_path}")


#run
if __name__ == "__main__":
    save_set_to_json("10696-1")