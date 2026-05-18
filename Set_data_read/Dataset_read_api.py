from pathlib import Path
import requests
import json
import os

# Setting
BASE_URL = "https://rebrickable.com/api/v3/lego"

BASE_FOLDER = Path(__file__).resolve().parent
SET_FOLDER = BASE_FOLDER / "Set_data"
SET_FOLDER.mkdir(exist_ok=True)


def save_set_to_json_api(set_number):
    API_KEY = os.getenv("REBRICKABLE_API_KEY")

    if API_KEY is None:
        raise ValueError(
            "REBRICKABLE_API_KEY environment variable is not set. "
            "Please set it before running this script."
        )

    headers = {
        "Authorization": f"key {API_KEY}"
    }

    parts = {}
    url = f"{BASE_URL}/sets/{set_number}/parts/?inc_spares=0"

    print(f"Fetching data for set {set_number} from Rebrickable...")

    # Fetch parts
    while url:
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print("Error fetching parts:", response.status_code)
            print(response.text)
            return

        data = response.json()

        for item in data["results"]:
            part_num = item["part"]["part_num"]
            part_name = item["part"]["name"]
            quantity = item["quantity"]

            if part_num not in parts:
                parts[part_num] = {
                    "part_num": part_num,
                    "part_name": part_name,
                    "quantity": quantity
                }
            else:
                parts[part_num]["quantity"] += quantity

        url = data["next"]

    # Sort result
    result = sorted(parts.values(), key=lambda x: x["part_num"])

    # Get set name
    set_url = f"{BASE_URL}/sets/{set_number}/"
    set_res = requests.get(set_url, headers=headers)

    if set_res.status_code != 200:
        print("Error getting set info:", set_res.status_code)
        print(set_res.text)
        return

    set_name = set_res.json()["name"]

    # Save json
    data = {
        "set_number": set_number,
        "set_name": set_name,
        "parts": result
    }

    output_path = SET_FOLDER / f"{set_number}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Saved to {output_path}")


# Run
if __name__ == "__main__":
    save_set_to_json_api("10696-1")