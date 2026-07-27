import json
import os
import csv

THETA_FILE = "theta.json"
DATA_FILE = "data.csv"

def load_thetas():
    if os.path.exists(THETA_FILE):
        with open(THETA_FILE, "r") as f:
            data = json.load(f)
            return data["theta0"], data["theta1"]
    return 0.0, 0.0

def get_training_mileage_range(csv_file=DATA_FILE):
    mileages = []
    if not os.path.exists(csv_file):
        return None, None
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mileages.append(float(row["km"]))
            except (ValueError, KeyError):
                continue
    if not mileages:
        return None, None
    return min(mileages), max(mileages)

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def main():
    theta0, theta1 = load_thetas()
    km_min, km_max = get_training_mileage_range()

    try:
        mileage = float(input("Enter mileage (km): "))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user (Ctrl+C). Exiting.")
        return
    except EOFError:
        print("\nInput stream ended (Ctrl+D). Exiting.")
        return
    except ValueError:
        print("Error: please enter a valid number.")
        return

    if mileage < 0:
        print("Error: mileage cannot be negative.")
        return

    if km_min is not None and km_max is not None:
        if mileage < km_min:
            print(f"Warning: mileage {mileage} is below training range [{km_min}, {km_max}]. Prediction may be unreliable.")
        elif mileage > km_max:
            print(f"Warning: mileage {mileage} exceeds training range [{km_min}, {km_max}]. Prediction may be unreliable.")

    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price: {price:.2f}")

if __name__ == "__main__":
    main()