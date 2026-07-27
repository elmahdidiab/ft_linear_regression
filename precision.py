import csv
import json
import os
import sys

DATA_FILE = "../mandatory/data.csv"
THETA_FILE = "../mandatory/theta.json"

def load_data(filepath):
    mileages = []
    prices = []
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # row 1 is header
                try:
                    km = float(row.get("km", ""))
                    price = float(row.get("price", ""))
                    mileages.append(km)
                    prices.append(price)
                except (ValueError, TypeError):
                    print(f"Warning: Skipping invalid row {row_num} (km='{row.get('km')}', price='{row.get('price')}')")
    except FileNotFoundError:
        print(f"Error: Data file '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    if not mileages or not prices:
        print(f"Error: No valid data found in '{filepath}'. File may be empty or missing 'km'/'price' columns.")
        sys.exit(1)

    return mileages, prices

def load_thetas():
    if not os.path.exists(THETA_FILE):
        print("Warning: theta.json not found. Using default thetas (0.0, 0.0).")
        return 0.0, 0.0
    try:
        with open(THETA_FILE, "r") as f:
            data = json.load(f)
            theta0 = data.get("theta0", 0.0)
            theta1 = data.get("theta1", 0.0)
            return theta0, theta1
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Warning: Could not read theta.json: {e}. Using default thetas (0.0, 0.0).")
        return 0.0, 0.0

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def main():
    mileages, prices = load_data(DATA_FILE)
    theta0, theta1 = load_thetas()
    m = len(mileages)

    if m == 0:
        print("Error: No data points available to compute precision.")
        sys.exit(1)

    predictions = [estimate_price(km, theta0, theta1) for km in mileages]

    mean_price = sum(prices) / m
    ss_tot = sum((prices[i] - mean_price) ** 2 for i in range(m))
    ss_res = sum((prices[i] - predictions[i]) ** 2 for i in range(m))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print(f"R²   (Coefficient of determination): {r2:.4f}  ({r2 * 100:.2f}%)")

if __name__ == "__main__":
    main()