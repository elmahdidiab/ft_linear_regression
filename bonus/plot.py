import csv
import json
import os
import sys
import matplotlib.pyplot as plt

DATA_FILE = '../mandatory/data.csv'
THETA_FILE = "../mandatory/theta.json"

def load_data(filepath):
    mileages = []
    prices = []
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
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

def main():
    mileages, prices = load_data(DATA_FILE)
    theta0, theta1 = load_thetas()

    x_min = min(mileages)
    x_max = max(mileages)
    line_x = [x_min, x_max]
    line_y = [theta0 + theta1 * x for x in line_x]

    plt.figure(figsize=(10, 6))
    plt.scatter(mileages, prices, color="blue", label="Data points", zorder=5)
    plt.plot(line_x, line_y, color="red", linewidth=2, label="Regression line")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car Price vs Mileage - Linear Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.show()
    print("Plot saved as plot.png")

if __name__ == "__main__":
    main()