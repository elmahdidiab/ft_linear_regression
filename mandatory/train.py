import csv
import json
import sys

DATA_FILE = "data.csv"
THETA_FILE = "theta.json"
LEARNING_RATE = 0.1
ITERATIONS = 1000

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

def normalize(data):
    if not data:
        print("Error: Cannot normalize empty list.")
        sys.exit(1)
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val:
        print("Error: All values are identical, cannot normalize (division by zero).")
        sys.exit(1)
    normalized = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized, min_val, max_val

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def train(mileages, prices, learning_rate, iterations):
    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages)

    for _ in range(iterations):
        tmp0 = learning_rate * (1 / m) * sum(
            estimate_price(mileages[i], theta0, theta1) - prices[i]
            for i in range(m)
        )
        tmp1 = learning_rate * (1 / m) * sum(
            (estimate_price(mileages[i], theta0, theta1) - prices[i]) * mileages[i]
            for i in range(m)
        )
        theta0 -= tmp0
        theta1 -= tmp1

    return theta0, theta1

def denormalize_thetas(theta0, theta1, km_min, km_max, price_min, price_max):
    price_range = price_max - price_min
    km_range = km_max - km_min

    real_theta1 = theta1 * price_range / km_range
    real_theta0 = theta0 * price_range + price_min - real_theta1 * km_min
    return real_theta0, real_theta1

def save_thetas(theta0, theta1):
    with open(THETA_FILE, "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)

def main():
    mileages, prices = load_data(DATA_FILE)

    norm_mileages, km_min, km_max = normalize(mileages)
    norm_prices, price_min, price_max = normalize(prices)

    theta0, theta1 = train(norm_mileages, norm_prices, LEARNING_RATE, ITERATIONS)

    real_theta0, real_theta1 = denormalize_thetas(
        theta0, theta1, km_min, km_max, price_min, price_max
    )

    save_thetas(real_theta0, real_theta1)
    print(f"Training complete!")
    print(f"theta0 = {real_theta0:.6f}")
    print(f"theta1 = {real_theta1:.6f}")

if __name__ == "__main__":
    main()