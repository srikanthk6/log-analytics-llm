import random
import uuid
from datetime import datetime
import argparse
import os

APPLICATIONS = [
    "order-service", "payment-service", "inventory-service", "shipping-service"
]
ORDER_STATUSES = [
    "PLACED", "PAID", "SHIPPED", "CANCELLED", "ON_HOLD"
]
HOLD_REASONS = [
    "Payment Declined", "Inventory Shortage", "Address Verification Failed"
]
PRODUCTS = [
    ("Wireless Mouse", 29.99),
    ("Mechanical Keyboard", 189.99),
    ("USB-C Cable", 9.99),
    ("Bluetooth Speaker", 49.99),
    ("Laptop Stand", 39.99)
]
SHIPPING_PROVIDERS = ["FedEx", "UPS", "DHL", "USPS"]

def random_order_lines():
    lines = []
    for _ in range(random.randint(1, 3)):
        product, price = random.choice(PRODUCTS)
        qty = random.randint(1, 3)
        lines.append(f"{{product={product},qty={qty},price={price:.2f}}}")
    return "[" + ",".join(lines) + "]"

def generate_log(anomaly=False):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    app = random.choice(APPLICATIONS)
    trace_id = str(uuid.uuid4())
    order_number = f"ORD{random.randint(100000, 999999)}"
    customer_id = f"CUST{random.randint(1000, 9999)}"
    status = random.choice(ORDER_STATUSES)
    if anomaly:
        status = "ON_HOLD"
    amount = round(random.uniform(20, 500), 2)
    lines = random_order_lines()
    log_level = random.choice(["INFO", "ERROR", "WARNING"])
    msg = ""
    extra = ""
    if app == "order-service":
        msg = f"Order {order_number} placed by customer {customer_id}. Status: {status}. Total: ${amount}"
        if status == "ON_HOLD":
            extra = f' holdReason="{random.choice(HOLD_REASONS)}"'
    elif app == "payment-service":
        if status == "ON_HOLD":
            log_level = "ERROR"
            extra = f' holdReason="Payment Declined"'
            msg = f"Payment failed for order {order_number}. Reason: Payment Declined. Order placed on hold."
        else:
            msg = f"Payment processed for order {order_number}. Amount: ${amount}"
    elif app == "inventory-service":
        msg = f"Inventory reserved for order {order_number}: {lines}."
    elif app == "shipping-service":
        if status == "SHIPPED":
            provider = random.choice(SHIPPING_PROVIDERS)
            tracking = f"{provider.upper()}{random.randint(1000000,9999999)}"
            msg = f"Order {order_number} shipped via {provider}. Tracking number: {tracking}."
            extra = f" shippingProvider={provider} trackingNumber={tracking}"
        else:
            msg = f"Shipping event for order {order_number}. Status: {status}."
    return (
        f"{now} {log_level} {app} traceId={trace_id} orderNumber={order_number} "
        f"customerId={customer_id} status={status} amount={amount} lines={lines}{extra} msg=\"{msg}\""
    )

def main():
    parser = argparse.ArgumentParser(description="Generate e-commerce log4j2-style logs.")
    parser.add_argument('--count', type=int, default=20, help='Number of logs to generate')
    parser.add_argument('--anomaly-rate', type=float, default=0.0, help='Fraction of logs to be anomalies (0.0-1.0)')
    parser.add_argument('--output', type=str, default=None, help='Output file path (default: logs/ecommerce.log)')
    args = parser.parse_args()
    count = args.count
    anomaly_rate = args.anomaly_rate
    num_anomalies = int(count * anomaly_rate)
    anomaly_indices = set(random.sample(range(count), num_anomalies)) if num_anomalies > 0 else set()
    output_path = args.output or 'log_analytics/logs/ecommerce.log'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for i in range(count):
            log_line = generate_log(anomaly=(i in anomaly_indices))
            f.write(log_line + '\n')
            # print(log_line)

if __name__ == "__main__":
    main()
