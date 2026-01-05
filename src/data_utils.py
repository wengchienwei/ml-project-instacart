import pandas as pd

# Get cached path dynamically
import kagglehub
DATA_PATH = kagglehub.dataset_download("psparks/instacart-market-basket-analysis")

def load_orders():
    return pd.read_csv(f"{DATA_PATH}/orders.csv")

def load_products():
    return pd.read_csv(f"{DATA_PATH}/products.csv")

def load_order_products_prior():
    return pd.read_csv(f"{DATA_PATH}/order_products__prior.csv")

def load_order_products_train():
    return pd.read_csv(f"{DATA_PATH}/order_products__train.csv")

def load_departments():
    return pd.read_csv(f"{DATA_PATH}/departments.csv")

def load_aisles():
    return pd.read_csv(f"{DATA_PATH}/aisles.csv")