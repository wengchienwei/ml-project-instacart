import kagglehub

def download_instacart_data():
    """Download Instacart dataset from Kaggle"""
    path = kagglehub.dataset_download("psparks/instacart-market-basket-analysis")
    print(f"Dataset downloaded to: {path}")

if __name__ == "__main__":
    download_instacart_data()