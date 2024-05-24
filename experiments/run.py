import os
from setups import *
from data.download import download_preprocessed_data

def run_experiments():

    if not os.path.exists("data_dir") or len(os.listdir("data_dir")):
        # download data first
        download_preprocessed_data()

    # Imputation experiments
    imputation_electricity.run()
    imputation_air_quality.run()

    # forecasting experiments
    forecasting_electricity.run()
    forecasting_traffic.run()
    forecasting_ettm2.run()

    # classification experiments
    classification_wsdm.run()

if __name__ == '__main__':
    GPU = "MIG-a1208c4e-caad-5519-9d69-6b0998c74b9f" #40gb
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

    run_experiments()