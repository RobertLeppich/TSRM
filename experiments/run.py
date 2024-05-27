import os
from setups import *

def run_experiments():

    if not os.path.exists("data/data_dir") or len(os.listdir("data/data_dir")):
        print("Download data first at: https://drive.google.com/drive/folders/1Sw6LClDcYhy5byltrezagiap9a5sGIfH "
              "and save into 'data/data_dir' folder")
        os.makedirs("data/data_dir", exist_ok=True)
        exit(0)

    # Imputation experiments
    imputation_air_quality.run()
    imputation_electricity.run()

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