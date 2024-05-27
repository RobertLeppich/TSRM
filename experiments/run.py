import os
from setups import *

def run_experiments(dry_run=False):

    if not os.path.exists("data/data_dir") or len(os.listdir("data/data_dir")):
        print("Download data first at: https://drive.google.com/drive/folders/1Sw6LClDcYhy5byltrezagiap9a5sGIfH "
              "and save into 'data/data_dir' folder")
        os.makedirs("data/data_dir", exist_ok=True)
        exit(0)

    # Imputation experiments
    imputation_air_quality.run(dry_run=dry_run)
    imputation_electricity.run(dry_run=dry_run)

    # forecasting experiments
    forecasting_electricity.run(dry_run=dry_run)
    forecasting_traffic.run(dry_run=dry_run)
    forecasting_ettm2.run(dry_run=dry_run)

    # classification experiments
    classification_wsdm.run(dry_run=dry_run)

if __name__ == '__main__':

    run_experiments()