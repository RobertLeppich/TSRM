import os
from setups import *


def run_experiments(dry_run=False, plot_attention=True):

    if not os.path.exists("data/data_dir") or len(os.listdir("data/data_dir")):
        print("Download data first at: https://drive.google.com/drive/folders/1Sw6LClDcYhy5byltrezagiap9a5sGIfH "
              "and save into 'data/data_dir' folder")
        os.makedirs("data/data_dir", exist_ok=True)
        exit(0)

    # Imputation experiments
    imputation_air_quality.run(dry_run=dry_run, plot_attention=plot_attention)
    imputation_electricity.run(dry_run=dry_run, plot_attention=plot_attention)

    # Forecasting experiments
    forecasting_electricity.run(dry_run=dry_run, plot_attention=plot_attention)
    forecasting_traffic.run(dry_run=dry_run, plot_attention=plot_attention)
    forecasting_ettm2.run(dry_run=dry_run, plot_attention=plot_attention)

    # Classification experiments
    classification_wsdm.run(dry_run=dry_run, plot_attention=plot_attention)

if __name__ == '__main__':
    run_experiments()