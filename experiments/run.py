import os
from setups import *


def run_experiments(dry_run=False, plot_attention=True):

    if not os.path.exists("data_dir") or len(os.listdir("data_dir")):
        print("Download data first at: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy "
              "and save into 'data_dir' folder")
        os.makedirs("data_dir", exist_ok=True)
        exit(0)

    # Forecasting experiments
    tsrm_fc_elc.run_downstream(dry_run=dry_run)
    tsrm_fc_ettm2.run_downstream(dry_run=dry_run)
    tsrm_fc_traffic.run_downstream(dry_run=dry_run)
    tsrm_fc_ettm1.run_downstream(dry_run=dry_run)
    tsrm_fc_etth1.run_downstream(dry_run=dry_run)
    tsrm_fc_etth2.run_downstream(dry_run=dry_run)
    tsrm_fc_exchange.run_downstream(dry_run=dry_run)
    tsrm_fc_weather.run_downstream(dry_run=dry_run)

    # Imputation experiments
    tsrm_imp_elc.run_downstream(dry_run=dry_run)
    tsrm_imp_ettm1.run_downstream(dry_run=dry_run)
    tsrm_imp_ettm2.run_downstream(dry_run=dry_run)
    tsrm_imp_etth1.run_downstream(dry_run=dry_run)
    tsrm_imp_etth2.run_downstream(dry_run=dry_run)
    tsrm_imp_weather.run_downstream(dry_run=dry_run)


if __name__ == '__main__':
    run_experiments()