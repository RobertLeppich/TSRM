from experiments.experiment_runner import ExperimentRun
from data.datasets import electricity, air_quality, ett, wsdm, traffic


imputation_electricity = ExperimentRun(
    dataset=electricity.ElectricityImputationSpecificSplitUnivariate,
    task="finetune_imputation",
    config_str="imputation_electricity")

imputation_air_quality = ExperimentRun(
    dataset=air_quality.AirQualityImputationSpecificSplit,
    task="finetune_imputation",
    config_str="imputation_air_quality")

forecasting_ettm2 = ExperimentRun(
    dataset=ett.ETTm2Dataset,
    task="finetune_forecasting",
    config_str="forecasting_ettm2")

forecasting_electricity = ExperimentRun(
    dataset=electricity.ElectricityForecastingSpecificSplitUnivariat,
    task="finetune_forecasting",
    config_str="forecasting_electricity")

forecasting_traffic = ExperimentRun(
    dataset=traffic.TrafficDataset,
    task="finetune_forecasting",
    config_str="forecasting_traffic")

classification_wsdm = ExperimentRun(
    dataset=wsdm.WSDMDataset,
    task="finetune_classification",
    config_str="classification_wsdm")
