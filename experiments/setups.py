from experiment import Experiment

tsrm_fc_elc = Experiment(exp_id="TSRM-FC-Electricity", codename="run1",
                         mode="downstream",
                         config_str="forecasting_electricity", task="forecasting",
                         hyperparameters={"h": [8],
                                          "N": [3, 4],
                                          "weight_decay": [0.0001],
                                          "encoding_size": [16, 32],
                                          "embed": ["timeF"],
                                          "dropout": [0.25],
                                          "attention_func": ["entmax15"],
                                          "conv_dims": [
                                              # ((kernel_size, dilation, groups), (...))
                                              [[3, 1, -1], [5, 2, -1], [10, 3, -1]],

                                          ],
                                          "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                          "seq_len": [96],
                                          "pred_len": [96, 192, 336, 720]})

tsrm_fc_ettm2 = Experiment(exp_id="TSRM-FC-ETTm2", codename="run1",
                           mode="downstream",
                           config_str="forecasting_ettm2", task="forecasting",
                           hyperparameters={"h": [32],
                                            "N": [8],
                                            "weight_decay": [0.001],
                                            "encoding_size": [64],
                                            "embed": ["timeF"],
                                            "dropout": [0.25],
                                            "attention_func": ["entmax15"],
                                            "conv_dims": [
                                                # ((kernel_size, dilation, groups), (...))
                                                [[3, 1, 1], [10, 2, 1], [15, 3, 1]],
                                            ],
                                            "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                            "seq_len": [96],
                                            "pred_len": [96, 192, 336, 720],
                                            })

tsrm_fc_traffic = Experiment(exp_id="TSRM-FC-Traffic", codename="run1",
                             mode="downstream",
                             config_str="forecasting_traffic_ds", task="forecasting",
                             hyperparameters={"h": [8],
                                              "N": [4],
                                              "encoding_size": [16],
                                              "attention_func": ["classic"],
                                              "conv_dims": [
                                                  # ((kernel_size, dilation, groups), (...))
                                                  [[3, 1, -1], [5, 2, -1], [10, 3, -1]],
                                              ],
                                              "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                              "seq_len": [96],
                                              "pred_len": [96, 192, 336, 720],
                                              })

tsrm_fc_ettm1 = Experiment(exp_id="TSRM-FC-ETTm1", codename="run1",
                           mode="downstream",
                           config_str="forecasting_ettm1", task="forecasting",
                           hyperparameters={"h": [32, 64],
                                            "N": [2, 4, 8, 10, 12],
                                            "weight_decay": [0.001],
                                            "encoding_size": [32, 64, 128],
                                            "embed": ["timeF"],
                                            "dropout": [0.25],
                                            "attention_func": ["entmax15"],
                                            "conv_dims": [
                                                # ((kernel_size, dilation, groups), (...))
                                                [[3, 1, 1], [10, 2, 1], [15, 3, 1]],
                                                [[3, 1, -1], [5, 2, -1], [10, 3, -1]],
                                                [[3, 1, 1], [5, 2, 1], [10, 3, 1]],
                                            ],
                                            "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                            "seq_len": [96],
                                            "pred_len": [96, 192, 336, 720],
                                            })

tsrm_fc_etth1 = Experiment(exp_id="TSRM-FC-ETTh1", codename="run1",
                           mode="downstream",
                           config_str="forecasting_etth1", task="forecasting",
                           hyperparameters={"h": [16],
                                            "N": [7],
                                            "weight_decay": [0.001],
                                            "encoding_size": [64],
                                            "embed": ["timeF"],
                                            "dropout": [0.25],
                                            "attention_func": ["entmax15"],
                                            "conv_dims": [
                                                # ((kernel_size, dilation, groups), (...))
                                                [[3, 1, -1], [5, 2, -1], [10, 3, -1]],
                                            ],
                                            "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                            "seq_len": [96],
                                            "pred_len": [96, 192, 336, 720],
                                            })

tsrm_fc_etth2 = Experiment(exp_id="TSRM-FC-ETTh2", codename="run1",
                           mode="downstream",
                           config_str="forecasting_etth2", task="forecasting",
                           hyperparameters={"h": [16],
                                            "N": [3],
                                            "weight_decay": [0.001],
                                            "encoding_size": [64],
                                            "embed": ["timeF"],
                                            "dropout": [0.25],
                                            "attention_func": ["entmax15"],
                                            "conv_dims": [
                                                # ((kernel_size, dilation, groups), (...))
                                                [[3, 1, -1], [5, 2, -1], [10, 3, -1]],
                                            ],
                                            "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                            "seq_len": [96],
                                            "pred_len": [96, 192, 336, 720], })

tsrm_fc_exchange = Experiment(exp_id="TSRM-FC-Exchange", codename="run1",
                              mode="downstream",
                              config_str="forecasting_exchange", task="forecasting",
                              hyperparameters={"h": [16, 32],
                                               "N": [8, 10, 12, 14],
                                               "weight_decay": [0.0001],
                                               "encoding_size": [128, 256],
                                               "embed": ["fixed"],
                                               "dropout": [0.25],
                                               "attention_func": ["entmax15"],
                                               "conv_dims": [
                                                   # ((kernel_size, dilation, groups), (...))
                                                   [[3, 1, 1], [10, 2, 1], [15, 3, 1]],
                                                   [[3, 1, 1], [5, 2, 1]],
                                               ],
                                               "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                               "seq_len": [96],
                                               "pred_len": [96, 192, 336, 720], })

tsrm_fc_weather = Experiment(exp_id="TSRM-FC-Weather", codename="run1",
                             mode="downstream",
                             config_str="forecasting_weather", task="forecasting",
                             hyperparameters={"h": [32],
                                              "N": [4],
                                              "weight_decay": [0.0001],
                                              "encoding_size": [128],
                                              "embed": ["timeF"],
                                              "dropout": [0.25],
                                              "attention_func": ["entmax15"],
                                              "conv_dims": [
                                                  # ((kernel_size, dilation, groups), (...))
                                                  [[3, 1, 1], [5, 2, 1], [10, 3, 1]]
                                              ],
                                              "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                              "seq_len": [96],
                                              "pred_len": [96, 192, 336, 720],
                                              })

tsrm_imp_elc = Experiment(exp_id="TSRM-Imputation-Electricity", codename="run1",
                          mode="downstream",
                          config_str="imputation_electricity", task="imputation",
                          hyperparameters={"h": [4],
                                           "N": [3, 4],
                                           "weight_decay": [0.001],
                                           "encoding_size": [8],
                                           "embed": ["timeF"],
                                           "dropout": [0.25],
                                           "attention_func": ["entmax15"],
                                           "conv_dims": [
                                               # ((kernel_size, dilation, groups), (...))
                                               [[3, 1, 1], [5, 2, 1], [10, 3, 1]],
                                           ],
                                           "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                           "missing_ratio": [0.125, 0.25, 0.375, 0.5]
                                           })

tsrm_imp_ettm1 = Experiment(exp_id="TSRM-Imputation-ETTm1", codename="run1",
                            mode="downstream",
                            config_str="imputation_ettm1", task="imputation",
                            hyperparameters={"h": [16, 32],
                                             "N": [2, 4, 8],
                                             "weight_decay": [0.001],
                                             "encoding_size": [32, 64],
                                             "embed": ["timeF"],
                                             "dropout": [0.25],
                                             "attention_func": ["entmax15"],
                                             "conv_dims": [
                                                 # ((kernel_size, dilation, groups), (...))
                                                 [[3, 1, 1], [5, 2, 1], [10, 3, 1]],
                                                 [[3, 1, 1], [10, 2, 1], [15, 3, 1]],
                                             ],
                                             "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                             "missing_ratio": [0.125, 0.25, 0.375, 0.5]
                                             })

tsrm_imp_ettm2 = Experiment(exp_id="TSRM-Imputation-ETTm2", codename="run1",
                            mode="downstream",
                            config_str="imputation_ettm2", task="imputation",
                            hyperparameters={"h": [16, 32],
                                             "N": [2, 4, 8],
                                             "weight_decay": [0.001],
                                             "encoding_size": [32, 64],
                                             "embed": ["timeF"],
                                             "dropout": [0.25],
                                             "attention_func": ["entmax15"],
                                             "conv_dims": [
                                                 # ((kernel_size, dilation, groups), (...))
                                                 [[3, 1, 1], [5, 2, 1], [10, 3, 1]],
                                                 [[3, 1, 1], [10, 2, 1], [15, 3, 1]],
                                             ],
                                             "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                             "missing_ratio": [0.125, 0.25, 0.375, 0.5]
                                             })

tsrm_imp_etth1 = Experiment(exp_id="TSRM-Imputation-ETTh1", codename="run1",
                            mode="downstream",
                            config_str="imputation_etth1", task="imputation",
                            hyperparameters={"h": [16, 32],
                                             "N": [3, 4, 8],
                                             "weight_decay": [0.001],
                                             "encoding_size": [32, 64],
                                             "embed": ["timeF"],
                                             "dropout": [0.25],
                                             "attention_func": ["entmax15"],
                                             "conv_dims": [
                                                 # ((kernel_size, dilation, groups), (...))
                                                 [[3, 1, 1], [5, 2, 1], [10, 3, 1]],
                                                 [[3, 1, 1], [10, 2, 1], [15, 3, 1]],
                                             ],
                                             "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                             "missing_ratio": [0.125, 0.25, 0.375, 0.5]
                                             })

tsrm_imp_etth2 = Experiment(exp_id="TSRM-Imputation-ETTh2", codename="run1",
                            mode="downstream",
                            config_str="imputation_etth2", task="imputation",
                            hyperparameters={"h": [16, 32],
                                             "N": [3, 4, 8],
                                             "weight_decay": [0.001],
                                             "encoding_size": [32, 64],
                                             "embed": ["timeF"],
                                             "dropout": [0.25],
                                             "attention_func": ["entmax15"],
                                             "conv_dims": [
                                                 # ((kernel_size, dilation, groups), (...))
                                                 [[3, 1, 1], [5, 2, 1], [10, 3, 1]],
                                                 [[3, 1, 1], [10, 2, 1], [15, 3, 1]],
                                             ],
                                             "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                             "missing_ratio": [0.125, 0.25, 0.375, 0.5]
                                             })

tsrm_imp_weather = Experiment(exp_id="TSRM-Imputation-Weather", codename="run1",
                              mode="downstream",
                              config_str="imputation_weather", task="imputation",
                              hyperparameters={"h": [16, 32],
                                               "N": [3, 4, 8],
                                               "weight_decay": [0.001],
                                               "encoding_size": [32, 64],
                                               "embed": ["timeF"],
                                               "dropout": [0.25],
                                               "attention_func": ["entmax15"],
                                               "conv_dims": [
                                                   # ((kernel_size, dilation, groups), (...))
                                                   [[3, 1, 1], [5, 2, 1], [10, 3, 1]],
                                                   [[3, 1, 1], [10, 2, 1], [15, 3, 1]],
                                               ],
                                               "feature_ff": [True, False],  # True=TSRM_IFC, False=TSRM
                                               "missing_ratio": [0.125, 0.25, 0.375, 0.5]
                                               })
