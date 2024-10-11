from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(config, flag):
    Data = data_dict[config["data"]]
    timeenc = 0 if config["embed"] != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = config["batch_size"]
        freq = config["freq"]
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = config["freq"]
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = config["batch_size"]
        freq = config["freq"]

    data_set = Data(
        root_path=config["root_path"],
        data_path=config["data_path"],
        flag=flag,
        size=[config["seq_len"], config["label_len"], config["pred_len"]],
        features=config["features"],
        target=config["target"],
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        pin_memory=True,
        num_workers=config["num_workers"],
        drop_last=drop_last)
    return data_set, data_loader
