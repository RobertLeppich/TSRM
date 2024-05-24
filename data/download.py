import gdown
import os


def download_preprocessed_data(dest_path="data_dir"):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    gdown.download("https://drive.google.com/file/d/1te1WoQ4Wwh5IlLjrREKRebJRW_m_u-W6/view?usp=sharing", os.path.join(dest_path, "electricity.pk"))
    gdown.download("https://drive.google.com/file/d/1qm1USkrKVzQLDo-ad6N0-a1M2LK9Zcqy/view?usp=sharing", os.path.join(dest_path, "air_quality.pk"))
    gdown.download("https://drive.google.com/file/d/11-YvojIE3PBqQiCSucMpyzrKXCMMrvZB/view?usp=sharing", os.path.join(dest_path, "traffic.pk"))
    gdown.download("https://drive.google.com/file/d/1w4YdUgQGVDdnrOV-2LGsqisNSAivM8Mm/view?usp=sharing", os.path.join(dest_path, "ettm2.pk"))
    gdown.download("https://drive.google.com/file/d/1pBOn83GpC1BB4OkgLHNc5FxfT-lSe4M3/view?usp=sharing", os.path.join(dest_path, "wsdm.pk"))

if __name__ == '__main__':
    download_preprocessed_data()