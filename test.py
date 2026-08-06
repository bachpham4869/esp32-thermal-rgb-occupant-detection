import pandas as pd

# Đọc dataset gốc
df = pd.read_csv("data_should_light.csv")

# Quy tắc: dưới 30 độ thì luôn bật đèn sưởi
df.loc[df["temperature"] < 30, "should_light"] = 1
df.loc[df["temperature"] > 30, "should_light"] = 0

# Lưu ra file mới
df.to_csv("dataset_heater.csv", index=False)
