import pandas as pd

# 读取parquet文件
df = pd.read_parquet("/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/OCRPaliGemma/tools/text_ecommerce_detection.parquet")

# 展示前10行
print(df.head(10))