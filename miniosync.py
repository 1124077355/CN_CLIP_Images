from minio import Minio
from minio.error import S3Error
import os
import csv
import uuid

# MinIO服务配置
endpoint = ""
access_key = ""
secret_key = ""
bucket_name = ""
local_dir = "E:\images\e"  # 本地保存文件的目录
csv_filename = "000.csv"  # 保存新文件名的CSV文件

# 初始化MinIO客户端
client = Minio(
    endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=False  # 如果你的MinIO服务使用HTTPS，这里设置为True
)

# 确保本地目录存在
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

# CSV文件用于保存新的文件名
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Original Filename", "New Filename"])  # 写入表头

    try:
        # 获取桶中所有对象的列表
        objects = client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            # 获取对象名（即文件名）
            original_filename = obj.object_name
            # 分割原始路径和文件名
            dir_path, old_filename = os.path.split(original_filename)
            if (dir_path != ''):
                new_filename = f"{dir_path}-{os.path.basename(old_filename)}"
            else:
                new_filename = f"{os.path.basename(old_filename)}"
            # 构造本地文件的完整路径
            local_file_path = os.path.join(local_dir, new_filename)
            # 从MinIO下载文件到本地
            client.fget_object(bucket_name, original_filename, local_file_path)
            print(f"Downloaded {original_filename} to {local_file_path}")

            # 将原始文件名和新文件名写入CSV文件
            csvwriter.writerow([original_filename, new_filename])

    except S3Error as err:
        print("error message: ", err)

print(f"New filenames have been saved to {csv_filename}")