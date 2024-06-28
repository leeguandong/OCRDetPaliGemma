import os
import zipfile


def zip_files_in_folder(folder_path, max_files_in_zip=10):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 计算需要创建的zip文件的数量
    num_zips = len(files) // max_files_in_zip
    if len(files) % max_files_in_zip != 0:
        num_zips += 1

    for i in range(num_zips):
        # 创建一个新的zip文件
        with zipfile.ZipFile(f'{folder_path}/archive_{i + 1}.zip', 'w') as zipf:
            # 获取需要添加到这个zip文件中的文件
            files_to_zip = files[i * max_files_in_zip:(i + 1) * max_files_in_zip]
            for file in files_to_zip:
                # 将文件添加到zip文件中
                zipf.write(os.path.join(folder_path, file), arcname=file)


# 使用方法
folder_path = '你的文件夹路径'
zip_files_in_folder(folder_path, max_files_in_zip=4)
