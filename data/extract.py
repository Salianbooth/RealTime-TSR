import os
import shutil

# 固定路径配置（请根据实际情况修改）
SOURCE_DIR = r"./tsrd_dataset/images/val"
TARGET_DIR = r"./tsrd_dataset/labels/train/"
OUTPUT_DIR = r"./tsrd_dataset/labels/test/"


def extract_matching_files_ignore_suffix(source_dir, target_dir, output_dir):
    if not os.path.isdir(source_dir):
        print(f"源目录不存在: {source_dir}")
        return
    if not os.path.isdir(target_dir):
        print(f"目标目录不存在: {target_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 获取源目录中所有文件主名（不含扩展名）
    source_basenames = set(
        os.path.splitext(f)[0] for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)))

    matched_files = []

    for filename in os.listdir(target_dir):
        full_path = os.path.join(target_dir, filename)
        if not os.path.isfile(full_path):
            continue
        basename, ext = os.path.splitext(filename)
        if basename in source_basenames:
            dst_path = os.path.join(output_dir, filename)
            shutil.copy2(full_path, dst_path)
            matched_files.append(filename)

    print(f"共匹配 {len(matched_files)} 个文件并复制到输出目录。")
    if matched_files:
        print("匹配的文件包括：")
        for f in matched_files:
            print(f"  - {f}")
    else:
        print("未找到任何同名文件（忽略扩展名匹配）。")


if __name__ == "__main__":
    extract_matching_files_ignore_suffix(SOURCE_DIR, TARGET_DIR, OUTPUT_DIR)
