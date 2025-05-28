import os
import shutil

def extract_images_by_prefix(source_dir, output_dir):
    if not os.path.exists(source_dir):
        print(f"源目录不存在：{source_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seen_prefixes = set()

    for filename in os.listdir(source_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue  # 跳过非图片文件

        prefix = filename[:3]  # 取前三个字符作为类别标识

        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"提取：{filename}")

    print(f"\n提取完成，总共提取 {len(seen_prefixes)} 类图片到 {output_dir}")

# 示例使用
source_folder = 'D:\\yolov5\\RealTime-TSR\\data\\train\images\\train'   # 替换为你的源文件夹路径
output_folder = 'output'               # 输出文件夹路径
extract_images_by_prefix(source_folder, output_folder)
