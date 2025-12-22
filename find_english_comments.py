#!/usr/bin/env python3
import os
import re
import subprocess

# 获取所有未修改的cs文件
result = subprocess.run(['git', 'ls-files', '*.cs'], capture_output=True, text=True)
all_files = result.stdout.strip().split('\n')

# 过滤掉bin和obj目录
all_files = [f for f in all_files if '/bin/' not in f and '/obj/' not in f]

# 获取修改过的文件列表
result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
modified_lines = result.stdout.strip().split('\n')
modified_files = set()
for line in modified_lines:
    if line.startswith(' M '):
        file = line[3:]
        modified_files.add(file)
    elif line.startswith('M '):
        file = line[2:]
        modified_files.add(file)
    elif line.startswith('AM '):
        file = line[3:]
        modified_files.add(file)
    elif line.startswith('A '):
        file = line[2:]
        modified_files.add(file)

# 查找未修改的包含英文注释的文件
files_with_english_comments = []
english_comment_pattern = re.compile(r'//.*[A-Za-z]{4,}')

for file in all_files:
    if file not in modified_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 查找英文注释
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # 检查单行注释
                    if '//' in line:
                        # 移除URL和特殊标记
                        comment_part = line.split('//')[1].strip()
                        # 简单判断是否包含英文单词（至少4个连续字母）
                        if re.search(r'[A-Za-z]{4,}', comment_part):
                            files_with_english_comments.append(file)
                            break
        except Exception as e:
            print(f"Error reading file {file}: {e}")

# 输出结果
print("包含英文注释的未修改文件列表：")
print("=" * 50)
for file in sorted(files_with_english_comments):
    print(file)

print(f"\n总计找到 {len(files_with_english_comments)} 个文件")