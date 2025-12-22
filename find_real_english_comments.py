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

# 识别中文的正则表达式
chinese_pattern = re.compile(r'[\u4e00-\u9fff]')  # 基本中文字符
# 识别英文单词（至少2个连续字母）
english_word_pattern = re.compile(r'[A-Za-z]{2,}')

for file in all_files:
    if file not in modified_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # 查找单行注释
                    comment_match = re.search(r'//(.*)', line)
                    if comment_match:
                        comment_text = comment_match.group(1).strip()
                        # 跳过空注释、URL、using语句等
                        if (comment_text and
                            not comment_text.startswith('http') and
                            not 'using' in comment_text.lower() and
                            not 'namespace' in comment_text.lower() and
                            not 'class' in comment_text.lower() and
                            not 'namespace' in comment_text.lower() and
                            not 'System' in comment_text and
                            not '.IO' in comment_text and
                            not 'Models' in comment_text and
                            not comment_text.startswith('<') and
                            not comment_text.startswith('/') and
                            not comment_text.startswith('*')):

                            # 判断是否主要包含英文而不是中文
                            has_chinese = bool(chinese_pattern.search(comment_text))
                            has_english_words = bool(english_word_pattern.search(comment_text))

                            # 如果有英文单词而没有中文字符，认为是英文注释
                            if has_english_words and not has_chinese:
                                # 进一步检查：确保不是纯符号或数字
                                letter_count = sum(c.isalpha() for c in comment_text)
                                if letter_count >= 4:  # 至少有4个字母
                                    files_with_english_comments.append(file)
                                    break

                    # 检查多行注释的起始
                    if '/*' in line and '*/' not in line:
                        # 开始收集多行注释
                        multi_line_comment = [line[line.find('/*') + 2:]]
                        in_multi_line = True
                        lineno = line_num + 1
                        while in_multi_line and lineno < len(lines):
                            if '*/' in lines[lineno]:
                                multi_line_comment.append(lines[lineno][:lines[lineno].find('*/')])
                                in_multi_line = False
                            else:
                                multi_line_comment.append(lines[lineno])
                            lineno += 1

                        comment_text = '\n'.join(multi_line_comment).strip()
                        # 应用同样的英文检测逻辑
                        if comment_text:
                            has_chinese = bool(chinese_pattern.search(comment_text))
                            has_english_words = bool(english_word_pattern.search(comment_text))

                            if has_english_words and not has_chinese:
                                letter_count = sum(c.isalpha() for c in comment_text)
                                if letter_count >= 4:
                                    files_with_english_comments.append(file)
                                    break

        except Exception as e:
            # 静默处理编码错误
            pass

# 输出结果
print("包含英文注释的未修改文件列表（需要翻译）：")
print("=" * 60)
for file in sorted(files_with_english_comments):
    print(file)

print(f"\n总计找到 {len(files_with_english_comments)} 个文件")