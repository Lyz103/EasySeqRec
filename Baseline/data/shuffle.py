import random

random.seed()

def shuffle_lines(file1, file2, output_file1, output_file2):
    # 读取两个文件的所有行
    with open(file1, 'r') as f1:
        lines1 = f1.readlines()
        
    with open(file2, 'r') as f2:
        lines2 = f2.readlines()
    
    # 确保两个文件的行数相同
    if len(lines1) != len(lines2):
        raise ValueError("两个文件的行数不相同")
    
    # 生成索引列表并打乱
    indices = list(range(len(lines1)))
    random.shuffle(indices)
    
    # 根据打乱后的索引重新排列行
    shuffled_lines1 = [lines1[i] for i in indices]
    shuffled_lines2 = [lines2[i] for i in indices]
    
    # 写入打乱后的内容到新文件
    with open(output_file1, 'w') as f1:
        f1.writelines(shuffled_lines1)
        
    with open(output_file2, 'w') as f2:
        f2.writelines(shuffled_lines2)



def replace_first_column_with_line_numbers(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 替换每行的第一个位置为行号
    modified_lines = []
    for i, line in enumerate(lines):
        # 分割行内容
        parts = line.split(maxsplit=1)
        # 使用行号替换第一个位置
        line_number = str(i + 1)  # 行号从1开始
        if len(parts) > 1:
            new_line = f"{line_number} {parts[1]}"
        else:
            new_line = f"{line_number}\n"  # 处理没有其他内容的行

        modified_lines.append(new_line)

    # 写入新文件
    with open(output_file, 'w') as f:
        f.writelines(modified_lines)

# 示例用法
file1 = 'Beauty.txt'
file2 = 'Beauty_sample.txt'
output_file1 = 'Beauty_ran.txt'
output_file2 = 'Beauty_ran_sample.txt'

file1s = ['Beauty.txt', 'Video_Games.txt', 'Toys_and_Games.txt', 'Sports_and_Outdoors.txt', 'Health_and_Personal_Care.txt']
file2s = ['Beauty_sample.txt', 'Video_Games_sample.txt', 'Toys_and_Games_sample.txt', 'Sports_and_Outdoors_sample.txt', 'Health_and_Personal_Care_sample.txt']
output_file1s = ['Beauty_ran.txt', 'Video_Games_ran.txt', 'Toys_and_Games_ran.txt', 'Sports_and_Outdoors_ran.txt', 'Health_and_Personal_Care_ran.txt', 'Beauty_ran1.txt', 'Video_Games_ran1.txt', 'Toys_and_Games_ran1.txt', 'Sports_and_Outdoors_ran1.txt', 'Health_and_Personal_Care_ran1.txt', 'Beauty_ran2.txt', 'Video_Games_ran2.txt', 'Toys_and_Games_ran2.txt', 'Sports_and_Outdoors_ran2.txt', 'Health_and_Personal_Care_ran2.txt']
output_file2s = ['Beauty_ran_sample.txt', 'Video_Games_ran_sample.txt', 'Toys_and_Games_ran_sample.txt', 'Sports_and_Outdoors_ran_sample.txt', 'Health_and_Personal_Care_ran_sample.txt','Beauty_ran1_sample.txt', 'Video_Games_ran1_sample.txt', 'Toys_and_Games_ran1_sample.txt', 'Sports_and_Outdoors_ran1_sample.txt', 'Health_and_Personal_Care_ran1_sample.txt','Beauty_ran2_sample.txt', 'Video_Games_ran2_sample.txt', 'Toys_and_Games_ran2_sample.txt', 'Sports_and_Outdoors_ran2_sample.txt', 'Health_and_Personal_Care_ran2_sample.txt']

for i in range(5):
    shuffle_lines(file1s[i], file2s[i], output_file1s[i], output_file2s[i])
    replace_first_column_with_line_numbers(output_file1s[i],output_file1s[i])
    replace_first_column_with_line_numbers(output_file2s[i],output_file2s[i])
for i in range(5):
    shuffle_lines(file1s[i], file2s[i], output_file1s[i + 5], output_file2s[i+ 5])
    replace_first_column_with_line_numbers(output_file1s[i+ 5],output_file1s[i+ 5])
    replace_first_column_with_line_numbers(output_file2s[i+ 5],output_file2s[i+ 5])
for i in range(5):
    shuffle_lines(file1s[i], file2s[i], output_file1s[i+ 10], output_file2s[i+ 10])
    replace_first_column_with_line_numbers(output_file1s[i+ 10],output_file1s[i+ 10])
    replace_first_column_with_line_numbers(output_file2s[i+ 10],output_file2s[i+ 10])


