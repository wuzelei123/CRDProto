# import json

# # 输入和输出文件路径
# txt_path = "map.txt"       # 替换成你的txt文件路径
# json_path = "map.json"        # 保存json的路径
#
# data_dict = {}
#
# with open(txt_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue  # 跳过空行
#         if ' ' not in line:
#             continue  # 跳过不包含空格的行
#         key, value = line.split(' ', 1)  # 只分割第一个空格
#         data_dict[key] = value.strip()
#
# # 保存为json文件
# with open(json_path, 'w', encoding='utf-8') as f:
#     json.dump(data_dict, f, ensure_ascii=False, indent=2)
#
# print(f"已保存为JSON文件：{json_path}")
import json

# 输入输出路径
input_json_path = "map.json"    # 替换成你的输入json文件路径
output_json_path = "map.json"  # 输出文件路径

# 读取JSON文件
with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理值，只保留第一个逗号前的部分
new_data = {k: v.split(',')[0].strip() for k, v in data.items()}

# 写入新的JSON文件
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"处理完成，已保存至：{output_json_path}")