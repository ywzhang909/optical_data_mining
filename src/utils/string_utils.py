import re

def parse_string(input_str):
    # 匹配时间部分，格式为 YYYYMMDD HH：MM：SS
    time_pattern = re.compile(r'(\d{8} \d{2}：\d{2}：\d{2})')
    # 匹配括号内的名称部分
    name_pattern = re.compile(r'\((.*?)\)')

    time_match = time_pattern.search(input_str)
    name_match = name_pattern.search(input_str)

    time_part = time_match.group(1) if time_match else None
    name_part = name_match.group(1) if name_match else None

    return time_part, name_part