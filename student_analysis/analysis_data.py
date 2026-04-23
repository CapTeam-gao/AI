# import json

# input_path = "/Users/kshi3430/CapTeam/data/student.jsonl"
# output_path = "/Users/kshi3430/CapTeam/data/student_analysis_data/students.json"

# data = []

# with open(input_path, "r", encoding="utf-8") as f:
#     for line in f:
#         if line.strip():  # 빈 줄 방지
#             data.append(json.loads(line))

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print("변환 완료")

# print(data[0])

import json
import os

def convert_jsonl_to_json():
    input_path = "/Users/kshi3430/CapTeam/data/student.jsonl"
    output_path = "/Users/kshi3430/CapTeam/data/student_analysis_data/students2.json"

    # 이미 파일 있으면 실행 안 함
    if os.path.exists(output_path):
        print("이미 변환된 파일이 존재합니다. 실행하지 않습니다.")
        return

    datas = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                datas.append(json.loads(line))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)

    print("변환 완료")
    print(datas[0])


if __name__ == "__main__":
    convert_jsonl_to_json()