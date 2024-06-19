import csv

# 텍스트 파일 읽기
with open('raw_data/sub_2000000.txt', 'r') as f:
    lines = f.readlines()

# CSV 파일 쓰기
with open('raw_data/sub_2000000.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    # 각 줄을 탭 문자로 분할하여 CSV 파일에 쓰기
    for line in lines:
        # 탭 문자를 기준으로 분할하여 데이터 추출
        data = line.strip().split('\t')

        # CSV 파일에 데이터 쓰기
        writer.writerow(data)
