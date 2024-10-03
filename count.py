from collections import Counter
import os
import pandas as pd

###########################################################
# 입력 폴더와 출력 파일 경로 설정
input_folder = 'center-frame_LLava_result/k100_spatio_3frame'
output_file = 'concept_set/k400_center_frame/k400_spatial_concepts_center_frame.txt'
classes_csv = "/data/psh68380/repos/Video-CBM_/data/kinetics400_classes.csv"
threshold = 20
###########################################################
classes_df = pd.read_csv(classes_csv)
class_names = classes_df['name'].tolist()

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# # 모든 .txt 파일에서 상위 10개의 빈도수가 높은 단어를 추출
# with open(output_file, 'w', encoding='utf-8') as out_f:
#     for file_name in os.listdir(input_folder):
#         if file_name.endswith('.txt'):
#             file_path = os.path.join(input_folder, file_name)
#     ##############
#     # for class_name in class_names:
#     #     file_name = f'kinetics400_ost_temporal_concepts_{class_name}.txt'
#     #     file_path = os.path.join(input_folder, file_name)
#     #     if os.path.exists(file_path):
#     ###############
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()

#             # 단어 카운트
#             words = []
#             for line in lines:
#                 segments = [segment.split(', ') for segment in line.strip().split(' and ')]
#                 words.extend([word.lower() for sublist in segments for word in sublist])
#             word_counts = Counter(words)
            
#             # 빈도수 threshold 넘는 단어만 필터링
#             most_common_words = [word for word, count in word_counts.items() if count >= threshold]
#             # most_common_words = word_counts.most_common(10) # class당 top N개 일괄

#             unique_words = set() # 중복제거용
#             for word in most_common_words:
#             # for word, _ in most_common_words:
#                 if word not in unique_words:
#                     unique_words.add(word)
#             for word in unique_words:
#                 out_f.write(f"{word}\n")

# print("모든 파일에서 상위 10개의 단어를 추출하여 기록했습니다.")

# 중복삭제
with open(output_file, 'r', encoding='utf-8') as f:
    concepts = [line.strip() for line in f.readlines()]

unique_lines = list(set(concepts))
unique_lines.sort()
with open(output_file, 'w', encoding='utf-8') as out_f:
    for word in unique_lines:
        out_f.write(f"{word}\n")
###########################################################


# spatio-temporal 파일간 중복삭제
# spatio_file = 'concept_set/k100/k100_spatio_concepts_ver2.txt'
# temporal_file = 'concept_set/k100/k100_temporal_concepts_ver2.txt'

# with open(spatio_file, 'r', encoding='utf-8') as f:
#     spatio_lines = set(f.readlines())

# with open(temporal_file, 'r', encoding='utf-8') as f:
#     temporal_lines = f.readlines()

# # filtered_temporal_lines = [line for line in temporal_lines if line not in spatio_lines]
# # filtered_spatial_lines = [line for line in spatio_lines if line not in temporal_lines]
# filtered_temporal_lines = [
#     line for line in temporal_lines 
#     if line.strip().lower() not in spatio_lines and line.strip().lower() not in class_names
# ]

# with open(temporal_file, 'w', encoding='utf-8') as f:
#     f.writelines(filtered_temporal_lines)
# # with open(spatio_file, 'w', encoding='utf-8') as f:
# #     f.writelines(filtered_spatial_lines)

# print(f"{temporal_file}에서 {spatio_file}와 동일한 줄이 삭제되었습니다.")

# label free 참고