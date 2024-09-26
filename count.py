from collections import Counter
import os

###########################################################
# 입력 폴더와 출력 파일 경로 설정
input_folder = 'center-frame_LLava_result/k100_spatio_descriptors'
output_file = 'concept_set/k100/k100_spatio_concepts.txt'
###########################################################
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 모든 .txt 파일에서 상위 10개의 빈도수가 높은 단어를 추출
with open(output_file, 'w', encoding='utf-8') as out_f:
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 단어 카운트
            words = []
            for line in lines:
                segments = [segment.split(', ') for segment in line.strip().split(' and ')]
                words.extend([word.lower() for sublist in segments for word in sublist])
                # words.extend([word.lower() for word in line.strip().split(', ')])  # 소문자로 변환
            word_counts = Counter(words)

            most_common_words = word_counts.most_common(10)
            # print(most_common_words)

            unique_words = set() # 중복제거용
            for word, _ in most_common_words:
                if word not in unique_words:
                    unique_words.add(word)
            for word in unique_words:
                out_f.write(f"{word}\n")

print("모든 파일에서 상위 10개의 단어를 추출하여 기록했습니다.")

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
# spatio_file = 'concept_set/k400_spatio_concepts.txt'
# temporal_file = 'concept_set/k400_temporal_concepts.txt'

# with open(spatio_file, 'r', encoding='utf-8') as f:
#     spatio_lines = set(f.readlines())

# with open(temporal_file, 'r', encoding='utf-8') as f:
#     temporal_lines = f.readlines()

# filtered_temporal_lines = [line for line in temporal_lines if line not in spatio_lines]
# filtered_spatial_lines = [line for line in spatio_lines if line not in temporal_lines]

# with open(temporal_file, 'w', encoding='utf-8') as f:
#     f.writelines(filtered_temporal_lines)
# with open(spatio_file, 'w', encoding='utf-8') as f:
#     f.writelines(filtered_spatial_lines)

# print(f"{temporal_file}에서 {spatio_file}와 동일한 줄이 삭제되었습니다.")

# label free 참고