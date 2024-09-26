from collections import Counter
import os

file_path = "center-frame_LLava_result/spatio_descriptors/kinetics400_ost_spatio_concepts_archery.txt"

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
print(most_common_words)