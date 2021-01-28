import librosa, librosa.display
import json, os, math


DATASET_PATH = "kpop_high_list"
JSON_PATH = "kpop_genres_data/K_POP_Data_mfcc.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # sec
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION  # 한 track의 sample 수


def save_mfcc(dataset_path, json_path, n_mfcc, n_fft, hop_length, n_segments):
    # wav 파일에서 MFCC를 추출하고, 장르 라벨과 함께 json 파일로 저장하기 위한 함수
    # dataset_path (str): Path to dataset
    # json_path (str): Path to json file used to save MFCCs
    # n_mfcc (int): Number of coefficients to extract
    # n_fft (int): 한 frame 당 sample 수
    # hop_length (int): 겹치는 frame의 sample 수
    # n_segments (int): 트랙자체를 split 시켜 데이터 Augmentation 효과를 주자

    # mapping, labels, and MFCCs 를 저장할 data라는 딕셔너리 생성
    data = {"mapping": [], "labels": [], "mfcc": []}

    samples_per_segment = int(SAMPLES_PER_TRACK / n_segments)  # track 전체 sample을 segment 수로 나눠서, segment 당 sample 수 정의
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)  # 굿노트 필기

    # 모든 sub 폴더에 대해 loop
    for i, (dirpath, dirnames, filenames) in enumerate(
            os.walk(dataset_path)):  # eumerate 안에 argument를 풀어 매 iter 마다 인덱스와 결과값을 출력
        # dirpath : 현재 경로, dirnames : 현재 경로상에 디렉토리 목록, filenames : 현재 경로상에 파일 목록
        if dirpath is not dataset_path:

            genre_label = dirpath.split("/")[-1]
            data["mapping"].append(genre_label)
            print("\n Processing : {}".format(genre_label))

            # 각 장르 폴더 밑에, 음원 파일 precessing
            for f in filenames:

                file_path = os.path.join(dirpath, f)  # 현재 dirpath와 file 이름을 경로명으로 이어준다
                sig, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Augmentation을 위해 segment 하므로, 각 segment에 대한 processing
                for d in range(n_segments):

                    # 각 segment의 시작, 끝 점을 지정
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(sig[start:finish], sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # input data 차원을 맞춰줘야 되기 때문에, 위에서 지정한 mfcc 차원에 맞으면 저장한다
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())  # mfcc는 넘파이 어레이 이므로, list로 바꿔서 append 해준다
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))  # 음원 파일 경로, 각 음원의 d 번째 segment

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)   # [발라드, 댄스, 포크, 랩/힙합, 인디음악, 트로트]


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, 13, 2048, 512, 5)