0. 작업 폴더를 생성하고,  코드, 전처리데이터 압출 파일을 각각 작업 폴더 안에 다운로드 받는다. %%

unzip codes.zip
rm codes.zip

mkdir data
unzip preprocessed_data.zip -d data
rm data/preprocessed_data.zip


1. 데이타 셋업
학습 및 검증 데이터셋과, 전처리해둔 데이터를 셋팅하기 위하여 작업 폴더에서 다음과 같은 리눅스 명령을 실행한다.

mkdir data
mkdir saved_models

wget -P data http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove
rm data/glove.6B.zip

wget -P data http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip data/v2_Questions_Train_mscoco.zip -d data
rm data/v2_Questions_Train_mscoco.zip

wget -P data http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip data/v2_Questions_Val_mscoco.zip -d data
rm data/v2_Questions_Val_mscoco.zip

wget -P data http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip data/v2_Questions_Test_mscoco.zip -d data
rm data/v2_Questions_Test_mscoco.zip

wget -P data http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip data/v2_Annotations_Train_mscoco.zip -d data
rm data/v2_Annotations_Train_mscoco.zip

wget -P data http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip data/v2_Annotations_Val_mscoco.zip -d data
rm data/v2_Annotations_Val_mscoco.zip

wget -P data https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
unzip data/trainval_36.zip -d data
rm data/trainval_36.zip



2. 환경 설정
Anaconda 사용시 다음 requirements 및 버전을 설치한 가상환경을 만들어서 실험한다.

python==2.7
pytorch==0.3.1
torchvision==0.2.0
cudatoolkit>=8.0 (CUDA 8.0 이상)
cudann>=7.0.5

h5py>=2.9.0
hdf5>=1.10.4
imageio>=2.5.0
matplotlib>=2.2.3
numpy>=1.15.4
pandas>=0.24.2
pillow>=5.4.1
scikit-image>=0.14.1
scipy=1.2.1
tqdm>=4.35.0



3. 제안 모델 학습 및 검증

다음을 실행하면, 설명 문장 레이블이 있는 VQA v2 데이터셋에서 제안모델이 학습되고 검증되며, 최고 검증 성능의 모델이 저장된다:

python main.py --epochs 30 --batch_size 512 --dim_hid 1024 --emb_rnn 'LSTM' --output 'saved_models/VQAE'

--dim_hid : 설명 문장 특징 추출기 은닉벡터 차원 (512 / 1024)
--emb_rnn : 설명 문장 특징 추출기 모델 ('GRU' / 'LSTM')
각각 두가지중 하나를 골라 실험 5.3 재현 가능.
제안 모델 기본값은 (1024, 'LSTM') 이다.

기타 옵션.

--output 'folder_name' 을 추가하면, 학습된 모델 결과와 학습 기록이 지정된 폴더 'folder_name'에 저장된다.
--resume True일 경우 --output 에 저장되어 있는 모델 파라미터부터 학습을 시작한다. 기본값은 False (scratch부터 학습)
--evaluate True일 경우 --output에 저장되어 있는 모델을 불러와 VQA 성능을 잰다.



4. 설명 문장 성능 측정

1. python results.py --output 'folder_name'
('floder_name'에 저장된 학습된 모델을 불러와, 검증 데이터셋에서 설명 문장 생성 결과를 저장한다.)

2. python coco-caption/cocoEvalCap.py --resDir 'folder_name'
'folder_name' 폴더 안에 있는 시각적 질의응답 설명문장 생성 모델의 성능 출력.
(BLEU-1,2,3,4, CIDEr, METEOR, ROUGE-L)

