1. `python -m venv venv` 가상환경 생성 및 실행
2. `pip install -r requirements.txt` 실행
3. https://drive.google.com/file/d/1P2BtqrIKbz2Dtp3qfPCkvp16bj9xSVIw/view?usp=sharing 여기서 `yolov3-df2_15000.weights` 파일을 받아 `pretrained_yolo` 폴더에 넣어주세요.
3. `crawling.py` 실행 => `crawled_images` 폴더에 `wconcept_men`, `wconcept_women`, `shein_men`, `shein_women` 폴더안에 각각의 사이트들에서 크롤링한 이미지들이 생깁니다. (총 5000장 정도 되는 양으로 인터넷 속도에 따라 다르나 어느정도 시간이 소요됩니다.)
4. `make_feature_excel.py` 혹은 `make_feature_excel.ipynb` 실행 (ipynb 파일이 가독성이 더 좋아 추천합니다.)=> `men.xlsx` 와 `women.xlsx` 생깁니다 (컴퓨터 사양에 따라 다르나 어느정도 시간이 소요됩니다. 현재 폴더에 존재함으로 생략가능합니다.)
5. `men.xlsx` 와 `women.xlsx`를 tableau로 작업 => 결과: `Men_Clothes_Trend.twb` 과 `Women_Clothes_Trend.twb`