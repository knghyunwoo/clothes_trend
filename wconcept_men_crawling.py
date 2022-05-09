import urllib.request
from bs4 import BeautifulSoup
import os

def wconcept_men(endpage=14):
    #접근할 페이지 번호
    pageNum = 1

    #저장할 이미지 경로 및 이름 (data폴더에 face0.jpg 형식으로 저장)
    imageNum = 0
    imageStr = "crawled_images/wconcept_men_crawled/best"
    path = 'crawled_images/wconcept_men_crawled/'
    if not os.path.exists(path):
        os.makedirs(path)



    while pageNum < endpage:
        url = "https://www.wconcept.co.kr/Men/0010?sort=1&page="
        url = url + str(pageNum)
        fp = urllib.request.urlopen(url)
        source = fp.read();
        fp.close()
        soup = BeautifulSoup(source, 'html.parser')
        soup = soup.findAll("div",class_ = "img")
        print(soup)
        #이미지 경로를 받아 로컬에 저장한다.
        for i in soup:
            imageNum += 1
            imgURL = i.find("img")["src"]
            imgURL = "https:" + imgURL
            urllib.request.urlretrieve(imgURL,imageStr + str(imageNum) + ".jpg")
            print(imgURL)
            print("Current ImageNum = ", imageNum)

        pageNum += 1
