import urllib.request
from bs4 import BeautifulSoup
import os

def shein_men(endpage=11):
    #접근할 페이지 번호
    pageNum = 1

    #저장할 이미지 경로 및 이름 (data폴더에 face0.jpg 형식으로 저장)
    imageNum = 0
    imageStr = "crawled_images/shein_men_crawled/best"
    path = 'crawled_images/shein_men_crawled/'
    if not os.path.exists(path):
        os.makedirs(path)

    while pageNum < endpage:
        url = "https://asia.shein.com/Men-Clothing-c-1969.html?ici=asiako_tab04navbar03&scici=navbar_MenHomePage~~tab04navbar03~~3~~real_1969~~~~0&srctype=category&userpath=category%3E%EB%82%A8%EC%84%B1%20%EC%9D%98%EB%A5%98&sort=8&page="
        url = url + str(pageNum)
        
        fp = urllib.request.urlopen(url)
        source = fp.read();
        fp.close()

        soup = BeautifulSoup(source, 'html.parser')
        soup = soup.findAll("div",class_ = "S-product-item__wrapper")
        # print(soup)
        #이미지 경로를 받아 로컬에 저장한다.
        for i in soup:
            imageNum += 1
            imgURL = i.find("img")["data-src"]
            imgURL = "https:" + imgURL
            print(imgURL)
            urllib.request.urlretrieve(imgURL,imageStr + str(imageNum) + ".jpg")
            print("Current ImageNum = ", imageNum)

        pageNum += 1
