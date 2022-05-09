import urllib.request
from bs4 import BeautifulSoup
import os

def shein_women(endpage=11):
    #접근할 페이지 번호
    pageNum = 1

    #저장할 이미지 경로 및 이름 (data폴더에 face0.jpg 형식으로 저장)
    imageNum = 0
    imageStr = "crawled_images/shein_women_crawled/best"
    path = 'crawled_images/shein_women_crawled/'
    if not os.path.exists(path):
        os.makedirs(path)

    while pageNum < endpage:
        url = "https://asia.shein.com/Clothing-c-2035.html?ici=asiako_tab01navbar05&scici=navbar_WomenHomePage~~tab01navbar05~~5~~webLink~~~~0&srctype=category&userpath=category%3E%EC%9D%98%EB%A5%98&sort=8&page="
        url = url + str(pageNum)
        
        fp = urllib.request.urlopen(url)
        source = fp.read();
        fp.close()

        soup = BeautifulSoup(source, 'html.parser')
        soup = soup.findAll("div",class_ = "S-product-item__wrapper")
        #이미지 경로를 받아 로컬에 저장
        for i in soup:
            imageNum += 1
            imgURL = i.find("img")["data-src"]
            imgURL = "https:" + imgURL
            urllib.request.urlretrieve(imgURL,imageStr + str(imageNum) + ".jpg")
            print("Current ImageNum = ", imageNum)
        pageNum += 1
