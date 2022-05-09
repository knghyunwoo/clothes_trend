import urllib.request
from bs4 import BeautifulSoup
import os

from wconcept_women_crawling import wconcept_women
from wconcept_men_crawling import wconcept_men
from shein_women_crawling import shein_women
from shein_men_crawling import shein_men


wconcept_women()
wconcept_men()
shein_women()
shein_men()