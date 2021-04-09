from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
import math
import time

def loadArticles(driver, times):
    WebDriverWait(driver, 10).until( #wait till you can click the load more button
        EC.presence_of_element_located((By.XPATH, "//button[@class='options__load-more']"))
    )
    load_more = driver.find_element_by_xpath("//button[@class='options__load-more']")
    for i in range(times):
        time.sleep(0.5)
        load_more.click()

def getArticleList(driver):
    WebDriverWait(driver, 5).until( #wait till you can find the div containing all articles
        EC.presence_of_element_located((By.ID, "infinitescroll"))
    )
    articleListDiv = driver.find_element_by_id("infinitescroll")
    allArticles = articleListDiv.find_elements_by_class_name("item-info") #inside the div get individual article clickable section
    return allArticles
def parseArticle(driver):
    try:
        divP = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "storytext"))
        )
    except:
        print("couldn't find any contents")
    #divP = driver.find_element_by_id("storytext")
    P = driver.find_elements_by_tag_name('p') #find all paragraph tags
    finalStringArticle = ""
    for i in range(len(P)):
        try:
            if(len(P[i].text) > 50): #if it's not authors etc.
                finalStringArticle = finalStringArticle + P[i].text #extract all strings and append
        except:
            print("something happened on paragraph: " + str(i))


    print("Paragraphs: ",finalStringArticle)
    return finalStringArticle


def main():
    PATH = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(PATH)
    driver.get("https://www.npr.org/sections/politics/")
    DRIVERCONST = driver.title # constant to check what page the driver is currently on
    time.sleep(0.5)
    loadArticles(driver, 2) #Number of times you load +24 articles
    allArticles = getArticleList(driver) #List of articles in WebContent Format. (List of articles you can click)
    finalArticleArray = [] #List or strings containg all of each articles contents

    for i in range(len(allArticles)):
        print("article: ", i)
        #time.sleep(0.5)
        WebDriverWait(driver, 5).until( #Wait until you find the div containing all articles
            EC.presence_of_element_located((By.ID, "infinitescroll"))
        )
        allArticles[i].click()


        time.sleep(0.5) #wait to swtich page
        if (DRIVERCONST == driver.title): #If you coudn't click on the page
            allArticles = getArticleList(driver)
            print("couldn't click article ", i)
        else:
            finalArticleArray.append(parseArticle(driver)) #add contents to article array
            driver.back() #Go back to main page
            loadArticles(driver, math.ceil((i+1)/22)+1) # reload the articles (only to how far we need to go)
            allArticles = getArticleList(driver)  #Re get List of articles in WebContent Format. (List of articles you can click) otherwise driver forgets about it

    lb = open("Articles.txt", "w+")
    for i in finalArticleArray:
        lb.write(i + "\n")
if __name__ == "__main__":
    main()






