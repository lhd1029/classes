from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import ElementNotInteractableException
import pandas as pd
import time


delay_time = 0.5 #休眠时间
chrome_options = Options()
driver = webdriver.Chrome(chrome_options=chrome_options)
phone_type = []
phone_price = []
phone_comment = []
url_phone = []   # 每页所有手机的链接

def click_lowest_settings(): # 选择最小配置
    temp = 1
    count = 0
    while temp == 1 and count < 5:
        count = count + 1
        time.sleep(delay_time)
        try:
            temp = 0
            driver.find_element_by_xpath("/html/body/div[6]/div/div[2]/*/div[7]/div[1]/div[2]/div[1]/a").click()
            time.sleep(2 * delay_time)
            driver.find_element_by_xpath("/html/body/div[6]/div/div[2]/*/div[7]/div[2]/div[2]/div[1]/a").click()
        except ElementClickInterceptedException:
            temp = 1
            driver.refresh()
        except NoSuchElementException:  # 没有此选项，直接pass
            temp = 0
            pass


def click_comment():  # 点击商品评论
    temp = 1
    count = 0
    while temp == 1 and count < 5:
        count = count + 1
        time.sleep(delay_time)
        click_lowest_settings()
        time.sleep(delay_time)
        try:
            temp = 0
            driver.find_element_by_xpath("/html/body/*/div[2]/div[1]/div[1]/ul/li[5]").click()
        except NoSuchElementException or ElementClickInterceptedException:
            temp = 1
            driver.refresh()
        except ElementNotInteractableException:
            pass


def click_look_current_comment():  # 点击只看当前商品
    temp = 1
    count = 0
    while temp == 1 and count < 5:
        count = count + 1
        time.sleep(delay_time)
        click_comment()
        time.sleep(delay_time)
        try:
            temp = 0
            driver.find_element_by_xpath("/html/body/*/div[2]/div[4]/div[2]/div[2]/div[1]/ul/li[9]/label").click()
        except NoSuchElementException or ElementClickInterceptedException:
            temp = 1
            driver.refresh()
    time.sleep(delay_time)
    if count < 5:
        try:
            comment = driver.find_element_by_xpath("/html/body/*/div[2]/div[4]/div[2]/div[2]/div[1]/ul/li[1]/a/em").text
            comment = comment.replace("(", "")
            comment = comment.replace("+", "")
            comment = comment.replace(")", "")
        except NoSuchElementException:
            comment = ""
        phone_comment.append(comment)
    else:
        comment = ""
    return comment


def click_info():  #点击商品信息
    temp = 1
    comment_phone = ""
    no_phone = ""
    memory_phone = ""
    storage_phone = ""
    while temp == 1:
        time.sleep(delay_time)
        comment_phone = click_look_current_comment()
        time.sleep(delay_time)
        try:
            temp = 0
            driver.find_element_by_xpath("/html/body/*/div[2]/div[1]/div[1]/ul/li[1]").click()
        except NoSuchElementException or ElementClickInterceptedException:
            temp = 1
            driver.refresh()

    time.sleep(delay_time)
    try:
        type_phone = driver.find_element_by_xpath(
            "/html/body/*/div[2]/div[1]/div[2]/div[1]/div[1]/ul[3]/li[1]").text  # 商品名称
        if "商品名称" in type_phone:
            type_phone = type_phone.replace("商品名称：", "")
            pass
        else:
            type_phone = ""


        memory_phone = driver.find_element_by_xpath(
            "/html/body/*/div[2]/div[1]/div[2]/div[1]/div[1]/ul[3]/li[6]").text  #商品内存
        if "运行内存" in memory_phone:
            memory_phone = memory_phone.replace("运行内存：", "")
            pass
        else:
            memory_phone = ""

        no_phone = " " + driver.find_element_by_xpath(
            "/html/body/*/div[2]/div[1]/div[2]/div[1]/div[1]/ul[3]/li[2]").text + " "  # 手机编号
        if "手机编号" in no_phone:
            no_phone = no_phone.replace("手机编号：", "")
            pass
        else:
            no_phone = ""


        storage_phone = driver.find_element_by_xpath(
            "/html/body/*/div[2]/div[1]/div[2]/div[1]/div[1]/ul[3]/li[7]").text  # 机身存储
        if "机身存储" in storage_phone:
            storage_phone = storage_phone.replace("机身存储：", "")
            pass
        else:
            storage_phone = ""

    except NoSuchElementException:
        type_phone = ""
        no_phone = ""
        memory_phone = ""
        storage_phone = ""
    phone_type.append(type_phone)
    return type_phone, no_phone, memory_phone, storage_phone, comment_phone


def main():
    # chrome_options.add_argument('--headless')  # 取消弹出窗口
    for i in range(1, 37):
        url1 = "https://search.jd.com/Search?keyword=%E6%89%8B%E6%9C%BA&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&psort=4&page=" + str(int(i * 2 - 1))
        driver.get(url1)
        time.sleep(delay_time)
        driver.execute_script("window.scrollTo(0, 3 * document.body.scrollHeight / 4);")
        time.sleep(3 * delay_time)
        driver.execute_script("window.scrollTo(0, 5 * document.body.scrollHeight / 6);")  #下拉页面，从而显示隐藏界面
        time.sleep(3 * delay_time)
        url_phone = []
        for j in range(1, 61):  # 一个页面六十部手机，一次爬取
            temp = 1
            while temp == 1:
                try:
                    temp = 0
                    url_temp = driver.find_element_by_xpath("/html/body/div[6]/div[2]/div[2]/div[1]/div/div[2]/ul/li[" + str(j) + "]/div/div[4]/a").get_attribute('href')
                    url_phone.append(url_temp)
                except NoSuchElementException:
                    temp = 1
                    driver.refresh()
                    driver.execute_script("window.scrollTo(0, 3 * document.body.scrollHeight / 4);")
                    time.sleep(2 * delay_time)
                    driver.execute_script("window.scrollTo(0, 5 * document.body.scrollHeight / 6);")
                    time.sleep(2 * delay_time)
        for j in range(0, 60):  # 逐一访问
            time.sleep(delay_time)
            url = url_phone[j]  # 手机界面
            driver.get(url)
            time.sleep(delay_time)
            type_phone, no_phone, memory_phone, storage_phone, comment_phone = click_info()
            time.sleep(delay_time)

            try:
                price = driver.find_element_by_xpath("/html/body/div[6]/div/div[2]/*/div/div[1]/div[2]/span[1]/span[2]").text
                try:
                    price = float(price)
                except ValueError:
                    price = 0.0
            except NoSuchElementException:
                price = 0.0


            if len(memory_phone) != 0 and len(storage_phone) != 0:
                item = [(type_phone, no_phone, memory_phone, storage_phone, price, comment_phone)]
                print(type_phone, end=" ")
                print(no_phone, end=" ")
                print(memory_phone, end=" ")
                print(storage_phone, end=" ")
                print(price, end=" ")
                print(comment_phone)
                keys = ['手机型号', '手机编号', '手机内存', '机身存储', '手机价格', '手机评论']
                df = pd.DataFrame.from_records(item, columns=keys)
                df.to_csv('jingdong.csv', mode='a', index=False, encoding='GBK', header=False)
    


main()