from selenium import webdriver
from selenium.webdriver.common.by import By
import time


driver = webdriver.Chrome()

urls = [
        'https://thanhnien.vn/kinh-te.htm', 
        'https://thanhnien.vn/xe.htm', 
        'https://thanhnien.vn/suc-khoe.htm', 
        'https://thanhnien.vn/giao-duc.htm', 
        'https://thanhnien.vn/cong-nghe-game.htm'
       ]

for url in urls:
    driver.get(url)     

    load_counter1 = 5
    load_counter2 = 9

    while (load_counter1 > 0):
        # Get page height before scrolling
        before_scroll_height = driver.execute_script("return document.body.scrollHeight")

        scroll_distance = int(before_scroll_height) - 1000
        driver.execute_script(f"window.scrollTo(0,{scroll_distance})")
        time.sleep(3)
        load_counter1 -= 1

        # Get page height after scrolling
        after_scroll_height = driver.execute_script("return document.body.scrollHeight")
        print(before_scroll_height)

        # Check button "Load more" (i.e can't load more content when croll)
        if after_scroll_height == before_scroll_height:
            print("after_scroll_height == before_scroll_height")
            break
    
    if url == 'https://thanhnien.vn/cong-nghe-game.htm':
        # Only Use for cong-nghe-game.htm
        clickable = driver.find_element(By.CSS_SELECTOR, "a.view-more.btn-viewmore") 
    else:
        # Use for others
        clickable = driver.find_element(By.CSS_SELECTOR, "a.list__center.view-more.list__viewmore")
    
    driver.execute_script("arguments[0].scrollIntoView();", clickable)
    driver.execute_script("arguments[0].click();", clickable)

    # ActionChains(driver)\
    # .click(clickable)\
    # .perform()

    print("Clicked Button 'Show more'")

    time.sleep(4)

    while (load_counter2 > 0):
        before_scroll_height = driver.execute_script("return document.body.scrollHeight")
        scroll_distance = int(before_scroll_height) - 1500
        driver.execute_script(f"window.scrollTo(0,{scroll_distance})")
        time.sleep(3)
        load_counter2 -= 1

    # Get page source
    full_page_html = driver.page_source

    filename = 'data-mining/Project/Crawler/content-web/' + url[21:] + 'l'

    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(full_page_html)
        print(f"Content saved to {filename}")
    except IOError as e:
        print("An error occurred while saving the file:", e)


driver.quit()

