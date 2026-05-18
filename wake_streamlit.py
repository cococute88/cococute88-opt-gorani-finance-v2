from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

APP_URL = "https://gorani-finance.streamlit.app/"

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)

try:
    print("Opening app...")
    driver.get(APP_URL)

    time.sleep(10)

    page_text = driver.page_source

    if "This app has gone to sleep" in page_text or "Yes, get this app back up" in page_text:
        print("App is sleeping. Trying to click wake button...")

        buttons = driver.find_elements(By.TAG_NAME, "button")
        clicked = False

        for button in buttons:
            button_text = button.text.lower()
            print("Button found:", button.text)

            if "get this app back up" in button_text or "yes" in button_text:
                button.click()
                clicked = True
                print("Wake button clicked.")
                break

        if not clicked:
            print("Wake button was not found.")
    else:
        print("App seems already awake.")

    time.sleep(20)
    print("Finished.")

except Exception as e:
    print("Error occurred:")
    print(e)

finally:
    driver.quit()
