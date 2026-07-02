import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def wake_streamlit():
    url = os.getenv("STREAMLIT_URL")
    if not url:
        print("Error: STREAMLIT_URL environment variable is missing.")
        return

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    try:
        print(f"Opening: {url}")
        driver.get(url)
        time.sleep(5)

        try:
            wake_button = driver.find_element(
                By.XPATH, "//button[contains(text(), 'Wake up app')]"
            )
            wake_button.click()
            print("App was asleep — clicked Wake up button.")
            time.sleep(10)
        except Exception:
            print("App already awake.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    wake_streamlit()
