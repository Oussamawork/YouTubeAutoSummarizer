from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Set up Chrome options to disable images
chrome_options = Options()
chrome_options.add_argument("--blink-settings=imagesEnabled=false")

# Path to the ChromeDriver
driver_path = "chromedriver/chromedriver"  # Replace with the path to your ChromeDriver

# Set up Selenium WebDriver with options
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Open the website
    driver.get('https://www.youtube.com/@patloeber/videos')

    # Wait for the page to load and the element to appear
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, 'thumbnail'))  # Wait until the thumbnail element is found
    )

    # Get the page source after rendering
    page_source = driver.page_source

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')

    # Find the anchor tag by its ID
    anchor_tag = soup.find('a', id='thumbnail')

    # Extract the href attribute
    if anchor_tag and 'href' in anchor_tag.attrs:
        href = anchor_tag['href']
        print("Href:", href)
        # Complete URL if it's relative
        if href.startswith('/'):
            full_url = f"https://www.youtube.com{href}"
        else:
            full_url = href
        print("Full URL:", full_url)
    else:
        print("Href attribute not found!")

finally:
    # Close the WebDriver
    driver.quit()
