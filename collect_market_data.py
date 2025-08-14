import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager

class CoinMarketCapScraper:
    """
    This class is responsible for scraping historical market data from CoinMarketCap. It
    uses Selenium to automate browsing and BeautifulSoup to parse the HTML content.
    """
    def __init__(self, driver):
        self.driver = driver

    def establish_connection(self, url):
        """
        Establishes a connection to the specified URL and waits for the page to load.
        """
        try:
            print(f"Connecting to {url}...")
            self.driver.get(url)
            time.sleep(15)
            print("Connection established.")

        except Exception as e:
            print(f"Connection error: {e}")

    def load_more_data(self):
        """
        Loads more market data by clicking the "Load More" button until it is no longer available.
        """
        print("Loading all available market data...")

        while True:
            try:
                button = self.driver.find_element(By.CLASS_NAME, 'sc-c0a10c7b-0.keYVYU')
                self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                button.click()
                time.sleep(3)

            except:
                print("No more data to load.")
                break

    def parse_data(self, soup):
        """
        Parses the HTML content to extract market data.
        """
        print("Parsing market data...")
        rows = soup.find_all('td')
        data = []

        for i in range(0, len(rows), 7):
            cells = [cell.text for cell in rows[i:i+7]]

            if len(cells) == 7:
                date = datetime.strptime(cells[0], '%b %d, %Y')
                numbers = [float(cell.replace('$', '').replace(',', '')) for cell in cells[1:]]
                data.append((date, *numbers))

        return data

    def scrape_market_data(self, asset, dir):
        """
        Scrapes the market data for a given asset and saves it to a CSV file.
        """
        print(f"Scraping market data...")
        os.makedirs(dir, exist_ok=True)

        self.load_more_data()
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        market_data = self.parse_data(soup)

        df = pd.DataFrame(market_data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'marketcap'])
        output_path = os.path.join(dir, f'{asset}.csv')
        df.to_csv(output_path, index=False)

        print(f"Completed scraping market data.\n")

if __name__ == "__main__":
    # Assets to scrape
    assets = {
        'BTC': 'https://coinmarketcap.com/currencies/bitcoin/historical-data/',
        'ETH': 'https://coinmarketcap.com/currencies/ethereum/historical-data/',
        'XRP': 'https://coinmarketcap.com/currencies/xrp/historical-data/',
        'SOL': 'https://coinmarketcap.com/currencies/solana/historical-data/',
        'DOGE': 'https://coinmarketcap.com/currencies/dogecoin/historical-data/',
        'TRX': 'https://coinmarketcap.com/currencies/tron/historical-data/',
        'ADA': 'https://coinmarketcap.com/currencies/cardano/historical-data/'
    }

    print("Starting market data collection from CoinMarketCap...\n")

    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
    scraper = CoinMarketCapScraper(driver)
    market_data_path = Path("market_data")

    try:
        for asset, url in assets.items():
            print(f"Collecting {asset} market data.")
            scraper.establish_connection(url)
            scraper.scrape_market_data(asset, market_data_path)
            time.sleep(3)

    finally:
        driver.quit()