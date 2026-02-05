import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": 
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

def extract_reviews(url, limit=50):
    reviews = []

    try:
        page = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(page.content, "html.parser")

        # Amazon review blocks
        review_blocks = soup.find_all("span", {"data-hook": "review-body"})

        for block in review_blocks[:limit]:
            text = block.get_text(strip=True)
            reviews.append(text)

    except Exception as e:
        print("Scraping Error:", e)

    return reviews
