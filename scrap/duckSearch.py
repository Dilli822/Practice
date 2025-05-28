import requests
from bs4 import BeautifulSoup
import time
import re

def duckduckgo_search(query, max_results=5):
    search_url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.post(search_url, data=params, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        results = []
        for a in soup.find_all("a", class_="result__a", limit=max_results):
            href = a.get("href")
            if href:
                results.append(href)
        return results
    except Exception as e:
        print("Search error:", e)
        return []

def is_valid_paragraph(text):
    invalid_keywords = ["cookie", "advertisement", "privacy", "subscribe", "terms"]
    if any(word in text.lower() for word in invalid_keywords):
        return False
    return len(text) > 80

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_page_summary(url, max_paragraphs=3):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        paragraphs = soup.find_all("p")
        summary_paras = []
        for p in paragraphs:
            text = clean_text(p.get_text())
            if is_valid_paragraph(text):
                summary_paras.append(text)
            if len(summary_paras) >= max_paragraphs:
                break
        
        if summary_paras:
            return "\n\n".join(summary_paras)
        
        if soup.title:
            return clean_text(soup.title.get_text())
        
        return "No good summary found."
    except Exception as e:
        return f"Failed to load page: {e}"

def get_best_summary(urls, max_paragraphs=3):
    for url in urls[:3]:
        summary = scrape_page_summary(url, max_paragraphs)
        if len(summary) > 100:
            return url, summary
    return urls[0], scrape_page_summary(urls[0], max_paragraphs)

def main():
    print("Free general web search + scrape summary")
    print("Type 'exit' to quit.")
    while True:
        query = input("\nAsk something: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        urls = duckduckgo_search(query)
        if not urls:
            print("No results found.")
            continue

        best_url, summary = get_best_summary(urls)
        print(f"Best result: {best_url}")
        print("\n📝 Summary:\n", summary)
        time.sleep(1)

if __name__ == "__main__":
    main()
