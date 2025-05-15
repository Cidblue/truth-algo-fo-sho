"""
Web Scraper - Tool for collecting statements from news sources
for analysis with the Truth Algorithm.

Usage:
    python -m utils.scraper [--output output.txt] [--count 100] [--append]
"""

import sys
import requests
from bs4 import BeautifulSoup
import re
import argparse
from pathlib import Path
import datetime
import time
import random
from urllib.parse import urlparse, urljoin, quote_plus

# News sources (used to guide search for mix of stories)
NEWS_SOURCES = [
    "https://www.bbc.com",  # Likely verified
    "https://www.reuters.com",  # Likely verified
    "https://www.ap.org",  # Likely verified
    "https://www.theguardian.com",  # Likely verified
    "https://www.cnn.com",  # Mainstream
    "https://www.nbcnews.com",  # Mainstream
    "https://nypost.com",  # Tabloid
    "https://www.dailymail.co.uk",  # Tabloid
    "https://www.breitbart.com",  # Tabloid
    "https://www.snopes.com",  # Fact-checking
    "https://www.politifact.com"  # Fact-checking
]

# Search terms (including 2020 election and Covid-19)
SEARCH_TERMS = [
    "2020 U.S. election",
    "Covid-19 pandemic",
    "politics news",
    "health crisis",
    "climate change",
    "crime news",
    "technology breakthrough",
    "natural disaster",
    "economy news"
]

# Time ranges (2016–2025)
TIME_RANGES = [
    "2016-01-01..2016-12-31",
    "2017-01-01..2017-12-31",
    "2018-01-01..2018-12-31",
    "2019-01-01..2019-12-31",
    "2020-01-01..2020-12-31",
    "2021-01-01..2021-12-31",
    "2022-01-01..2022-12-31",
    "2023-01-01..2023-12-31",
    "2024-01-01..2024-12-31",
    "2025-01-01..2025-05-05"
]

# Headers to avoid being blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_article_urls(search_term, time_range, max_results=5):
    """Search Google News for articles in a time range."""
    query = search_term.replace(" ", "+")
    url = f"https://www.google.com/search?q={query}&tbm=nws&tbs=cdr:1,cd_min:{time_range.split('..')[0]},cd_max:{time_range.split('..')[1]}"
    try:
        print(f"Requesting: {url}")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        # Debug: Check if we're getting a response
        print(f"Response status: {response.status_code}")

        soup = BeautifulSoup(response.text, "html.parser")

        # Debug: Check if we're getting any links at all
        all_links = soup.find_all("a", href=True)
        print(f"Found {len(all_links)} links on the page")

        links = []
        for a in all_links:
            href = a["href"]
            if href.startswith("/url?q="):
                link = href.split("/url?q=")[1].split("&")[0]
                print(f"Found link: {link}")
                # More permissive check - accept any news-like domain
                if any(source in link for source in NEWS_SOURCES) or any(domain in link for domain in ["news", "article", "story"]):
                    links.append(link)

        # If no links found with strict filtering, try a more permissive approach
        if not links and all_links:
            print("No news source links found, trying more permissive approach")
            for a in all_links:
                href = a["href"]
                if href.startswith("/url?q=") and "google" not in href:
                    link = href.split("/url?q=")[1].split("&")[0]
                    links.append(link)

        print(f"Returning {len(links[:max_results])} article URLs")
        return links[:max_results]
    except Exception as e:
        print(f"Error searching for {search_term} in {time_range}: {e}")
        return []


def extract_statements(article_url):
    """Extract all sentences and quotes from an article, preserving order."""
    try:
        response = requests.get(article_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract metadata with better fallbacks
        title = soup.find("title").text if soup.find(
            "title") else "Unknown Title"

        # Try multiple approaches to find date
        date = "Unknown Date"

        # 1. Try meta tags first (most reliable)
        meta_date_tags = [
            soup.find("meta", {"name": "date"}),
            soup.find("meta", {"property": "article:published_time"}),
            soup.find("meta", {"property": "og:published_time"}),
            soup.find("meta", {"name": "publish-date"}),
            soup.find("meta", {"name": "publication_date"}),
            soup.find("meta", {"name": "published-date"}),
            soup.find("meta", {"itemprop": "datePublished"}),
            soup.find("meta", {"name": "pubdate"})
        ]

        for tag in meta_date_tags:
            if tag and tag.get("content"):
                date = tag["content"]
                break

        # 2. Try time elements
        if date == "Unknown Date":
            time_elements = soup.find_all("time")
            for time_el in time_elements:
                if time_el.get("datetime"):
                    date = time_el["datetime"]
                    break
                elif time_el.text and re.search(r'\d{4}', time_el.text):
                    date = time_el.text.strip()
                    break

        # 3. Try classes that typically contain dates
        if date == "Unknown Date":
            date_classes = soup.find_all(
                class_=re.compile("date|publish|time|posted", re.I))
            for el in date_classes:
                if el.text and re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}', el.text, re.I):
                    date = el.text.strip()
                    break

        # 4. Look for date patterns in the URL itself
        if date == "Unknown Date":
            url_date_match = re.search(
                r'/(\d{4}/\d{1,2}/\d{1,2})/', article_url)
            if url_date_match:
                date = url_date_match.group(1).replace('/', '-')

        # 5. If all else fails, use current date
        if date == "Unknown Date":
            date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # Try to find author with better fallbacks
        author = "Unknown"
        author_elements = [
            soup.find("meta", {"name": "author"}),
            soup.find("meta", {"property": "article:author"}),
            soup.find(class_=re.compile("author|byline", re.I)),
            soup.find(rel="author"),
            soup.find("a", href=re.compile("/author/", re.I))
        ]

        for element in author_elements:
            if element:
                if element.get("content"):
                    author = element["content"]
                    break
                elif element.text and len(element.text.strip()) > 0:
                    author = element.text.strip()
                    break

        # Extract domain as source
        domain = urlparse(article_url).netloc

        # Find main content area (most likely to contain the article)
        main_content = None
        content_candidates = [
            soup.find("article"),
            soup.find(class_=re.compile("article|story|content", re.I)),
            soup.find(id=re.compile("article|story|content", re.I)),
            soup.find("main"),
            soup.find("body")  # Fallback to body if nothing else found
        ]

        for candidate in content_candidates:
            if candidate:
                main_content = candidate
                break

        if not main_content:
            main_content = soup

        # Extract all paragraphs from main content
        paragraphs = main_content.find_all("p")

        # If no paragraphs found, try to get text directly
        if not paragraphs:
            text = main_content.get_text(separator="\n")
            paragraphs = [type('obj', (object,), {'text': t.strip()}) for t in text.split(
                '\n') if t.strip()]

        # Process all paragraphs to extract sentences
        statements = []

        for p in paragraphs:
            text = p.text.strip()
            if not text or len(text) < 10:  # Skip very short paragraphs
                continue

            # Try to split into sentences using NLTK if available
            try:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(text)
            except:
                # Fallback to regex-based sentence splitting
                sentences = re.split(r'(?<=[.!?])\s+', text)

            # Add each sentence as a statement
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Skip very short sentences
                    statements.append({
                        "statement": sentence,
                        "context": text[:200] + "..." if len(text) > 200 else text,
                        "is_quote": bool(re.search(r'"[^"]+"', sentence))
                    })

        return {
            "url": article_url,
            "title": title,
            "date": date,
            "author": author,
            "domain": domain,
            "statements": statements
        }
    except Exception as e:
        print(f"Error scraping {article_url}: {e}")
        return None


def save_statements(data, output_file, append=False):
    """Save statements to a text file with improved article grouping."""
    mode = "a" if append else "w"

    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, mode, encoding="utf-8") as f:
        for article_index, article in enumerate(data):
            # Add article separator and metadata
            if article_index > 0:
                f.write("\n")  # Extra line between articles

            # Try to parse date into ISO format for consistency
            iso_date = "Unknown Date"
            try:
                # Try various date formats
                for date_format in [
                    "%Y-%m-%d",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%B %d, %Y",
                    "%b %d, %Y"
                ]:
                    try:
                        date_obj = datetime.datetime.strptime(
                            article["date"], date_format)
                        iso_date = date_obj.isoformat() + ".000Z"
                        break
                    except (ValueError, TypeError):
                        continue

                # If we couldn't parse it but it looks like an ISO date already
                if iso_date == "Unknown Date" and re.match(r'\d{4}-\d{2}-\d{2}', article["date"]):
                    iso_date = article["date"]
            except:
                pass

            # Get domain as source
            domain = article["domain"]

            # Write each statement with metadata
            for stmt in article["statements"]:
                # Format statement with metadata including ISO date
                formatted_stmt = f"({iso_date}, {domain}){stmt['statement']}"
                f.write(formatted_stmt + "\n")


def extract_statements_from_news_site(url):
    """Extract statements directly from a news site's front page."""
    try:
        print(f"Requesting: {url}")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        print(f"Response status: {response.status_code}")
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract domain for source attribution
        domain = urlparse(url).netloc

        # Get current date for statements without dates
        current_date = datetime.datetime.now().strftime("%b %d, %Y")

        # Find all links that might be articles
        article_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Skip navigation, social media, etc.
            if any(skip in href.lower() for skip in ['javascript:', 'mailto:', '#', 'twitter.com', 'facebook.com']):
                continue

            # Make relative URLs absolute
            if href.startswith('/'):
                href = f"https://{domain}{href}"
            elif not href.startswith(('http://', 'https://')):
                href = f"{url.rstrip('/')}/{href.lstrip('/')}"

            # Look for article indicators in URL
            if any(indicator in href.lower() for indicator in ['article', 'story', 'news', '/20', 'politics', 'world', 'health']):
                article_links.append(href)

        # Limit to reasonable number and remove duplicates
        article_links = list(set(article_links))[:10]
        print(f"Found {len(article_links)} potential article links")

        # Extract statements from each article
        all_statements = []
        for link in article_links:
            try:
                article_data = extract_statements(link)
                if article_data and article_data["statements"]:
                    all_statements.append(article_data)
                    print(
                        f"  - Found {len(article_data['statements'])} statements in {link}")
            except Exception as e:
                print(f"  - Error processing {link}: {e}")

        return all_statements
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []


def main():
    """Main entry point for the scraper."""
    parser = argparse.ArgumentParser(
        description='Scrape news sources for Truth Algorithm analysis')
    parser.add_argument('--output', default='docs/Statementstoclass.txt',
                        help='Output file for scraped statements (default: docs/Statementstoclass.txt)')
    parser.add_argument('--count', type=int, default=50,
                        help='Maximum number of statements to collect (default: 50)')
    parser.add_argument('--append', action='store_true',
                        help='Append to output file instead of overwriting')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay between requests in seconds (default: 2.0)')

    args = parser.parse_args()

    # Direct news sources to scrape
    news_sources = [
        "https://www.bbc.com/news",
        "https://www.reuters.com/world/",
        "https://www.theguardian.com/us-news",
        "https://www.cnn.com/politics",
        "https://www.nbcnews.com/politics",
        "https://nypost.com/news/",
        "https://www.dailymail.co.uk/news/",
        "https://www.breitbart.com/politics/",
        "https://www.snopes.com/fact-check/",
        "https://www.politifact.com/factchecks/",
        "https://apnews.com/",
        "https://www.foxnews.com/politics",
        "https://www.washingtonpost.com/politics/",
        "https://www.nytimes.com/section/politics"
    ]

    all_articles = []
    statement_count = 0

    # Scrape each news source
    for url in news_sources:
        if statement_count >= args.count:
            break

        print(f"\nScraping news source: {url}")
        articles = extract_statements_from_news_site(url)

        if articles:
            all_articles.extend(articles)
            new_statements = sum(len(article["statements"])
                                 for article in articles)
            statement_count += new_statements
            print(
                f"Found {new_statements} statements from {url}. Total: {statement_count}")

        time.sleep(args.delay)  # Polite scraping

    # Save results
    if all_articles:
        save_statements(all_articles, args.output, args.append)
        print(
            f"Collected {statement_count} statements. Saved to {args.output}")
    else:
        print("No statements found.")


if __name__ == "__main__":
    main()
