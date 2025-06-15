#!/usr/bin/env python3
"""
Tesco Product Scraper - Fixed Version
Uses updated selectors and improved bot detection avoidance
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import re
import os

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from bs4 import BeautifulSoup
except ImportError:
    print("❌ Missing dependencies. Please install:")
    print("   pip install selenium beautifulsoup4")
    exit(1)

@dataclass
class Product:
    name: str
    price: float
    price_text: str
    category: str
    url: str = ""
    image_url: str = ""

class TescoBrowserScraper:
    def __init__(self, headless=True, debug=True):
        self.debug = debug
        self.headless = headless
        self.driver = None
        self.scraped_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scraped-data')
        os.makedirs(self.scraped_data_dir, exist_ok=True)
        
        # Load search terms from JSON file
        try:
            json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'search-items.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.search_terms = data.get('search_terms', [])
                if not self.search_terms:
                    print("⚠️  No search terms found in search-items.json, using default terms")
                    self.search_terms = [
                        "meat", "chicken", "beef", "pork", "fish", "salmon",
                        "vegetables", "dairy", "cheese", "milk", "bread", "pasta", "rice", "canned"
                    ]
        except Exception as e:
            print(f"⚠️  Error loading search-items.json: {str(e)}")
            print("Using default search terms")
            self.search_terms = [
                "meat", "chicken", "beef", "pork", "fish", "salmon",
                "vegetables", "dairy", "cheese", "milk", "bread", "pasta", "rice", "canned"
            ]
        
        # Initialize or load existing products
        self.products_file = os.path.join(self.scraped_data_dir, 'tesco_products.json')
        self.existing_products = self._load_existing_products()
    
    def _load_existing_products(self) -> List[Product]:
        """Load existing products from JSON file"""
        try:
            if os.path.exists(self.products_file):
                with open(self.products_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [Product(**item) for item in data]
            return []
        except Exception as e:
            print(f"⚠️  Error loading existing products: {str(e)}")
            return []
    
    def _save_product(self, product: Product) -> bool:
        """Save a single product to JSON file"""
        try:
            # Check if product already exists
            for existing in self.existing_products:
                if (existing.name.lower() == product.name.lower() and 
                    abs(existing.price - product.price) < 0.01):
                    if self.debug:
                        print(f"   ⏭️  Skipping duplicate: {product.name} - £{product.price:.2f}")
                    return False
            
            # Add new product
            self.existing_products.append(product)
            
            # Save to file
            with open(self.products_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(p) for p in self.existing_products], f, indent=2, ensure_ascii=False)
            
            if self.debug:
                print(f"   💾 Saved: {product.name} - £{product.price:.2f}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error saving product: {str(e)}")
            return False
    
    def setup_driver(self):
        """Setup Chrome driver with enhanced anti-detection"""
        print("🚀 Setting up Chrome browser...")
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")  # Updated headless flag
        
        # Enhanced anti-detection options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # More realistic user agent (updated)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Additional options for Tesco
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Enhanced script to hide automation
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                window.chrome = {runtime: {}};
            """)
            
            print("✅ Chrome browser ready")
            return True
        except Exception as e:
            print(f"❌ Error setting up Chrome driver: {str(e)}")
            return False
    
    def scrape_search_results(self, search_term: str) -> List[Product]:
        """Scrape products using updated Tesco URL structure"""
        products = []
        
        # Updated URL format for Tesco
        base_url = "https://www.tesco.com/groceries/en-GB/search"
        url = f"{base_url}?query={search_term}"
        
        print(f"\n🔍 Scraping {search_term}")
        print(f"   📱 Loading: {url}")
        
        try:
            self.driver.get(url)
            time.sleep(5)  # Longer wait for Tesco
            
            # Wait for products to load with the correct Tesco selector
            try:
                print(f"   ⏳ Waiting for Tesco products to load...")
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "li[data-testid]"))
                )
                print(f"   ✅ Products loaded!")
                
            except TimeoutException:
                print(f"   ❌ No products found")
                return products
            
            # Get total number of items from pagination
            try:
                print("   🔍 Looking for pagination element...")
                pagination_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='pagination-result-count']"))
                )
                pagination_text = pagination_element.text
                print(f"   📄 Found pagination text: {pagination_text}")
                
                # Extract total number of items using regex
                total_items_match = re.search(r'of\s+([\d,]+)\s+items', pagination_text)
                if total_items_match:
                    # Remove commas and convert to integer
                    total_items = int(total_items_match.group(1).replace(',', ''))
                    items_per_page = 24  # Tesco typically shows 24 items per page
                    total_pages = (total_items + items_per_page - 1) // items_per_page
                    
                    print(f"   📊 Found {total_items} items across {total_pages} pages")
                    
                    # Scrape first page
                    html = self.driver.page_source
                    page_products = self._parse_products_from_html(html, search_term)
                    
                    # Save each product immediately
                    new_products = []
                    for product in page_products:
                        if self._save_product(product):
                            new_products.append(product)
                    products.extend(new_products)
                    print(f"   ✅ Found {len(new_products)} new products on first page")
                    
                    # Scrape remaining pages
                    for page in range(2, total_pages + 1):
                        page_url = f"{url}&page={page}"
                        print(f"   📄 Loading page {page} of {total_pages}: {page_url}")
                        
                        self.driver.get(page_url)
                        time.sleep(5)  # Wait for page load
                        
                        try:
                            # Wait for both pagination and products to load
                            WebDriverWait(self.driver, 15).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='pagination-result-count']"))
                            )
                            WebDriverWait(self.driver, 15).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "li[data-testid]"))
                            )
                            
                            html = self.driver.page_source
                            page_products = self._parse_products_from_html(html, search_term)
                            
                            # Save each product immediately
                            new_products = []
                            for product in page_products:
                                if self._save_product(product):
                                    new_products.append(product)
                            products.extend(new_products)
                            
                            print(f"   ✅ Found {len(new_products)} new products on page {page}")
                            
                        except TimeoutException:
                            print(f"   ⚠️  Timeout loading page {page}")
                            continue
                            
                else:
                    print(f"   ⚠️  Could not parse total items from text: {pagination_text}")
                    # Fallback to single page scraping
                    html = self.driver.page_source
                    page_products = self._parse_products_from_html(html, search_term)
                    
                    # Save each product immediately
                    new_products = []
                    for product in page_products:
                        if self._save_product(product):
                            new_products.append(product)
                    products.extend(new_products)
                    
            except TimeoutException:
                print("   ⚠️  Could not find pagination element - scraping single page only")
                # Try to get the page source anyway
                html = self.driver.page_source
                page_products = self._parse_products_from_html(html, search_term)
                
                # Save each product immediately
                new_products = []
                for product in page_products:
                    if self._save_product(product):
                        new_products.append(product)
                products.extend(new_products)
            
            print(f"   📦 Found {len(products)} new products total")
            
        except Exception as e:
            print(f"   ❌ Error scraping '{search_term}': {str(e)}")
        
        return products
    
    def _parse_products_from_html(self, html: str, category: str) -> List[Product]:
        """Parse products with correct Tesco selectors based on real HTML"""
        products = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Based on actual Tesco HTML structure - products are in <li> with data-testid
            product_containers = soup.find_all('li', {'data-testid': re.compile(r'^\d+')})
            
            print(f"   🔍 Found {len(product_containers)} product containers using correct Tesco selector")
            
            if not product_containers:
                # Fallback: try finding by data-testid attribute (products have numeric IDs)
                product_containers = soup.find_all('li', {'data-testid': re.compile(r'^\d+')})
                print(f"   🔍 Fallback: Found {len(product_containers)} containers with numeric data-testid")
            
            for container in product_containers:
                try:
                    product = self._parse_single_product(container, category)
                    if product:
                        products.append(product)
                except Exception as e:
                    if self.debug:
                        print(f"   ⚠️  Error parsing product: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"   ❌ Error parsing HTML: {str(e)}")
        
        return products
    
    def _parse_single_product(self, container, category: str) -> Optional[Product]:
        """Parse single product using actual Tesco HTML structure"""
        
        # Get product name from the title link (actual structure from HTML)
        name = ""
        try:
            # Look for the title link inside h2 > a with aria-label
            title_link = container.select_one("h2 a[aria-label]")
            if title_link:
                name = title_link.get('aria-label', '').strip()
            
            # Fallback: look for any link with aria-label
            if not name:
                any_link = container.select_one("a[aria-label]")
                if any_link:
                    name = any_link.get('aria-label', '').strip()
        except:
            pass
        
        if not name or len(name) < 3:
            return None
        
        # Get product URL
        url = ""
        try:
            link = container.select_one("h2 a[href]")
            if link:
                href = link.get('href', '')
                if href:
                    url = href if href.startswith('http') else f"https://www.tesco.com{href}"
        except:
            pass
        
        # Get price using actual Tesco structure: p.styled__PriceText-sc-v0qv7n-1
        price_text = ""
        try:
            price_element = container.select_one("p[class*='PriceText']")
            if price_element:
                price_text = price_element.get_text(strip=True)
        except:
            pass
        
        if not price_text:
            return None
        
        # Extract numeric price
        price_match = re.search(r'£(\d+\.?\d*)', price_text)
        if not price_match:
            return None
        
        try:
            price = float(price_match.group(1))
        except ValueError:
            return None
        
        if not (0.1 <= price <= 200):
            return None
        
        # Get image URL using actual structure
        image_url = ""
        try:
            img = container.select_one("img[src]")
            if img:
                image_url = img.get('src', '')
                if image_url and not image_url.startswith('http'):
                    if image_url.startswith('//'):
                        image_url = f"https:{image_url}"
                    elif image_url.startswith('/'):
                        image_url = f"https://www.tesco.com{image_url}"
        except:
            pass
        
        # Clean name
        name = self._clean_text(name)
        
        return Product(
            name=name,
            price=price,
            price_text=price_text,
            category=category,
            url=url,
            image_url=image_url
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text of HTML entities and extra whitespace"""
        replacements = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
            '&#39;': "'", '&nbsp;': ' ', '&pound;': '£'
        }
        
        for entity, replacement in replacements.items():
            text = text.replace(entity, replacement)
        
        return ' '.join(text.split()).strip()
    
    def _remove_duplicates(self, products: List[Product]) -> List[Product]:
        """Remove duplicate products"""
        unique_products = []
        seen = set()
        
        for product in products:
            identifier = (product.name.lower(), product.price)
            if identifier not in seen:
                seen.add(identifier)
                unique_products.append(product)
        
        return unique_products
    
    def scrape_all_categories(self, max_categories: int = None) -> List[Product]:
        """Scrape products from multiple search terms"""
        all_products = []
        
        print("🛒 Starting Tesco browser-based scraping...")
        print("=" * 60)
        
        if not self.setup_driver():
            return all_products
        
        try:
            search_terms_to_try = self.search_terms if max_categories is None else self.search_terms[:max_categories]
            
            print(f"📋 Will scrape {len(search_terms_to_try)} categories")
            
            for search_term in search_terms_to_try:
                print(f"\n🎯 Searching for: {search_term}")
                
                products = self.scrape_search_results(search_term)
                all_products.extend(products)
                
                if products:
                    print(f"✅ Success! Found {len(products)} products for '{search_term}'")
                else:
                    print(f"❌ No products found for '{search_term}'")
                
                # Longer delay between searches for Tesco
                time.sleep(5)
            
            # Remove duplicates
            unique_products = self._remove_duplicates(all_products)
            
            print(f"\n🎉 Scraping complete!")
            print(f"📊 Total unique products found: {len(unique_products)}")
            print(f"🔄 Duplicates removed: {len(all_products) - len(unique_products)}")
            
            return unique_products
            
        finally:
            if self.driver:
                print("\n🔒 Closing browser...")
                self.driver.quit()

# Add the DataAnalyzer class from your original code here...
class DataAnalyzer:
    """Analyze scraped product data"""
    
    def analyze_products(self, products: List[Product]) -> Dict:
        """Analyze the products and return insights"""
        if not products:
            return {"error": "No products to analyze"}
        
        # Basic stats
        total_products = len(products)
        prices = [p.price for p in products]
        
        # Category breakdown
        categories = {}
        for product in products:
            if product.category not in categories:
                categories[product.category] = []
            categories[product.category].append(product.price)
        
        # Price ranges
        price_ranges = {
            'under_1': len([p for p in prices if p < 1]),
            '1_to_5': len([p for p in prices if 1 <= p < 5]),
            '5_to_10': len([p for p in prices if 5 <= p < 10]),
            'over_10': len([p for p in prices if p >= 10])
        }
        
        # Category stats
        category_stats = {}
        for cat, cat_prices in categories.items():
            category_stats[cat] = {
                'count': len(cat_prices),
                'min_price': min(cat_prices),
                'max_price': max(cat_prices),
                'avg_price': sum(cat_prices) / len(cat_prices)
            }
        
        # Find extremes
        cheapest_product = min(products, key=lambda p: p.price)
        most_expensive_product = max(products, key=lambda p: p.price)
        
        return {
            'total_products': total_products,
            'categories': category_stats,
            'price_ranges': price_ranges,
            'cheapest_item': {
                'name': cheapest_product.name,
                'price': cheapest_product.price,
                'category': cheapest_product.category
            },
            'most_expensive_item': {
                'name': most_expensive_product.name,
                'price': most_expensive_product.price,
                'category': most_expensive_product.category
            }
        }

def main():
    """Main function to run the browser-based Tesco scraper"""
    
    print("🛒 Starting Fixed Tesco Browser Scraper...")
    print("=" * 50)
    
    # Test with non-headless mode first for debugging
    headless_choice = input("\nRun in headless mode? (y/n) [n]: ").strip().lower()
    headless = headless_choice == 'y'
    
    # Start with fewer categories for testing
    try:
        max_categories = input("\nHow many categories to scrape? (press Enter for all): ").strip()
        max_categories = None if max_categories == "" else int(max_categories)
        if max_categories is not None and max_categories < 1:
            print("⚠️  Using minimum of 1 category")
            max_categories = 1
    except ValueError:
        print("⚠️  Invalid input, will scrape all categories")
        max_categories = None
    
    print(f"🔳 Running with {'headless' if headless else 'visible'} browser")
    if max_categories:
        print(f"📋 Testing with {max_categories} categories")
    else:
        print(f"📋 Scraping ALL categories with full pagination")
    
    # Initialize scraper
    scraper = TescoBrowserScraper(headless=headless, debug=True)
    
    # Scrape products
    products = scraper.scrape_all_categories(max_categories=max_categories)
    
    if not products:
        print("\n❌ No new products found.")
        print("\n💡 Try these debugging steps:")
        print("   1. Run with headless=False to see what's happening")
        print("   2. Check if Tesco changed their website structure")
        print("   3. Try adding more delays between requests")
        print("   4. Check your internet connection")
        print("   5. All products were already in the database")
        return
    
    # Quick analysis
    print(f"\n✅ Found {len(products)} new products!")
    print("📝 Sample products:")
    for i, product in enumerate(products[:5], 1):
        print(f"   {i}. {product.name} - £{product.price:.2f}")
    
    # Save analysis data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_filename = os.path.join(scraper.scraped_data_dir, f"tesco_analysis_{timestamp}.json")
    
    try:
        analyzer = DataAnalyzer()
        analysis = analyzer.analyze_products(products)
        
        with open(analysis_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Analysis saved to: {analysis_filename}")
        
    except Exception as e:
        print(f"❌ Error saving analysis: {str(e)}")

if __name__ == "__main__":
    main()