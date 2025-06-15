#!/usr/bin/env python3
"""
ASDA Product Scraper - Browser Automation Version
Uses Selenium to wait for JavaScript content to load properly
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

class AsdaBrowserScraper:
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
        self.products_file = os.path.join(self.scraped_data_dir, 'asda_products.json')
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
        """Setup Chrome driver with appropriate options"""
        print("🚀 Setting up Chrome browser...")
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        # Add options to avoid detection and improve performance
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Set a realistic user agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("✅ Chrome browser ready")
            return True
        except Exception as e:
            print(f"❌ Failed to setup Chrome driver: {str(e)}")
            return False
    
    def get_total_product_count(self) -> int:
        """Extract total product count from the search results page"""
        try:
            # Look for the total count in the pagination element
            count_element = self.driver.find_element(By.CSS_SELECTOR, "span.search-pagination__count-text--total")
            count_text = count_element.text
            
            # Extract number from text
            count_match = re.search(r'(\d+)', count_text)
            if count_match:
                total_count = int(count_match.group(1))
                print(f"   📊 Total products available: {total_count}")
                return total_count
            else:
                print(f"   ⚠️  Could not parse count from: {count_text}")
                return 0
                
        except NoSuchElementException:
            print(f"   ⚠️  Product count element not found")
            return 0
        except Exception as e:
            print(f"   ⚠️  Error getting product count: {str(e)}")
            return 0

    def scrape_search_results(self, search_term: str) -> List[Product]:
        """Scrape products from search results, waiting for JavaScript to load"""
        products = []
        
        # First, load the search page
        url = f"https://groceries.asda.com/search/{search_term}/products"
        print(f"\n🔍 Scraping {search_term}")
        print(f"   📱 Loading: {url}")
        
        try:
            self.driver.get(url)
            time.sleep(3)  # Wait for page load
            
            # Get total product count
            total_products = self.get_total_product_count()
            if total_products == 0:
                print(f"   ❌ No products found for '{search_term}'")
                return products
            
            # Calculate total pages needed (ASDA shows 60 items per page)
            items_per_page = 60
            total_pages = (total_products + items_per_page - 1) // items_per_page
            print(f"   📄 Need to scrape {total_pages} pages ({total_products} items ÷ {items_per_page} per page)")
            
            # Scrape each page
            for page in range(1, total_pages + 1):
                print(f"\n   📖 Scraping page {page}/{total_pages}")
                
                # Navigate to the specific page (if not already on page 1)
                if page > 1:
                    page_url = f"{url}?page={page}"
                    print(f"      Loading: {page_url}")
                    self.driver.get(page_url)
                    time.sleep(3)  # Wait for page load
                
                try:
                    # Wait for product list to appear
                    print(f"      ⏳ Waiting for products to load...")
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "ul.co-product-list__main-cntr"))
                    )
                    print(f"      ✅ Products loaded!")
                    
                except TimeoutException:
                    print(f"      ⏰ Timeout waiting for products on page {page}")
                    continue
                
                # Additional wait to ensure all products are loaded
                time.sleep(2)
                
                # Get the page source after JavaScript has executed
                html = self.driver.page_source
                
                # Parse products from the loaded HTML
                page_products = self._parse_products_from_html(html, search_term)
                
                # Save each product immediately
                new_products = []
                for product in page_products:
                    if self._save_product(product):
                        new_products.append(product)
                
                products.extend(new_products)
                print(f"      📦 Found {len(new_products)} new products on page {page}")
                
                time.sleep(2)  # Be respectful between requests
                
        except Exception as e:
            print(f"   ❌ Error scraping '{search_term}': {str(e)}")
        
        return products
    
    def _parse_products_from_html(self, html: str, category: str) -> List[Product]:
        """Parse products from the fully loaded HTML using BeautifulSoup"""
        products = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all product containers
            product_containers = soup.find_all('li', class_='co-item')
            
            print(f"   🔍 Found {len(product_containers)} product containers")
            
            if self.debug:
                # Print first container HTML for debugging
                if product_containers:
                    print("\n   📄 First product container HTML:")
                    print("   " + "-" * 50)
                    print("   " + str(product_containers[0])[:500] + "...")
                    print("   " + "-" * 50)
            
            for container in product_containers:
                try:
                    product = self._parse_single_product(container, category)
                    if product:
                        products.append(product)
                except Exception as e:
                    if self.debug:
                        print(f"   ⚠️  Error parsing product: {str(e)}")
                    continue
            
            if self.debug:
                print(f"\n   📊 Parsing summary:")
                print(f"   - Total containers found: {len(product_containers)}")
                print(f"   - Successfully parsed: {len(products)}")
                print(f"   - Failed to parse: {len(product_containers) - len(products)}")
            
        except Exception as e:
            print(f"   ❌ Error parsing HTML: {str(e)}")
        
        return products
    
    def _parse_single_product(self, container, category: str) -> Optional[Product]:
        """Parse a single product from its container"""
        try:
            # Get product name from title - ASDA uses h3.co-product__title a
            title_element = container.select_one("h3.co-product__title a")
            if not title_element:
                if self.debug:
                    print(f"   ⚠️  No title element found in container")
                return None
            
            name = title_element.get_text(strip=True)
            if not name or len(name) < 3:
                if self.debug:
                    print(f"   ⚠️  Invalid name: {name}")
                return None
            
            # Get price - ASDA uses strong.co-product__price
            price_element = container.select_one("strong.co-product__price")
            if not price_element:
                if self.debug:
                    print(f"   ⚠️  No price element found for: {name}")
                return None
            
            price_text = price_element.get_text(strip=True)
            
            # Extract numeric price
            price_match = re.search(r'£(\d+\.?\d*)', price_text)
            if not price_match:
                if self.debug:
                    print(f"   ⚠️  Could not extract price from: {price_text}")
                return None
            
            try:
                price = float(price_match.group(1))
            except ValueError:
                if self.debug:
                    print(f"   ⚠️  Invalid price value: {price_match.group(1)}")
                return None
            
            if not (0.1 <= price <= 200):
                if self.debug:
                    print(f"   ⚠️  Price out of range: {price}")
                return None
            
            # Get product URL
            url = ""
            if title_element.get('href'):
                href = title_element.get('href')
                url = href if href.startswith('http') else f"https://groceries.asda.com{href}"
            
            # Get image URL - ASDA uses img.asda-img
            img_element = container.select_one("img.asda-img")
            image_url = ""
            if img_element:
                # Try srcset first for higher quality
                srcset = img_element.get('srcset', '')
                if srcset:
                    # Get the highest quality image (usually the last one)
                    srcset_urls = [part.strip().split(' ')[0] for part in srcset.split(',')]
                    if srcset_urls:
                        image_url = srcset_urls[-1]
                
                # Fallback to src
                if not image_url:
                    image_url = img_element.get('src', '')
                
                # Ensure full URL
                if image_url and not image_url.startswith('http'):
                    if image_url.startswith('//'):
                        image_url = f"https:{image_url}"
                    else:
                        image_url = f"https://groceries.asda.com{image_url}"
            
            # Clean HTML entities from name
            name = self._clean_text(name)
            
            if self.debug:
                print(f"   ✅ Successfully parsed: {name} - £{price:.2f}")
            
            return Product(
                name=name,
                price=price,
                price_text=price_text,
                category=category,
                url=url,
                image_url=image_url
            )
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ Error parsing product: {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text of HTML entities and extra whitespace"""
        # Replace HTML entities
        replacements = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&nbsp;': ' '
        }
        
        for entity, replacement in replacements.items():
            text = text.replace(entity, replacement)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def scrape_all_categories(self, max_categories: int = None) -> List[Product]:
        """Scrape products from multiple search terms"""
        all_products = []
        
        print("🛒 Starting ASDA browser-based scraping...")
        print("=" * 60)
        
        if not self.setup_driver():
            return all_products
        
        try:
            # Use all search terms if max_categories is None
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
                
                # Small delay between different searches
                time.sleep(3)
            
            # Remove duplicates
            unique_products = self._remove_duplicates(all_products)
            
            print(f"\n🎉 Scraping complete!")
            print(f"   Total products found: {len(unique_products)}")
            print(f"   Search terms tried: {len(search_terms_to_try)}")
            
            return unique_products
            
        finally:
            # Always close the browser
            if self.driver:
                print("\n🔒 Closing browser...")
                self.driver.quit()
    
    def _remove_duplicates(self, products: List[Product]) -> List[Product]:
        """Remove duplicate products based on name and price"""
        unique_products = []
        seen = set()
        
        for product in products:
            key = f"{product.name.lower().strip()}-{product.price:.2f}"
            if key not in seen:
                seen.add(key)
                unique_products.append(product)
        
        return sorted(unique_products, key=lambda p: (p.category, p.price))

class DataAnalyzer:
    def analyze_products(self, products: List[Product]) -> Dict:
        """Analyze the scraped product data"""
        
        if not products:
            return {"error": "No products to analyze"}
        
        # Basic statistics
        total_products = len(products)
        categories = {}
        price_ranges = {'under_1': 0, '1_to_5': 0, '5_to_10': 0, 'over_10': 0}
        
        # Categorize products
        for product in products:
            category = product.category
            if category not in categories:
                categories[category] = {'count': 0, 'prices': []}
            
            categories[category]['count'] += 1
            categories[category]['prices'].append(product.price)
            
            # Price ranges
            if product.price < 1:
                price_ranges['under_1'] += 1
            elif product.price < 5:
                price_ranges['1_to_5'] += 1
            elif product.price < 10:
                price_ranges['5_to_10'] += 1
            else:
                price_ranges['over_10'] += 1
        
        # Calculate statistics per category
        for category in categories:
            prices = categories[category]['prices']
            categories[category]['avg_price'] = sum(prices) / len(prices)
            categories[category]['min_price'] = min(prices)
            categories[category]['max_price'] = max(prices)
        
        # Find cheapest and most expensive items
        cheapest = min(products, key=lambda x: x.price)
        most_expensive = max(products, key=lambda x: x.price)
        
        return {
            'total_products': total_products,
            'categories': categories,
            'price_ranges': price_ranges,
            'cheapest_item': {
                'name': cheapest.name,
                'price': cheapest.price,
                'category': cheapest.category
            },
            'most_expensive_item': {
                'name': most_expensive.name,
                'price': most_expensive.price,
                'category': most_expensive.category
            },
            'analysis_date': datetime.now().isoformat()
        }

def main():
    """Main function to run the browser-based ASDA scraper"""
    
    print("🛒 Starting ASDA Browser Scraper...")
    print("=" * 50)
    print("⚠️  This will open a Chrome browser to load JavaScript content")
    
    # Ask user preference for headless mode
    headless_choice = input("\nRun in headless mode? (y/n) [y]: ").strip().lower()
    headless = headless_choice != 'n'
    
    # Ask about number of categories
    try:
        max_categories = input("\nHow many categories to scrape? (press Enter for all): ").strip()
        max_categories = None if max_categories == "" else int(max_categories)
        if max_categories is not None and max_categories < 1:
            print("⚠️  Using minimum of 1 category")
            max_categories = 1
    except ValueError:
        print("⚠️  Invalid input, will scrape all categories")
        max_categories = None
    
    if headless:
        print("🔲 Running in headless mode (no visible browser)")
    else:
        print("🔳 Running with visible browser (you can watch the scraping)")
    
    # Initialize scraper
    scraper = AsdaBrowserScraper(headless=headless, debug=True)
    
    # Scrape products
    products = scraper.scrape_all_categories(max_categories=max_categories)
    
    if not products:
        print("\n❌ No new products found.")
        print("\n💡 Possible issues:")
        print("   1. ChromeDriver not installed or not in PATH")
        print("   2. ASDA's website blocking automated browsers")
        print("   3. Network connectivity issues")
        print("   4. Website structure has changed")
        print("   5. All products were already in the database")
        return
    
    # Analyze the data
    print("\n📊 Analyzing scraped data...")
    analyzer = DataAnalyzer()
    analysis = analyzer.analyze_products(products)
    
    # Save analysis data as JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_filename = os.path.join(scraper.scraped_data_dir, f"asda_analysis_{timestamp}.json")
    try:
        with open(analysis_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Analysis data saved to: {analysis_filename}")
        
    except Exception as e:
        print(f"❌ Error saving analysis data: {str(e)}")

if __name__ == "__main__":
    main() 