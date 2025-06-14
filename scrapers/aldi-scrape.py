#!/usr/bin/env python3
"""
Aldi Product Scraper - Browser Automation Version
Uses Selenium to wait for JavaScript content to load properly
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import re

# You'll need to install these packages:
# pip install selenium beautifulsoup4

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
    print("   You'll also need ChromeDriver: https://chromedriver.chromium.org/")
    exit(1)

@dataclass
class Product:
    name: str
    price: float
    price_text: str
    category: str
    url: str = ""
    image_url: str = ""

class AldiBrowserScraper:
    def __init__(self, headless=True, debug=True):
        self.debug = debug
        self.headless = headless
        self.driver = None
        
        # Search terms to try
        self.search_terms = [
            "meat", "chicken", "beef", "pork", "fish", "salmon",
            "vegetables", "dairy", "cheese", "milk", "bread", "pasta", "rice", 'canned'
        ]
    
    def setup_driver(self):
        """Setup Chrome driver with appropriate options"""
        print("🚀 Setting up Chrome browser...")
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        
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
            print("💡 Make sure ChromeDriver is installed and in your PATH")
            print("   Download from: https://chromedriver.chromium.org/")
            return False
    
    def get_total_product_count(self) -> int:
        """Extract total product count from the search results page"""
        try:
            # Look for the span with product count
            count_element = self.driver.find_element(By.CSS_SELECTOR, "span[data-mn='product-search-count']")
            count_text = count_element.text
            
            # Extract number from text like "(146)"
            count_match = re.search(r'\((\d+)\)', count_text)
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

    def scrape_search_results(self, search_term: str, max_pages: int = 7) -> List[Product]:
        """Scrape products from search results, waiting for JavaScript to load"""
        products = []
        
        # First, load the search page to get total count
        url = f"https://www.aldi.co.uk/results?q={search_term}&sort=price_asc"
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
            
            # Calculate total pages needed (30 items per page)
            items_per_page = 30
            total_pages = (total_products + items_per_page - 1) // items_per_page
            print(f"   📄 Need to scrape {total_pages} pages ({total_products} items ÷ {items_per_page} per page)")
            
            # Scrape each page
            for page in range(1, total_pages + 1):
                print(f"\n   📖 Scraping page {page}/{total_pages}")
                
                # Navigate to the specific page (if not already on page 1)
                if page > 1:
                    page_url = f"{url}&page={page}"
                    print(f"      Loading: {page_url}")
                    self.driver.get(page_url)
                    time.sleep(3)  # Wait for page load
                
                try:
                    # Wait for product tiles to appear
                    print(f"      ⏳ Waiting for products to load...")
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "product-tile"))
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
                products.extend(page_products)
                
                print(f"      📦 Found {len(page_products)} products on page {page}")
                
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
            product_containers = soup.find_all('div', class_=lambda x: x and 'product-teaser-item' in x)
            
            print(f"   🔍 Found {len(product_containers)} product containers")
            
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
        """Parse a single product from its container"""
        
        # Find the product tile within the container
        product_tile = container.find('div', class_=lambda x: x and 'product-tile' in x)
        if not product_tile:
            return None
        
        # Get product name from title attribute
        name = product_tile.get('title', '').strip()
        if not name or len(name) < 3:
            return None
        
        # Find price
        price_element = container.find('span', class_=lambda x: x and 'base-price__regular' in x)
        if not price_element:
            return None
        
        price_text = price_element.get_text(strip=True)
        
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
        
        # Get product URL
        link_element = container.find('a', class_=lambda x: x and 'product-tile__link' in x)
        url = ""
        if link_element and link_element.get('href'):
            href = link_element.get('href')
            url = href if href.startswith('http') else f"https://www.aldi.co.uk{href}"
        
        # Get image URL
        img_element = container.find('img', class_='base-image')
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
                    image_url = f"https://www.aldi.co.uk{image_url}"
        
        # Clean HTML entities from name
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
        
        print("🛒 Starting Aldi browser-based scraping...")
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
    
    def create_summary_report(self, products: List[Product], analysis: Dict) -> str:
        """Create a comprehensive summary report"""
        
        if 'error' in analysis:
            return f"Analysis Error: {analysis['error']}"
        
        report = f"""
ALDI UK PRODUCT SCRAPING REPORT (Browser-Based)
===============================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Products Found: {analysis['total_products']}
Scraping Method: Selenium Browser Automation
Data Quality: ✅ Real-time data with images and URLs

CATEGORY BREAKDOWN:
------------------
"""
        
        for category, data in analysis['categories'].items():
            report += f"{category.upper()}: {data['count']} products\n"
            report += f"  Price range: £{data['min_price']:.2f} - £{data['max_price']:.2f}\n"
            report += f"  Average price: £{data['avg_price']:.2f}\n\n"
        
        report += f"""
PRICE DISTRIBUTION:
------------------
Under £1: {analysis['price_ranges']['under_1']} products
£1 - £5: {analysis['price_ranges']['1_to_5']} products
£5 - £10: {analysis['price_ranges']['5_to_10']} products
Over £10: {analysis['price_ranges']['over_10']} products

BUDGET HIGHLIGHTS:
-----------------
Cheapest: {analysis['cheapest_item']['name']} - £{analysis['cheapest_item']['price']:.2f} ({analysis['cheapest_item']['category']})
Most Expensive: {analysis['most_expensive_item']['name']} - £{analysis['most_expensive_item']['price']:.2f} ({analysis['most_expensive_item']['category']})

TOP 10 BUDGET FINDS:
-------------------
"""
        
        sorted_products = sorted(products, key=lambda x: x.price)[:10]
        for i, product in enumerate(sorted_products, 1):
            report += f"{i:2d}. {product.name} - £{product.price:.2f} ({product.category})\n"
        
        # Show products with complete data
        complete_products = [p for p in products if p.image_url and p.url]
        if complete_products:
            report += f"""

SAMPLE PRODUCTS WITH COMPLETE DATA:
----------------------------------
"""
            for product in complete_products[:5]:
                report += f"• {product.name} - £{product.price:.2f}\n"
                report += f"  Image: {product.image_url}\n"
                report += f"  URL: {product.url}\n\n"
        
        return report

def main():
    """Main function to run the browser-based Aldi scraper"""
    
    print("🛒 Starting Aldi Browser Scraper...")
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
    scraper = AldiBrowserScraper(headless=headless, debug=True)
    
    # Scrape products
    products = scraper.scrape_all_categories(max_categories=max_categories)
    
    if not products:
        print("\n❌ No products found.")
        print("\n💡 Possible issues:")
        print("   1. ChromeDriver not installed or not in PATH")
        print("   2. Aldi's website blocking automated browsers")
        print("   3. Network connectivity issues")
        print("   4. Website structure has changed")
        return
    
    # Analyze the data
    print("\n📊 Analyzing scraped data...")
    analyzer = DataAnalyzer()
    analysis = analyzer.analyze_products(products)
    
    # Save all data locally
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed product data as JSON
    products_filename = f"aldi_products_browser_{timestamp}.json"
    try:
        products_data = [asdict(product) for product in products]
        
        with open(products_filename, 'w', encoding='utf-8') as f:
            json.dump(products_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Product data saved to: {products_filename}")
        
    except Exception as e:
        print(f"❌ Error saving product data: {str(e)}")
    
    # Save analysis data as JSON
    analysis_filename = f"aldi_analysis_browser_{timestamp}.json"
    try:
        with open(analysis_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Analysis data saved to: {analysis_filename}")
        
    except Exception as e:
        print(f"❌ Error saving analysis data: {str(e)}")
    
    # Save and display summary report
    report_filename = f"aldi_report_browser_{timestamp}.txt"
    try:
        summary_report = analyzer.create_summary_report(products, analysis)
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"📝 Summary report saved to: {report_filename}")
        
        # Show preview of the report
        print("\n📄 Report preview:")
        print("-" * 60)
        lines = summary_report.split('\n')[:25]
        print('\n'.join(lines))
        if len(summary_report.split('\n')) > 25:
            print("... (see full report in file)")
        
    except Exception as e:
        print(f"❌ Error saving summary report: {str(e)}")

if __name__ == "__main__":
    main()