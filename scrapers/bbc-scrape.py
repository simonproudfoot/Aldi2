#!/usr/bin/env python3
"""
BBC Food Recipe Scraper - Browser Automation Version
Uses Selenium to scrape recipes from BBC Food website
Based on the Aldi scraper structure
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import re
from urllib.parse import urljoin, urlparse

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
class Recipe:
    name: str
    description: str
    url: str
    chef: str
    prep_time: str
    cook_time: str
    serves: str
    ingredients: List[str]  # Simple list of ingredients
    method: List[str]  # List of method steps
    dietary_info: List[str]  # Dietary restrictions/info
    category: str  # The search term used to find this recipe
    image_url: str = ""
    programme: str = ""  # TV show if applicable

class BBCFoodScraper:
    def __init__(self, headless=True, debug=True):
        self.debug = debug
        self.headless = headless
        self.driver = None
        self.base_url = "https://www.bbc.co.uk"
        
        # Recipe search terms to try
        self.search_terms = [
            "curry", "pasta", "chicken", "beef", "fish", "vegetarian", "vegan",
            "dessert", "cake", "soup", "salad", "pizza", "bread", "pie",
            "stir-fry", "roast", "grill", "casserole", "rice", "noodles", "chinese", "mexican", "spanish", "thai", "breakfast", "air fryer"
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
    
    def scrape_search_results(self, search_term: str, max_pages: int = 5) -> List[str]:
        """Scrape recipe URLs from search results pages"""
        recipe_urls = []
        
        print(f"\n🔍 Searching for '{search_term}' recipes")
        
        for page in range(1, max_pages + 1):
            print(f"   📖 Scraping page {page}/{max_pages}")
            
            # Construct search URL
            if page == 1:
                url = f"https://www.bbc.co.uk/food/search?q={search_term}"
            else:
                url = f"https://www.bbc.co.uk/food/search?q={search_term}&page={page}"
            
            print(f"      Loading: {url}")
            
            try:
                self.driver.get(url)
                time.sleep(3)  # Wait for page load
                
                # Wait for recipe links to appear
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".promo"))
                    )
                    print(f"      ✅ Search results loaded!")
                except TimeoutException:
                    print(f"      ⏰ Timeout waiting for search results on page {page}")
                    continue
                
                # Get the page source
                html = self.driver.page_source
                
                # Parse recipe URLs from the loaded HTML
                page_urls = self._parse_recipe_urls_from_html(html)
                recipe_urls.extend(page_urls)
                
                print(f"      📦 Found {len(page_urls)} recipe URLs on page {page}")
                
                if not page_urls:  # No more results
                    print(f"      ℹ️  No more results found, stopping at page {page}")
                    break
                
                time.sleep(2)  # Be respectful between requests
                
            except Exception as e:
                print(f"      ❌ Error scraping page {page}: {str(e)}")
                continue
        
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(recipe_urls))
        print(f"   ✅ Found {len(unique_urls)} unique recipe URLs for '{search_term}'")
        return unique_urls
    
    def _parse_recipe_urls_from_html(self, html: str) -> List[str]:
        """Parse recipe URLs from search results HTML"""
        urls = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all recipe links - looking for promo elements with recipe links
            promo_elements = soup.find_all('a', class_='promo')
            
            for promo in promo_elements:
                href = promo.get('href')
                if href and '/food/recipes/' in href:
                    # Ensure full URL
                    if href.startswith('/'):
                        full_url = urljoin(self.base_url, href)
                    else:
                        full_url = href
                    urls.append(full_url)
            
        except Exception as e:
            if self.debug:
                print(f"      ⚠️  Error parsing URLs: {str(e)}")
        
        return urls
    
    def scrape_recipe_details(self, recipe_url: str, category: str) -> Optional[Recipe]:
        """Scrape detailed recipe information from a recipe page"""
        try:
            print(f"      🍳 Scraping recipe: {recipe_url}")
            
            self.driver.get(recipe_url)
            time.sleep(4)  # Increased wait time for page load
            
            # Wait for recipe content to load
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1, [data-testid='main-heading']"))
                )
                
                # Additional wait for dynamic content
                time.sleep(3)
                
            except TimeoutException:
                print(f"         ⏰ Timeout loading recipe page")
                return None
            
            # Get the page source
            html = self.driver.page_source
            
            # Debug: Print part of the HTML to understand structure
            if self.debug:
                soup_debug = BeautifulSoup(html, 'html.parser')
                
                # Check for ingredients containers
                ingredients_testid = soup_debug.find('[data-testid="recipe-ingredients"]')
                method_testid = soup_debug.find('[data-testid="recipe-method"]')
                
                print(f"         🔍 Found ingredients with data-testid: {'YES' if ingredients_testid else 'NO'}")
                print(f"         🔍 Found method with data-testid: {'YES' if method_testid else 'NO'}")
                
                # Look for alternative ingredient containers
                alt_ingredients = soup_debug.find_all(['div', 'section'], class_=lambda x: x and any(term in str(x).lower() for term in ['ingredient', 'recipe']))
                print(f"         🔍 Found alternative ingredient containers: {len(alt_ingredients)}")
                
                # Look for text containing "ingredients"
                ingredients_headers = soup_debug.find_all(string=lambda x: x and 'ingredient' in x.lower())
                print(f"         🔍 Found text containing 'ingredients': {len(ingredients_headers)}")
                
                # Sample some content to see structure
                sample_divs = soup_debug.find_all('div', limit=10)
                for i, div in enumerate(sample_divs):
                    classes = div.get('class', [])
                    if classes:
                        print(f"         📄 Sample div {i}: {' '.join(classes[:2])}")
            
            # Parse recipe details
            recipe = self._parse_recipe_from_html(html, recipe_url, category)
            
            if recipe:
                print(f"         ✅ Successfully scraped: {recipe.name}")
                print(f"         📝 Ingredients found: {len(recipe.ingredients)}")
                print(f"         📋 Method steps found: {len(recipe.method)}")
            else:
                print(f"         ❌ Failed to parse recipe details")
            
            return recipe
            
        except Exception as e:
            print(f"         ❌ Error scraping recipe: {str(e)}")
            return None
    
    def _parse_recipe_from_html(self, html: str, url: str, category: str) -> Optional[Recipe]:
        """Parse recipe details from the recipe page HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Recipe name
            name_element = soup.find('h1', {'data-testid': 'main-heading'}) or soup.find('h1')
            name = name_element.get_text(strip=True) if name_element else "Unknown Recipe"
            
            # Get description
            description = self._extract_description(soup)
            
            # Chef/Author
            chef_element = soup.find('a', href=lambda x: x and '/food/chefs/' in x)
            chef = chef_element.get_text(strip=True) if chef_element else "Unknown Chef"
            
            # Time and serving info
            prep_time = self._extract_time_info(soup, "Prep")
            cook_time = self._extract_time_info(soup, "Cook") 
            serves = self._extract_serve_info(soup)
            
            # Ingredients
            ingredients = self._extract_ingredients(soup)
            
            # Method
            method = self._extract_method(soup)
            
            # Dietary info
            dietary_info = self._extract_dietary_info(soup)
            
            # Image URL
            image_url = self._extract_image_url(soup)
            
            # Programme info
            programme = self._extract_programme_info(soup)
            
            return Recipe(
                name=name,
                description=description,
                url=url,
                chef=chef,
                prep_time=prep_time,
                cook_time=cook_time,
                serves=serves,
                ingredients=ingredients,
                method=method,
                dietary_info=dietary_info,
                category=category,
                image_url=image_url,
                programme=programme
            )
            
        except Exception as e:
            if self.debug:
                print(f"         ⚠️  Error parsing recipe HTML: {str(e)}")
            return None
    
    def _extract_description(self, soup) -> str:
        """Extract recipe description"""
        try:
            # Try multiple selectors for description
            desc_element = (
                soup.find('[data-testid="recipe-description"]') or
                soup.find('p', class_=lambda x: x and 'description' in x.lower()) or
                soup.find('div', class_=lambda x: x and 'description' in x.lower())
            )
            
            if desc_element:
                return desc_element.get_text(strip=True)
            
            # Fallback: look for the first substantial paragraph
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 50 and len(text) < 500:  # Reasonable description length
                    return text
            
            return ""
        except:
            return ""
    
    def _extract_time_info(self, soup, time_type: str) -> str:
        """Extract prep or cook time information"""
        try:
            # Look for dt/dd pairs
            dt_elements = soup.find_all('dt')
            for dt in dt_elements:
                if time_type.lower() in dt.get_text().lower():
                    dd = dt.find_next_sibling('dd')
                    if dd:
                        return dd.get_text(strip=True)
            return "Not specified"
        except:
            return "Not specified"
    
    def _extract_serve_info(self, soup) -> str:
        """Extract serving information"""
        try:
            # Look for serve/serves information
            dt_elements = soup.find_all('dt')
            for dt in dt_elements:
                if 'serve' in dt.get_text().lower():
                    dd = dt.find_next_sibling('dd')
                    if dd:
                        return dd.get_text(strip=True)
            return "Not specified"
        except:
            return "Not specified"
    
    def _extract_ingredients(self, soup) -> List[str]:
        """Extract ingredients with multiple fallback strategies"""
        ingredients = []
        
        try:
            # Strategy 1: Look for data-testid="recipe-ingredients"
            ingredients_container = soup.find('[data-testid="recipe-ingredients"]')
            
            if ingredients_container:
                if self.debug:
                    print(f"         ✅ Found ingredients container with data-testid")
                
                # Extract from ALL ul elements within the ingredients container
                all_lists = ingredients_container.find_all('ul')
                
                for ul in all_lists:
                    # Look for the specific BBC structure
                    if 'ssrcss-1ynsflq-UnorderedList' in ul.get('class', []):
                        list_items = ul.find_all('li', class_=lambda x: x and 'ssrcss-131sxq9-Stack' in x)
                        
                        for item in list_items:
                            ingredient_text = item.get_text(strip=True)
                            if ingredient_text and len(ingredient_text) > 2:
                                ingredients.append(ingredient_text)
                    else:
                        # Fallback: any li within ul in ingredients container
                        list_items = ul.find_all('li')
                for item in list_items:
                    ingredient_text = item.get_text(strip=True)
                            if ingredient_text and len(ingredient_text) > 2:
                        ingredients.append(ingredient_text)
            
                if ingredients:
                    if self.debug:
                        print(f"         📝 Strategy 1 success: {len(ingredients)} ingredients")
                        print(f"         📝 Sample: {ingredients[0][:50]}...")
                    return ingredients
            
            # Strategy 2: Look for headings containing "ingredients" followed by lists
            if self.debug:
                print(f"         🔍 Trying strategy 2: Looking for ingredient headings")
            
            # Find headings that contain "ingredient"
            ingredient_headings = soup.find_all(['h1', 'h2', 'h3', 'h4'], 
                                               string=lambda x: x and 'ingredient' in x.lower())
            
            for heading in ingredient_headings:
                if self.debug:
                    print(f"         🔍 Found heading: {heading.get_text()}")
                
                # Look for all lists after this heading until we find another major heading
                current_element = heading.find_next_sibling()
                while current_element:
                    if current_element.name in ['h1', 'h2'] and 'method' in current_element.get_text().lower():
                        break  # Stop at method section
                    
                    if current_element.name == 'ul' or (current_element.name == 'div' and current_element.find('ul')):
                        # Found a list, extract ingredients
                        if current_element.name == 'ul':
                            target_ul = current_element
                        else:
                            target_ul = current_element.find('ul')
                        
                        if target_ul:
                            items = target_ul.find_all('li')
                            for item in items:
                                text = item.get_text(strip=True)
                                if text and len(text) > 2:
                                    ingredients.append(text)
                    
                    # Look for section headers and continue collecting
                    elif current_element.name == 'h3':
                        section_text = current_element.get_text().lower()
                        if any(word in section_text for word in ['for the', 'sauce', 'marinade', 'garnish', 'topping']):
                            # This is a sub-section, continue
                            pass
                        else:
                            # This might be a different section, stop
                            break
                    
                    current_element = current_element.find_next_sibling()
                
                if ingredients:
                    if self.debug:
                        print(f"         📝 Strategy 2 success: {len(ingredients)} ingredients")
                    return ingredients
            
            # Strategy 3: Look for lists that seem like ingredients anywhere on page
            if self.debug:
                print(f"         🔍 Trying strategy 3: Analyzing all lists")
            
                all_lists = soup.find_all('ul')
            ingredient_candidates = []
            
                for ul in all_lists:
                    items = ul.find_all('li')
                if len(items) >= 3:  # Likely an ingredient list
                    potential_ingredients = []
                        for item in items:
                            text = item.get_text(strip=True)
                        # Filter out navigation and other non-ingredient items
                        if (text and len(text) > 5 and len(text) < 300 and
                            not any(skip in text.lower() for skip in 
                                   ['home', 'news', 'sport', 'weather', 'iplayer', 'sounds', 'cbbc', 'cbeebies', 
                                    'food', 'bitesize', 'arts', 'taster', 'local', 'tv', 'radio', 'menu',
                                    'search', 'sign in', 'more', 'back to', 'skip to', 'accessibility', 
                                    'terms', 'privacy', 'cookies', 'contact', 'help']) and
                            # Look for ingredient-like patterns
                            (any(measure in text.lower() for measure in 
                                ['tbsp', 'tsp', 'cup', 'ml', 'fl oz', 'oz', 'g', 'kg', 'lb', 'litre', 'pint']) or
                             any(food in text.lower() for food in 
                                ['onion', 'garlic', 'oil', 'salt', 'pepper', 'flour', 'butter', 'egg', 'milk', 
                                 'sugar', 'tomato', 'chicken', 'beef', 'fish', 'cheese', 'lemon', 'lime']))):
                            potential_ingredients.append(text)
                    
                    # If this list has a good ratio of ingredient-like items, use it
                    if len(potential_ingredients) >= max(3, len(items) * 0.6):
                        ingredient_candidates.extend(potential_ingredients)
            
            if ingredient_candidates:
                ingredients = ingredient_candidates[:30]  # Reasonable limit
                if self.debug:
                    print(f"         📝 Strategy 3 success: {len(ingredients)} ingredients")
                return ingredients
            
            # Strategy 4: Pattern matching in text
            if self.debug:
                print(f"         🔍 Trying strategy 4: Pattern matching")
                
            # Look for text patterns that look like ingredients
            all_text = soup.get_text()
            lines = all_text.split('\n')
            
            for line in lines:
                line = line.strip()
                # Look for lines that start with numbers/amounts and contain food words
                if (line and len(line) > 10 and len(line) < 200 and
                    (line[0].isdigit() or line.startswith(('½', '¼', '¾', '⅓', '⅔'))) and
                    any(food_word in line.lower() for food_word in 
                       ['tbsp', 'tsp', 'cup', 'ml', 'fl oz', 'oz', 'g', 'kg', 'lb', 'clove', 'onion', 'garlic'])):
                    ingredients.append(line)
            
            # Take only reasonable number of ingredients
            ingredients = ingredients[:25] if ingredients else []
            
            if self.debug:
                print(f"         📝 Final ingredient count: {len(ingredients)}")
                if ingredients:
                    print(f"         📝 Sample ingredient: {ingredients[0][:50]}...")
            
        except Exception as e:
            if self.debug:
                print(f"         ⚠️  Error extracting ingredients: {str(e)}")
        
        return ingredients
    
    def _extract_method(self, soup) -> List[str]:
        """Extract cooking method steps with multiple fallback strategies"""
        method_steps = []
        
        try:
            # Strategy 1: Look for data-testid="recipe-method"
            method_container = soup.find('[data-testid="recipe-method"]')
            
            if method_container:
                if self.debug:
                    print(f"         ✅ Found method container with data-testid")
                
                # Look for ordered lists within the method container
                ordered_lists = method_container.find_all('ol')
                
                for ol in ordered_lists:
                    # Look for the specific BBC structure
                    if 'ssrcss-1o787j8-OrderedList' in ol.get('class', []):
                        list_items = ol.find_all('li', class_=lambda x: x and 'ssrcss-131sxq9-Stack' in x)
                        
                        for item in list_items:
                            # Look for the text container within each list item
                            text_container = item.find('div', class_=lambda x: x and 'ssrcss-k5ghct-ListItemText' in x)
                            if text_container:
                                step_text = text_container.get_text(strip=True)
                                if step_text and len(step_text) > 10:
                                    method_steps.append(step_text)
                            else:
                                # Fallback: get text directly from list item
                                step_text = item.get_text(strip=True)
                                if step_text and len(step_text) > 10:
                                    method_steps.append(step_text)
                    else:
                        # Fallback: any li within ol in method container
                        list_items = ol.find_all('li')
                        for item in list_items:
                            step_text = item.get_text(strip=True)
                            if step_text and len(step_text) > 10:
                        method_steps.append(step_text)
            
                if method_steps:
                    if self.debug:
                        print(f"         📋 Strategy 1 success: {len(method_steps)} method steps")
                        print(f"         📋 Sample: {method_steps[0][:50]}...")
                    return method_steps
            
            # Strategy 2: Look for headings containing "method" followed by lists
            if self.debug:
                print(f"         🔍 Trying strategy 2: Looking for method headings")
            
            method_headings = soup.find_all(['h1', 'h2', 'h3', 'h4'], 
                                          string=lambda x: x and 'method' in x.lower())
            
            for heading in method_headings:
                if self.debug:
                    print(f"         🔍 Found method heading: {heading.get_text()}")
                
                # Look for ordered lists after this heading
                current_element = heading.find_next_sibling()
                while current_element:
                    if current_element.name in ['h1', 'h2'] and current_element.get_text().lower() != heading.get_text().lower():
                        break  # Stop at next major section
                    
                    if current_element.name == 'ol' or (current_element.name == 'div' and current_element.find('ol')):
                        if current_element.name == 'ol':
                            target_ol = current_element
                        else:
                            target_ol = current_element.find('ol')
                        
                        if target_ol:
                            items = target_ol.find_all('li')
                            for item in items:
                                text = item.get_text(strip=True)
                            if text and len(text) > 10:
                                method_steps.append(text)
                            break
                    
                    current_element = current_element.find_next_sibling()
                
                if method_steps:
                    if self.debug:
                        print(f"         📋 Strategy 2 success: {len(method_steps)} method steps")
                    return method_steps
            
            # Strategy 3: Look for any ordered lists that seem like methods
            if self.debug:
                print(f"         🔍 Trying strategy 3: Analyzing ordered lists")
            
            all_ordered_lists = soup.find_all('ol')
            for ol in all_ordered_lists:
                items = ol.find_all('li')
                if len(items) >= 2:  # Likely a method list
                    potential_steps = []
                    cooking_words = ['heat', 'add', 'cook', 'stir', 'mix', 'place', 'pour', 'serve', 'season', 'chop', 'slice', 'fry', 'boil', 'simmer', 'bake', 'roast']
                    
                    for item in items:
                        text = item.get_text(strip=True)
                        # Check for cooking-like content
                        if (text and len(text) > 20 and len(text) < 2000 and
                            any(cooking_word in text.lower() for cooking_word in cooking_words)):
                            potential_steps.append(text)
                    
                    # If most items look like cooking steps, use this list
                    if len(potential_steps) >= max(2, len(items) * 0.7):
                        method_steps = potential_steps
                        break
                
            if method_steps:
                if self.debug:
                    print(f"         📋 Strategy 3 success: {len(method_steps)} method steps")
                return method_steps
            
            # Strategy 4: Look for numbered paragraphs or method-like text
            if self.debug:
                print(f"         🔍 Trying strategy 4: Looking for numbered steps")
                
            # Look for paragraphs that might be method steps
                    paragraphs = soup.find_all('p')
            numbered_steps = []
            
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                # Look for text that starts with numbers or contains cooking verbs
                if (text and len(text) > 30 and len(text) < 1000 and
                            (text[0].isdigit() or 
                     any(word in text.lower()[:100] for word in 
                        ['heat the', 'add the', 'cook for', 'stir in', 'place the', 'mix the', 'pour in',
                         'meanwhile', 'for the', 'preheat', 'season with']))):
                    numbered_steps.append(text)
                    
            if numbered_steps:
                method_steps = numbered_steps[:20]  # Limit to reasonable number
                if self.debug:
                    print(f"         📋 Strategy 4 success: {len(method_steps)} method steps")
            
            if self.debug:
                print(f"         📋 Final method steps count: {len(method_steps)}")
                if method_steps:
                    print(f"         📋 Sample step: {method_steps[0][:50]}...")
            
        except Exception as e:
            if self.debug:
                print(f"         ⚠️  Error extracting method: {str(e)}")
        
        return method_steps
    
    def _extract_dietary_info(self, soup) -> List[str]:
        """Extract dietary information"""
        dietary_info = []
        
        try:
            # Look for dietary information in dt/dd pairs
            dt_elements = soup.find_all('dt')
            for dt in dt_elements:
                if 'dietary' in dt.get_text().lower():
                    dd = dt.find_next_sibling('dd')
                    if dd:
                        # Find all links in the dd (dietary categories)
                        links = dd.find_all('a')
                        for link in links:
                            dietary_info.append(link.get_text(strip=True))
            
        except Exception as e:
            if self.debug:
                print(f"         ⚠️  Error extracting dietary info: {str(e)}")
        
        return dietary_info
    
    def _extract_image_url(self, soup) -> str:
        """Extract recipe image URL"""
        try:
            # Look for the holding_image class
            img_element = soup.find('img', class_='holding_image')
            if img_element and img_element.get('src'):
                return img_element['src']
            return ""
        except:
            return ""
    
    def _extract_programme_info(self, soup) -> str:
        """Extract TV programme information if available"""
        try:
            # Look for programme links
            programme_link = soup.find('a', href=lambda x: x and '/food/programmes/' in x)
            if programme_link:
                return programme_link.get_text(strip=True)
            return ""
        except:
            return ""
    
    def scrape_all_categories(self, max_categories: int = None, max_pages: int = 5) -> List[Recipe]:
        """Scrape recipes from multiple search categories"""
        all_recipes = []
        
        print("🍳 Starting BBC Food recipe scraping...")
        print("=" * 60)
        
        if not self.setup_driver():
            return all_recipes
        
        try:
            # Use all search terms if max_categories is None
            search_terms_to_try = self.search_terms if max_categories is None else self.search_terms[:max_categories]
            
            print(f"📋 Will scrape {len(search_terms_to_try)} categories, {max_pages} pages each")
            
            for search_term in search_terms_to_try:
                print(f"\n🎯 Processing category: {search_term}")
                
                # Get recipe URLs from search results
                recipe_urls = self.scrape_search_results(search_term, max_pages)
                
                if not recipe_urls:
                    print(f"   ❌ No recipe URLs found for '{search_term}'")
                    continue
                
                print(f"   📥 Found {len(recipe_urls)} recipe URLs, now scraping details...")
                
                # Scrape details for each recipe
                category_recipes = []
                for i, url in enumerate(recipe_urls, 1):
                    print(f"   📖 Recipe {i}/{len(recipe_urls)}")
                    
                    recipe = self.scrape_recipe_details(url, search_term)
                    if recipe:
                        category_recipes.append(recipe)
                    
                    # Small delay between recipe requests
                    time.sleep(2)
                
                all_recipes.extend(category_recipes)
                print(f"   ✅ Successfully scraped {len(category_recipes)} recipes for '{search_term}'")
                
                # Delay between different categories
                time.sleep(3)
            
            print(f"\n🎉 Scraping complete!")
            print(f"   Total recipes scraped: {len(all_recipes)}")
            print(f"   Categories processed: {len(search_terms_to_try)}")
            
            return all_recipes
            
        finally:
            # Always close the browser
            if self.driver:
                print("\n🔒 Closing browser...")
                self.driver.quit()

class RecipeAnalyzer:
    def analyze_recipes(self, recipes: List[Recipe]) -> Dict:
        """Analyze the scraped recipes"""
        if not recipes:
            return {
                'total_recipes': 0,
                'categories': {},
                'ingredients': {},
                'dietary_info': {},
                'cooking_times': {},
                'serving_sizes': {},
                'recipes': []
            }
        
        analysis = {
            'total_recipes': len(recipes),
            'categories': {},
            'ingredients': {},
            'dietary_info': {},
            'cooking_times': {},
            'serving_sizes': {},
            'recipes': []
        }
        
        # Process each recipe
        for recipe in recipes:
            # Add recipe to list with all its data
            recipe_data = {
                'name': recipe.name,
                'description': recipe.description,
                'url': recipe.url,
                'chef': recipe.chef,
                'prep_time': recipe.prep_time,
                'cook_time': recipe.cook_time,
                'serves': recipe.serves,
                'ingredients': recipe.ingredients,
                'method': recipe.method,
                'dietary_info': recipe.dietary_info,
                'category': recipe.category,  # Include the search term
                'image_url': recipe.image_url,
                'programme': recipe.programme
            }
            analysis['recipes'].append(recipe_data)
            
            # Count categories
            analysis['categories'][recipe.category] = analysis['categories'].get(recipe.category, 0) + 1
        
            # Count ingredients
            for ingredient in recipe.ingredients:
                analysis['ingredients'][ingredient] = analysis['ingredients'].get(ingredient, 0) + 1
            
            # Count dietary info
            for dietary in recipe.dietary_info:
                analysis['dietary_info'][dietary] = analysis['dietary_info'].get(dietary, 0) + 1
            
            # Count cooking times
            if recipe.prep_time != "Not specified":
                analysis['cooking_times']['prep'] = analysis['cooking_times'].get('prep', 0) + 1
            if recipe.cook_time != "Not specified":
                analysis['cooking_times']['cook'] = analysis['cooking_times'].get('cook', 0) + 1
            
            # Count serving sizes
            if recipe.serves != "Not specified":
                analysis['serving_sizes'][recipe.serves] = analysis['serving_sizes'].get(recipe.serves, 0) + 1
        
        return analysis
    
    def create_summary_report(self, recipes: List[Recipe], analysis: Dict) -> str:
        """Create a summary report of the recipe analysis"""
        if not recipes:
            return "No recipes found to analyze."
        
        report = []
        report.append(f"Recipe Analysis Summary")
        report.append(f"=====================")
        report.append(f"Total Recipes: {analysis['total_recipes']}")
        
        # Categories summary
        report.append("\nCategories:")
        for category, count in sorted(analysis['categories'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {category}: {count} recipes")
        
        # Top ingredients
        report.append("\nTop 10 Most Common Ingredients:")
        for ingredient, count in sorted(analysis['ingredients'].items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"- {ingredient}: {count} recipes")
        
        # Dietary info
        report.append("\nDietary Information:")
        for dietary, count in sorted(analysis['dietary_info'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {dietary}: {count} recipes")
        
        # Cooking times
        report.append("\nCooking Time Information:")
        report.append(f"- Recipes with prep time: {analysis['cooking_times'].get('prep', 0)}")
        report.append(f"- Recipes with cook time: {analysis['cooking_times'].get('cook', 0)}")
        
        # Serving sizes
        report.append("\nCommon Serving Sizes:")
        for size, count in sorted(analysis['serving_sizes'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {size}: {count} recipes")
        
        return "\n".join(report)

def main():
    """Main function to run the scraper"""
    print("BBC Food Recipe Scraper")
    print("======================")
    
    # Get user input for number of categories
    max_categories_input = input("How many recipe categories to scrape? (press Enter for all 20): ").strip()
    max_categories = int(max_categories_input) if max_categories_input else None
    
    # Get user input for number of pages
    max_pages_input = input("How many pages to scrape per category? (press Enter for 7): ").strip()
    max_pages = int(max_pages_input) if max_pages_input else 7
    
    # Initialize scraper
    scraper = BBCFoodScraper(headless=True)
    
    try:
    # Scrape recipes
        print("\nStarting recipe scraping...")
    recipes = scraper.scrape_all_categories(max_categories=max_categories, max_pages=max_pages)
    
    if not recipes:
            print("No recipes found!")
        return
    
        # Save recipes to JSON
        output_file = 'bbc_recipes.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([{
                'name': r.name,
                'description': r.description,
                'url': r.url,
                'chef': r.chef,
                'prep_time': r.prep_time,
                'cook_time': r.cook_time,
                'serves': r.serves,
                'ingredients': r.ingredients,
                'method': r.method,
                'dietary_info': r.dietary_info,
                'category': r.category,
                'image_url': r.image_url,
                'programme': r.programme
            } for r in recipes], f, indent=2, ensure_ascii=False)
        
        print(f"\nScraped {len(recipes)} recipes")
        print(f"Results saved to {output_file}")
    
        # Analyze recipes
        analyzer = RecipeAnalyzer()
        analysis = analyzer.analyze_recipes(recipes)
        report = analyzer.create_summary_report(recipes, analysis)
        print("\nAnalysis Report:")
        print(report)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()