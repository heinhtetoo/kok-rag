import os
import re
import uuid
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from src.constants import RECIPE_DIR

def scrape_recipe(url: str) -> str | None:
    if "theburmalicious" in url:
        return scrape_theburmalicious_recipe(url)
    else:
        print(f"[ERROR] Scraping not implemented for {url}. Please use a supported recipe URL.")
        return None

def scrape_theburmalicious_recipe(url: str) -> str | None:
    print(f"[INFO] Scraping: {url}...")

    # Fetch the webpage
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"[ERROR] Failed to fetch page. Status code: {response.status_code}")
        return None
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the recipe name
    name_tag = soup.find('h3', class_="ccm-name")
    if name_tag:
        recipe_name = name_tag.get_text(strip=True)
    else:
        parsed_url = urlparse(url)
        slug = parsed_url.path.strip('/').split('/')[-1]
        recipe_name = slug if slug else uuid.uuid4().hex

    # Extract ingredients
    ingredients_div = soup.find('div', class_="ccm-section-ingredients")
    ingredients = []
    if ingredients_div:
        # Target the 'span' inside the 'li' to extract only the text, ignoring checkboxes
        for item in ingredients_div.find_all('li', itemprop="recipeIngredient"):
            span = item.find('span')
            if span:
                ingredients.append(span.get_text(strip=True))

    # Extract instructions
    instructions_div = soup.find('div', class_="ccm-section-instructions")
    instructions = []
    if instructions_div:
        # Target the span inside the instruction list items
        for item in instructions_div.find_all('li', itemprop="recipeInstructions"):
            span = item.find('span')
            if span:
                instructions.append(span.get_text(strip=True))

    # Format the final output text
    recipe_text = f"Title: {recipe_name}\nSource: {url}\n\n"

    recipe_text += "\nINGREDIENTS:\n"
    if ingredients:
        for ing in ingredients:
            recipe_text += f"- {ing}\n"
    else:
        recipe_text += "(Could not parse ingredients)\n"

    recipe_text += "\nINSTRUCTIONS:\n"
    if instructions:
        for i, step in enumerate(instructions, 1):
            recipe_text += f"{i}. {step}\n"
    else:
        recipe_text += "(Could not parse instructions)\n"

    # Save it to data directory
    filename = re.sub(r'[^a-z0-9]', '-', recipe_name.lower()) + ".txt"
    filepath = os.path.join(RECIPE_DIR, filename)

    # Ensure the directory exists
    os.makedirs(RECIPE_DIR, exist_ok=True)

    with open(filepath, "w", encoding='utf-8') as f:
        f.write(recipe_text)

    print(f"[INFO] Successfully saved {recipe_name} to {filepath}")

    return filename