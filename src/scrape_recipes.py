import os
import requests
from bs4 import BeautifulSoup
import re

def scrape_burmese_recipe(url):
    print(f"Scraping: {url}...")

    # Fetch the webpage
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch page. Status code: {response.status_code}")
        return
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the recipe name
    name_tag = soup.find('h3', class_="ccm-name")
    # Fallback to "Unknown Recipe" if the tag isn't found
    recipe_name = name_tag.get_text(strip=True) if name_tag else "Unknown Recipe"

    # Extract ingredients
    ingredients_div = soup.find('div', class_="ccm-section-ingredients")
    ingredients = []
    if ingredients_div:
        # Find all list items since checkboxes usually use <li> tags inside a <ul>
        for item in ingredients_div.find_all('li'):
            ingredients.append(item.get_text(strip=True))

    # Extract instructions
    instructions_div = soup.find('div', class_="ccm-section-instructions")
    instructions = []
    if instructions_div:
        # Instructions might be in <li> or <p> tags depending on the plugin formatting
        for item in instructions_div.find_all(['li', 'p']):
            text = item.get_text(strip=True)
            if text:
                instructions.append(text)

    # Format the final output text
    recipe_text = f"Title: {recipe_name}\nSource: {url}\n\n"

    recipe_text += "INGREDIENTS:\n"
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
    filepath = os.path.join("data/recipes", filename)

    # Ensure the directory exists
    os.makedirs("data/recipes", exist_ok=True)

    with open(filepath, "w", encoding='utf-8') as f:
        f.write(recipe_text)

    print(f"Successfully saved {recipe_name} to {filepath}")

if __name__ == "__main__":
    test_url = "https://www.theburmalicious.com/blog/laphet"
    scrape_burmese_recipe(test_url)