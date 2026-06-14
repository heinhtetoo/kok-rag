"""Web scraping service for recipe extraction."""

import os
import re
import uuid
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.core.logging import get_logger

logger = get_logger(__name__)


def scrape_recipe(url: str, recipe_dir: str) -> str | None:
    """Dispatch to the appropriate scraper based on URL domain.

    Args:
        url: The recipe page URL.
        recipe_dir: Directory to save scraped recipe files.

    Returns:
        Filename of the saved recipe, or ``None`` if scraping failed.
    """
    if "theburmalicious" in url:
        return _scrape_theburmalicious(url, recipe_dir)

    logger.error("Scraping not implemented for %s", url)
    return None


def _scrape_theburmalicious(url: str, recipe_dir: str) -> str | None:
    """Scrape a recipe from theburmalicious.com.

    Args:
        url: Full URL to the recipe page.
        recipe_dir: Directory to save the resulting text file.

    Returns:
        Filename of the saved recipe, or ``None`` on failure.
    """
    logger.info("Scraping: %s", url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        logger.error("Failed to fetch page. Status code: %d", response.status_code)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract recipe name
    name_tag = soup.find("h3", class_="ccm-name")
    if name_tag:
        recipe_name = name_tag.get_text(strip=True)
    else:
        parsed_url = urlparse(url)
        slug = parsed_url.path.strip("/").split("/")[-1]
        recipe_name = slug if slug else uuid.uuid4().hex

    # Extract ingredients
    ingredients: list[str] = []
    ingredients_div = soup.find("div", class_="ccm-section-ingredients")
    if ingredients_div:
        for item in ingredients_div.find_all("li", itemprop="recipeIngredient"):
            span = item.find("span")
            if span:
                ingredients.append(span.get_text(strip=True))

    # Extract instructions
    instructions: list[str] = []
    instructions_div = soup.find("div", class_="ccm-section-instructions")
    if instructions_div:
        for item in instructions_div.find_all("li", itemprop="recipeInstructions"):
            span = item.find("span")
            if span:
                instructions.append(span.get_text(strip=True))

    # Format output
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

    # Save to disk
    filename = re.sub(r"[^a-z0-9]", "-", recipe_name.lower()) + ".txt"
    filepath = os.path.join(recipe_dir, filename)
    os.makedirs(recipe_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(recipe_text)

    logger.info("Successfully saved '%s' to %s", recipe_name, filepath)
    return filename
