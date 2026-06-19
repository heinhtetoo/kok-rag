"""Web scraping service for recipe extraction."""

import json

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.logging import get_logger
from src.models.recipe import Recipe

logger = get_logger(__name__)


def scrape_recipe(url: str, cuisine: str = "Unknown", dish_type: str = "Unknown") -> Recipe | None:
    """Dispatch to the appropriate scraper based on URL domain.

    Args:
        url: The recipe page URL.
        cuisine: Cuisine type for the recipe.
        dish_type: Dish type for the recipe.

    Returns:
        A Recipe object or None if scraping failed.
    """
    if "theburmalicious" in url:
        return _scrape_theburmalicious(url, cuisine, dish_type)

    logger.info("Using generic JSON-LD scraper for %s", url)
    return _scrape_jsonld_generic(url, cuisine, dish_type)


def _format_recipe_text(recipe: Recipe) -> str:
    """Format a recipe into plain text for the parent store."""
    recipe_text = f"Title: {recipe.title}\nSource: {recipe.source_url}\n\n"

    recipe_text += "INGREDIENTS:\n"
    if recipe.ingredients:
        for ing in recipe.ingredients:
            recipe_text += f"- {ing}\n"
    else:
        recipe_text += "(Could not parse ingredients)\n"

    recipe_text += "\nINSTRUCTIONS:\n"
    if recipe.instructions:
        for i, step in enumerate(recipe.instructions, 1):
            recipe_text += f"{i}. {step}\n"
    else:
        recipe_text += "(Could not parse instructions)\n"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _scrape_theburmalicious(url: str, cuisine: str, dish_type: str) -> Recipe | None:
    """Scrape a recipe from theburmalicious.com."""
    logger.info("Scraping: %s", url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error("Failed to fetch page. Error: %s", e)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract recipe name
    name_tag = soup.find("h3", class_="ccm-name")
    recipe_name = name_tag.get_text(strip=True) if name_tag else "Unknown Recipe"

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

    recipe = Recipe(
        title=recipe_name,
        source_url=url,
        ingredients=ingredients,
        instructions=instructions,
        cuisine=cuisine,
        dish_type=dish_type,
    )
    recipe.raw_text = _format_recipe_text(recipe)
    return recipe


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _scrape_jsonld_generic(url: str, cuisine: str, dish_type: str) -> Recipe | None:
    """Fallback scraper parsing schema.org/Recipe JSON-LD."""
    logger.info("Scraping generic JSON-LD: %s", url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error("Failed to fetch page. Error: %s", e)
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    jsonld_tags = soup.find_all("script", type="application/ld+json")

    recipe_data = None
    for tag in jsonld_tags:
        try:
            data = json.loads(tag.string)
            if isinstance(data, dict):
                if data.get("@type") == "Recipe" or (
                    isinstance(data.get("@type"), list) and "Recipe" in data.get("@type")
                ):
                    recipe_data = data
                    break
            elif isinstance(data, list):
                for item in data:
                    if item.get("@type") == "Recipe":
                        recipe_data = item
                        break
            if recipe_data:
                break
        except Exception:
            continue

    if not recipe_data:
        logger.error("No Recipe JSON-LD found on page.")
        return None

    title = recipe_data.get("name", "Unknown Recipe")

    # Parse ingredients
    ingredients = recipe_data.get("recipeIngredient", [])
    if not isinstance(ingredients, list):
        ingredients = [ingredients] if ingredients else []

    # Parse instructions
    instructions_raw = recipe_data.get("recipeInstructions", [])
    instructions = []
    if isinstance(instructions_raw, list):
        for step in instructions_raw:
            if isinstance(step, dict) and "text" in step:
                instructions.append(step["text"])
            elif isinstance(step, str):
                instructions.append(step)
    elif isinstance(instructions_raw, str):
        instructions.append(instructions_raw)

    recipe = Recipe(
        title=title,
        source_url=url,
        ingredients=ingredients,
        instructions=instructions,
        cuisine=cuisine,
        dish_type=dish_type,
    )
    recipe.raw_text = _format_recipe_text(recipe)
    return recipe
