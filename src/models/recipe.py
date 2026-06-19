"""Recipe data model for the Kök RAG pipeline."""

from dataclasses import dataclass, field


@dataclass
class Recipe:
    """Canonical representation of a recipe."""

    title: str
    source_url: str
    ingredients: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    cuisine: str = "Unknown"
    dish_type: str = "Unknown"
    raw_text: str = ""

    def to_metadata(self, parent_id: str, section: str = "unknown") -> dict:
        """Convert to a metadata dictionary for ChromaDB."""
        return {
            "parent_id": parent_id,
            "source": self.source_url,
            "cuisine": self.cuisine,
            "dish_type": self.dish_type,
            "title": self.title,
            "section": section,
            "ingredient_list": ", ".join([ing.lower() for ing in self.ingredients]),
        }
