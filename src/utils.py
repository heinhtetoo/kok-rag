def extract_filters_from_query(query: str, ollama_client: any, model: str) -> dict:
    """
    Uses the LLM to parse natural language into metadata filters.

    Args:
        query (str): The user's input query.
        ollama_client (Ollama): The Ollama client for interacting with the LLM.
        model (str): The LLM model to use.

    Returns:
        dict: A dictionary containing the extracted filters.
    """
    import json
    from ollama import Client

    if ollama_client is None or not isinstance(ollama_client, Client):
        print("[ERROR] Ollama client is not initialised.")
        return {"cuisine": None, "dish_type": None}
    elif model is None:
        print("[ERROR] Model name is not provided.")
        return {"cuisine": None, "dish_type": None}

    system_prompt = """
    You are a strict data extraction routing agent. Analyze the user's query and extract the 'cuisine' and 'dish_type' if they are mentioned.
    
    Rules:
    - If a cuisine is mentioned (e.g., Burmese, Italian), set "cuisine" to that value.
    - If a single-word dish type is mentioned (e.g., Soup, Salad, Curry, Noodle), set "dish_type" to that value.
    - If either is missing, set the value to null.
    - You must ONLY output a valid JSON object. Do not add any conversational text.
    
    Example Output:
    {"cuisine": "Burmese", "dish_type": "Soup"}
    """

    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            options={
                "temperature": 0.0 # Ensure deterministic output
            }
        )

        raw_output = response['message']['content'].strip()

        if raw_output.startswith("```json"):
            raw_output = raw_output[len("```json"):].strip()
            if raw_output.endswith("```"):
                raw_output = raw_output[:-len("```")].strip()

        filters = json.loads(raw_output)
        return filters
    
    except Exception as e:
        print(f"[ERROR] Filter extraction failed, defaulting to no filters: {e}")
        return {"cuisine": None, "dish_type": None}