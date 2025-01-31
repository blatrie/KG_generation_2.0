import json
from groq import Groq

def count_unique_triplets_from_prompt_output(out):
    """
    Count the number of unique triplets from the output of a prompt.
    
    Args:
        out (str): The output string from the prompt, which may contain JSON triplets.
    
    Returns:
        int: The number of unique triplets.
    """
    # Handle cases where the output is not a clean JSON list
    if out[0] != '[':
        lines = out.split('\n')
        if out[-1] != ']':
            # Extract the JSON substring by removing the first 2 and last 2 lines
            json_string = '\n'.join(lines[2:-2])
        else:
            # Extract the JSON substring by removing the first 2 lines
            json_string = '\n'.join(lines[2:])
        # Remove any surrounding quotes
        out = json_string.strip('"""')
    else:
        # Remove any surrounding quotes
        out = out.strip('"""')

    # Parse the JSON string into a list of triplets
    triplets = json.loads(out)
    
    # Use a set to store unique triplets
    unique_triplets = set()

    # Iterate through each triplet and extract (head, type, tail) as a tuple
    for triplet in triplets:
        triplet_tuple = (triplet['head'], triplet['type'], triplet['tail'])
        unique_triplets.add(triplet_tuple)
    
    # Return the count of unique triplets
    return len(unique_triplets)

def count_unique_triplets_from_triplets_list(triplets):
    """
    Count the number of unique triplets from a list of triplets.
    
    Args:
        triplets (list): A list of triplets.
    
    Returns:
        int: The number of unique triplets.
    """
    unique_triplets = set()
    for triplet in triplets:
        # Extract (head, type, tail) as a tuple and add to the set
        triplet_tuple = (triplet['head'], triplet['type'], triplet['tail'])
        unique_triplets.add(triplet_tuple)
    
    # Return the count of unique triplets
    return len(unique_triplets)

# Initialize the Groq client with the API key
client = Groq(
    api_key='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',
)

# Define the entity types for triplet extraction
entity_types = ['person', 'organisation', 'date', 'place', 'event', 'concept', 'technology']

def extract_triplets(text):
    """
    Extract triplets from the given text using the Groq API.
    
    Args:
        text (str): The input text from which to extract triplets.
    
    Returns:
        list: A list of triplets in JSON format.
    """
    # Send a request to the Groq API to extract triplets
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                    You are an advanced algorithm designed to extract essential and unique knowledge in structured triplet format for building a knowledge graph.
                    Please focus on extracting only informative, non-generic entities and relationships from the provided text.
                    Please extract triplets in **strict JSON format** with the following structure:
                    ```json
                    [
                        {{
                            "head": "<concise entity text, NOT A LONG STATEMENT>",
                            "head_type": "<one of {entity_types}>",
                            "type": "<verb or short phrase  describing the relationship>",
                            "tail": "<concise second entity text, NOT A LONG STATEMENT>",
                            "tail_type": "<one of {entity_types}>"
                        }},
                        ...
                    ]

                    IMPORTANT:
                    - Avoid extracting triplets that convey general knowledge, common facts, or redundant information.
                    - Don't skip important information even when the input is long.
                    - Prioritize unique, specific information relevant to this context, and skip triplets that lack detailed informational value.
                    - The head and tail shouldn be less that 4 words.
                    - Each triplet must have all fields filled with relevant values based on the text.
                    - Avoid any redundant or generic information; focus on unique, specific facts.

                    Here's the input: {text}
                   **Guidelines**:
                    - Only return the JSON list of triplets, without any additional text or commentary.
                    - Do not surround the JSON list with quotes, and ensure it is well-formatted as JSON for parsing.
                    - Do not include any explanations, headers, or extra text before or after the JSON list.
                """
            }
        ],
        model="llama3-70b-8192",
    )
    # Parse the API response into a JSON object
    return json.loads(chat_completion.choices[0].message.content)

def split_article_into_batches(article, n_batches=4):
    """
    Split the article into `n_batches` equal parts.
    
    Args:
        article (str): The input article text.
        n_batches (int): The number of batches to split the article into.
    
    Returns:
        list: A list of article batches.
    """
    # Calculate the size of each batch
    batch_size = len(article) // n_batches
    # Split the article into `n_batches` parts
    return [article[i * batch_size: (i + 1) * batch_size] for i in range(n_batches)]

def merge_triplets(triplets):
    """
    Merge similar triplets using the Groq API.
    
    Args:
        triplets (list): A list of triplets to merge.
    
    Returns:
        list: A list of merged triplets in JSON format.
    """
    # Send a request to the Groq API to merge triplets
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                    You are an advanced algorithm designed to merge triplets knowledge extracted from a text in a structured triplet format for building a knowledge graph.
                    The input triplets are in **strict JSON format** with the following structure:
                    ```json
                    [
                        {{
                            "head": "<entity text>",
                            "head_type": "<one of {entity_types}>",
                            "type": "<verb or short phrase  describing the relationship>",
                            "tail": "<second entity text>",
                            "tail_type": "<one of {entity_types}>"
                        }},
                        ...
                    ]

                    IMPORTANT:
                    - Merge triplets representing similar information.
                    - Pay attention to not merge triplets with same head and tail but different relation type.

                    Here's the input: {triplets}
                   **Guidelines**:
                     - Only return the JSON list of triplets left after merge, without any additional text or commentary.
                     - Do not surround the JSON list with quotes, and ensure it is well-formatted as JSON for parsing.
                     - Do not include any explanations, headers, or extra text before or after the JSON list.
                """
            }
        ],
        model="llama3-70b-8192",
    )
    # Parse the API response into a JSON object
    return json.loads(chat_completion.choices[0].message.content)

