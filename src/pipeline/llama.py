import json
from groq import Groq


def count_unique_triplets_from_prompt_output(out):

      if out[0]!= '[' :
          lines = out.split('\n')
          if out[-1] != ']':
            # Récupérer la sous-chaîne à partir de la deuxième ligne depuis le debut & fin
            json_string = '\n'.join(lines[2:-2])
          else:
            json_string = '\n'.join(lines[2:])

          out = json_string.strip('"""')
      else:
          out = out.strip('"""')

      triplets = json.loads(out)
      # Création d'un ensemble pour stocker les triplets uniques
      unique_triplets = set()

      # Itération à travers chaque triplet pour en extraire (head, relation, tail)
      for triplet in triplets:
          triplet_tuple = (triplet['head'], triplet['type'], triplet['tail'])
          unique_triplets.add(triplet_tuple)
      ut = len(unique_triplets)
      # Affichage du nombre de triplets uniques
      # print("Nombre de triplets uniques :", ut)
      return ut

def count_unique_triplets_from_triplets_list(triplets):
      unique_triplets = set()
      for triplet in triplets:
          triplet_tuple = (triplet['head'], triplet['type'], triplet['tail'])
          unique_triplets.add(triplet_tuple)
      ut = len(unique_triplets)
      return ut



# Initialiser le client avec votre clé API
client = Groq(
    api_key='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',
)


entity_types = ['person', 'organisation', 'date', 'place', 'event', 'concept', 'technology']

def extract_triplets(text):
    """Envoie une requête pour extraire des triplets à partir du texte donné."""
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
    # Convertir la réponse en un objet JSON
    return json.loads(chat_completion.choices[0].message.content)


# Diviser l'article en lots (ex. 4 lots) et extraire les triplets pour chaque lot
def split_article_into_batches(article, n_batches=4):
    """Divise l'article en `n_batches` parties égales."""
    batch_size = len(article) // n_batches
    return [article[i * batch_size: (i + 1) * batch_size] for i in range(n_batches)]



def merge_triplets(triplets):
    """Envoie une requête pour extraire des triplets à partir du texte donné."""
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
    # Convertir la réponse en un objet JSON
    return json.loads(chat_completion.choices[0].message.content)

if __name__ == '__main__':
    text = ""
    print(extract_triplets(merge_triplets( text )))