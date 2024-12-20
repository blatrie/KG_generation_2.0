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
    #print(chat_completion.choices[0].message.content)
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
    text = """ Predicting how White House policy is going to affect the American economy is always fraught with uncertainty. Donald J. Trump’s return to the White House has taken the doubt up a notch.
        Mr. Trump has proposed or hinted at a range of policies — including drastically higher tariffs, mass deportations, deregulation and a fraught relationship with the Federal Reserve as it sets interest rates — that could shape the economy in complex ways.
        There are two multiplicative sources of uncertainty: One, of course, is what they’re going to do,” said Michael Feroli, the chief U.S. economist at J.P. Morgan. The other is: Even if you know what they’re going to do, what is it going to mean for the economy?
        What forecasters do know is that America’s economy is solid heading into 2025, with low unemployment, solid wage gains, gradually declining Federal Reserve interest rates, and inflation that has been slowly returning to a normal pace after years of rapid price increases. Factory construction took off under the Biden administration, and those facilities will be slowly opening their doors in the coming years.
        But what comes next for growth and for inflation is unclear — especially because when it comes to huge issues like whether or not to assert more control over the Federal Reserve, Mr. Trump is getting different advice from different people in his orbit. Here are some of the crucial wild cards. Tariffs: Likely Inflationary. How Much Is Unclear.
        If economists agree about one thing regarding Mr. Trump’s policies, it is that his tariff proposals could boost prices for consumers and lift inflation. But the range of estimates over how much is wide.
        When Mr. Trump imposed tariffs during his first term, they pushed up prices for consumers, but only slightly.
        But what he is promising now could be more sweeping. Mr. Trump has floated a variety of plans this time, but they have often included across-the-board tariffs and levies of 60 percent or more on goods from China. It’s not at all clear that this is going to be anything like it was the last time around,” said Omair Sharif, founder of Inflation Insights. Fed staff suggested back in 2018 that the central bank could hold steady in the face of price increases coming from tariffs, assuming that consumers and investors expected inflation to remain fairly steady over time. But Jerome H. Powell, the Fed chair, acknowledged last week that this time, we’re in a different situation.”
        Six years ago, inflation had been slow for more than a decade, so a small bump to prices barely registered. This time, inflation has recently been rapid, which could change how price increases filter through the economy. Deportations: Could Slow Growth, but Details Matter.
        Tariffs are not the only thing that economists are struggling to figure out. It is also unclear what immigration policy might look like under a Trump administration, making it difficult to model the possible economic impact.
        Mr. Trump has repeatedly promised the biggest deportation in American history while on the campaign trail, and he has at times hinted at high-skill immigration reform. During an interview on the All In” podcast, he said “what I will do is, you graduate from a college, I think you should get, automatically, as part of your diploma, a green card to be able to stay in this country.”
        But reforming the legal immigration system for highly educated workers would require Congress’s participation, and the campaign barely talked about such plans.
        And when it comes to lower-skill immigration, while there are things the administration can do unilaterally to start deportations, there’s a huge range of estimates around how many people might be successfully removed. It’s hard to round people up, cases might get caught up in the court system and newcomers may replace recent deportees.
        Economists at Goldman Sachs have estimated that a Trump administration might expel anywhere from 300,000 to 2.1 million people in 2025. The low end is based on removal trends from Mr. Trump’s earlier term in office, and the higher end is based on deportation trends from the Eisenhower administration in the 1950s, which Mr. Trump has suggested he would like to emulate.
        Kent Smetters, the faculty director of the Penn Wharton Budget Model, which measures the fiscal impact of public policies, said he was assuming that the administration managed to deport a few hundred thousand people in its first year in office — which he said would have a relatively small effect on either growth or inflation in an economy the size of America’s.
        It’s not as big of an effect as you might think, he said. It’s not the same as if you were getting rid of all undocumented workers, and they’re going to fall far short of that, is my guess.The tariffs Mr. Trump put in effect in 2018 do not offer a good economic precedent for how such a large tariff on goods coming from China in particular might play out, Mr. Sharif said. The earlier rounds heavily affected imports like aluminum, steel and other economic inputs, rather than final products.
        These are not things you go out and buy at Home Depot on the weekend,” he said. The new ones, by contrast, would hit things like T-shirts and tennis shoes, so they could feed much more directly into consumer price inflation."""

    print(merge_triplets(extract_triplets( text )))