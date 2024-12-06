import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def triplextract(model, tokenizer, text, entity_types, predicates):

    input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.
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
        """

    message = input_format.format(
                entity_types = json.dumps({"entity_types": entity_types}),
                predicates = json.dumps({"predicates": predicates}),
                text = text)

    messages = [{'role': 'user', 'content': message}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors="pt")#.to(DEVICE)
    output = tokenizer.decode(model.generate(input_ids=input_ids, max_length=2048)[0], skip_special_tokens=True)
    return output

model = AutoModelForCausalLM.from_pretrained("sciphi/triplex", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("sciphi/triplex", trust_remote_code=True)

entity_types = [ "LOCATION", "POSITION", "DATE", "CITY", "COUNTRY", "NUMBER" , "FRUIT"]
predicates = [ "eat", "is in" ]
text = """
San Francisco,[24] officially the City and County of San Francisco, is a commercial, financial, and cultural center in Northern California. 

Steve eats apple every day.
With a population of 808,437 residents as of 2022, San Francisco is the fourth most populous city in the U.S. state of California behind Los Angeles, San Diego, and San Jose.
"""

prediction = triplextract(model, tokenizer, text, entity_types, predicates)
print(prediction)
