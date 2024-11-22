import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

class XMLParentMap:
    def __init__(self, tree):
        self.tree = tree
        self.parent_map = self.build_parent_map()

    def build_parent_map(self):
        parent_map = {}
        for p in self.tree.iter():
            for c in p:
                if c in parent_map:
                    parent_map[c].append(p)
                else:
                    parent_map[c] = [p]
        return parent_map

    def get_all_parents(self, element):
        all_parents = []
        self._collect_all_parents(element, all_parents)
        return all_parents

    def _collect_all_parents(self, element, all_parents):
        if element in self.parent_map:
            parents = self.parent_map[element]
            for parent in parents:
                all_parents.append(parent)
                self._collect_all_parents(parent, all_parents)

class XMLDirectoryReader:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self):
        documents = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".xml"):
                file_path = os.path.join(self.directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    tree = ET.parse(file)
                    root = tree.getroot()
                    parent_map_instance = XMLParentMap(tree)
                    text_content = self.extract_text_from_xml(root, parent_map_instance)
                    if text_content:  # Checks that the text is not empty
                        documents.append(text_content)
        return documents

    def extract_text_from_xml(self, root, parent_map):
        namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
        texts = []
        for elem in root.iter():
            all_parents = parent_map.get_all_parents(elem)

            # Check whether the element or one of its parents is in <biblStruct>
            if any(p.tag in ['{http://www.tei-c.org/ns/1.0}biblStruct', '{http://www.tei-c.org/ns/1.0}figure'] for p in all_parents):
                continue  # Skip to next element if in <biblStruct>


            # Exclude certain specific elements (such as metadata or publication information)
            if elem.tag in ['{http://www.tei-c.org/ns/1.0}title', '{http://www.tei-c.org/ns/1.0}head', '{http://www.tei-c.org/ns/1.0}p']:
                if elem.text and elem.text.strip():  # Exclude elements with empty text
                    text = elem.text.strip()
                    if elem.tag == '{http://www.tei-c.org/ns/1.0}title':
                        text = f"\n\ntitle : {text}"  # Add the "title :" prefix
                    elif elem.tag == '{http://www.tei-c.org/ns/1.0}head':
                        text = f"\nhead : {text}"  # Add the "head :" prefix
                    texts.append(text)
        return "\n".join(texts)
    
documents = XMLDirectoryReader("xml_files").load_data()

with open("output_xml.txt", "w", encoding="utf-8") as f:
    f.write(documents[0])