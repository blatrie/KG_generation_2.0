import os
from bs4 import BeautifulSoup

class HTMLDirectoryReader:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self):
        documents = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".html") or filename.endswith(".htm"):
                file_path = os.path.join(self.directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'lxml')
                    self.remove_references(soup) 
                    text_content = self.extract_text_from_html(soup)
                    if text_content:  
                        documents.append(text_content)
        return documents

    def remove_references(self, soup):
        references_section = soup.find('h2', id='references')
        if references_section:
            next_sibling = references_section.find_next_sibling()
            references_section.decompose()
            while next_sibling :
                next_sibling_to_decompose = next_sibling
                next_sibling = next_sibling.find_next_sibling()
                next_sibling_to_decompose.decompose()

    def extract_text_from_html(self, soup):
        text = soup.get_text()
        
        return text
    

extract_text_from_html("html_files/Rapport_de_stage_final.html", "output_html.txt")