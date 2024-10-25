import os
from params import PATH_TO_PDF_FILES, PATH_TO_RDF_FILES

def get_files(path):
    """
    Get a list of files with the ".pdf" extension in the specified path.

    Args:
        path (str): The path to the directory containing the files.

    Returns:
        list: A list of file names with the ".pdf" extension.
    """
    files = []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            files.append(file)
    return files

def read_ttl(fname) :
    """
    Read and parse an RDF file in Turtle format.

    Parameters:
    fname (str): The filename of the RDF file to be read.

    Returns:
    None
    """
    # Create an RDF graph
    g = Graph()

    # Parse the RDF data
    g.parse(PATH_TO_RDF_FILES+fname, format="n3")

    # Iterate through triples and print human-readable data
    for subject, predicate, obj in g:
        subject_str = str(subject).split('#')[-1]  # Extract the part after the last '#' character
        predicate_str = str(predicate).split('#')[-1]  # Extract the part after the last '#' character
        obj_str = str(obj)
        
        print(f"Subject: {subject_str}")
        print(f"Predicate: {predicate_str}")
        print(f"Object: {obj_str}\n")
        

if __name__ == "__main__" :
    read_ttl("test_20Go.ttl")