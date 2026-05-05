import nbformat
import glob

for nb_path in glob.glob("**/*.ipynb", recursive=True):
    with open(nb_path, encoding = "utf-8") as f:
        print(nb_path)
        try:
            nb = nbformat.read(f, as_version=4)
        
        except Exception as e:
            print(e)
            continue
        
    nb['metadata']['kernelspec'] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python"
    }
    with open(nb_path, 'w', encoding = "utf-8") as f:
        nbformat.write(nb, f)

