# NER
Experimenting with extracting medically relevant entities from PDF documents using a pre-trained NER model.

## Setup

1. Install [Conda](https://docs.conda.io/en/latest/miniconda.html).
2. Create a virtual environment:
    ```sh
    conda create --name ner python=3.9
    conda activate ner
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```


## Test the Application

```sh
export PYTHONPATH=$(pwd)
pytest
```

## Run the Application

```sh
python run.py
curl -X POST "http://0.0.0.0:5001/api/v1/extract" -F "file=@docs/Post-exertional malaise.pdf"

```

This will write a json file to `/tmp/extracted_entities.json`



# Models tested and working (although not performing well)


d4data/biomedical-ner-all
https://huggingface.co/d4data/biomedical-ner-all

GPL/trec-covid-msmarco-distilbert-gpl"
