mlflow
==============================

Projeto teste da ferramenta MLflow.

Comandos
------------

Comandos de ambiente local:

- Ativa como default o python3 como python no path (zshell):
``echo "export PATH=\"`python3 -m site --user-base`/ bin:\$PATH\"" >> ~/.zshrc``

Comandos do MLflow (execução e treino):

- Executar interface mlflow:
`mlflow ui`

- Buscar nova dependência para inserçao no projeto:
`conda search <NOME_PACOTE>`

- Ativar o ambiente do conda já criado:
`conda activate <NOME_ENV>`

- Criar ambiente (conda) e instalar dependências do projeto:
`conda env create --file conda.yaml -n <NOME_ENV>`

- Inserir/atualizar lib do projeto (inserida no arquivo conda.yaml):
`conda env update --file conda.yaml`

 Opções de execução do projeto na estrutura do MLflow:

- Execução do script de treino diretamente:
`python src/models/train_model.py`

- Execução do script de treino via MLproject:
`mlflow run . -P data_file=src/models/train_model.py`

Essa última opção roda o script indicado no caminho e já instala todas dependências via conda.

Predição dos modelos (após modelo treinado):

- Por meio de script manual:
`python src/models/predict_model.py`

- Por meio da API do mlflow:
`mlflow models predict -m 'runs:/<RUN_ID>/model' -i 'data/processed/casas_X.csv' -t 'csv' -o 'precos2.csv'`

--------

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
