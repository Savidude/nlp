### Project Description
Most of the Restaurants ask reviews to the customers and based on the reviews the restaurant can improve
the customer satisfaction. Thus, reviews plays a vital role for the
successful growth of the restaurant.
The goal of this project is to study, how to analyse the
restaurant’s reviews using Natural Language Processing & Text
Mining techniques such as similarity measures, vectorization, and
Machine Learning

### Setup
Primary operating system that has been used during the development stage - Windows. If you are using other OS the installation process might be
slightly different but the overall workflow will not different too mych. Create environment -> Install libraries from requirements.txt -> launch the file on a Jupiter Notebook

It is quite convenient to install anaconda from https://docs.anaconda.com/anaconda/install/windows/
and manipulate with your virtual env from Anaconda Navigator.

  1. Launch the Anaconda Prompt
  2. Create new environment with the conda command ```conda create --name <name_of_your_env>```
  3. Active your environment ```conda activate <name_of_your_env>```
  4. Also make sure that you are using Python 3.8.13 by executing the following command ```python --version```
  5. After cloning the repository go to the root folder of the project and sure that your ```<name_of_your_env>``` is activated and execute the following command ```pip install -r requirements.txt```
  6. It will take some time so please be patient :)
  7. There are two python scripts that needs to be run in order to generate ```restaurant_reviews_sentistrength.csv``` and ```restaurant_reviews_textblob.csv```. These files already exists inside ```data\derived``` folder.
  8. As an example, you can launch the Jupiter Notebook from conda terminal by executing the following command ```python -m jupiter 1_pearson_correlation.ipynb```. Jupiter Notebook should open on a browser and the code that is related to the task 1 can be found/executing just from the Notebook

### Used libraries
```
absl-py==0.13.0
anyio==3.6.2
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
asttokens==2.1.0
attrs==22.1.0
backcall==0.2.0
backports.entry-points-selectable==1.1.0
beautifulsoup4==4.11.1
bleach==5.0.1
bs4==0.0.1
cffi==1.15.1
charset-normalizer==2.1.1
click==8.0.3
colorama==0.4.5
contourpy==1.0.5
cycler==0.10.0
debugpy==1.6.3
decorator==5.1.1
defusedxml==0.7.1
distlib==0.3.2
entrypoints==0.4
executing==1.2.0
fastjsonschema==2.16.2
filelock==3.0.12
Flask==2.0.2
Flask-SQLAlchemy==2.5.1
fonttools==4.37.4
gensim==4.2.0
greenlet==1.1.2
idna==3.4
importlib-metadata==5.0.0
importlib-resources==5.10.0
ipykernel==6.16.2
ipython==8.5.0
ipython-genutils==0.2.0
ipywidgets==8.0.2
itsdangerous==2.0.1
jedi==0.18.1
Jinja2==3.0.3
joblib==1.2.0
jsonschema==4.16.0
jupyter==1.0.0
jupyter-console==6.4.4
jupyter-server==1.21.0
jupyter_client==7.4.4
jupyter_core==4.11.2
jupyterlab-pygments==0.2.2
jupyterlab-widgets==3.0.3
kiwisolver==1.3.2
MarkupSafe==2.0.1
matplotlib==3.4.3
matplotlib-inline==0.1.6
mediapipe==0.8.9.1
mistune==2.0.4
nbclassic==0.4.7
nbclient==0.7.0
nbconvert==7.2.3
nbformat==5.7.0
nest-asyncio==1.5.6
nltk==3.7
notebook==6.5.2
notebook_shim==0.2.0
numpy==1.23.4
opencv-contrib-python==4.5.3.56
opencv-python==4.5.3.56
packaging==21.3
pandas==1.5.0
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==8.3.1
pkgutil_resolve_name==1.3.10
platformdirs==2.3.0
prometheus-client==0.15.0
prompt-toolkit==3.0.31
protobuf==3.17.3
psutil==5.9.3
ptyprocess==0.7.0
pure-eval==0.2.2
pycparser==2.21
Pygments==2.13.0
pyparsing==2.4.7
pyrsistent==0.18.1
python-dateutil==2.8.2
pytz==2022.4
pyzmq==24.0.1
qtconsole==5.3.2
QtPy==2.2.1
regex==2022.9.13
requests==2.28.1
scikit-learn==1.1.2
scipy==1.9.1
Send2Trash==1.8.0
sentistrength==0.0.9
six==1.16.0
sklearn==0.0
smart-open==6.2.0
sniffio==1.3.0
soupsieve==2.3.2.post1
SQLAlchemy==1.4.29
stack-data==0.6.0
terminado==0.17.0
textblob==0.17.1
threadpoolctl==3.1.0
tinycss2==1.2.1
torchaudio==0.9.0
tornado==6.2
tqdm==4.64.1
traitlets==5.5.0
urllib3==1.26.12
virtualenv==20.7.2
wcwidth==0.2.5
webencodings==0.5.1
websocket-client==1.4.1
Werkzeug==2.0.2
widgetsnbextension==4.0.3
wikipedia==1.4.0
wordcloud==1.8.2.2
zipp==3.10.0
```

### Task files
Every task has been done separately. For example,
```1_pearson_correlation.ipynb```, ```2_cosine_similarity```, ```3_stylistic_classification``` and etc.
You can run any of these files by typing the following command on Anaconda Prompt terminal
```python -m <name_of_the_file>.ipynb```
