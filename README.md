# Multi-Labeled Text Classifier
NLP and Deep Neural Network

Follow steps:
<ol>
    <li> Clone this repo `git clone https://github.com/rawat999/MultiLabelTextClassification.git` </li>
    <li> change your directory into `cd MultiLabelTextClassification` </li>
    <li> Download [dataset] (https://drive.google.com/file/d/1slGtHKHYTtiuC98yomV0hP3C85Q5V8sg/view?usp=sharing) </li>
<li> Create Virtual Environment with following steps:

<ul>
    <li> Make sure you have install python3.9</li>
    <li> Upgrade pip: `python -m pip install --upgrade pip`</li>
    <li> Check pip version: `py -m pip --version`</li>
    <li> Install virtualenv package: `py -m pip install --user virtualenv`</li>
    <li> Create virtualenv: `py -m venv env`</li>
    <li> Activate venv: `.\env\Scripts\activate`</li>
    <li> Deactivate venv (Optional if not want to use this venv): `.\env\Scripts\activate`</li>
</ul></li>
<li> Extract the dataset into `data/` folder </li>
<li> Install dependencies using `pip install -r requirements.txt` </li>
<li> Train MultiClassifier module using command: `python train.py` </li>
<li> Evaluate Model: `python evaluation.py -d ./data/valid_data.csv` </li>
</ol>

### Model Architecture
<img src="./notebooks/model.png" width="100%" alt="model_architecture">


#### Training Loss Curve
<img src="./notebooks/epoch_loss.svg" width="100%" alt="loss_curve">

Ongoing Project...
