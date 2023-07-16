# Multi-Labeled Text Classifier
Text Based

Follow steps:
1. Clone this repo `git clone https://github.com/rawat999/MultiLabelTextClassification.git`
2. change your directory into `cd MultiLabelTextClassification`
3. Download [dataset](https://drive.google.com/file/d/1slGtHKHYTtiuC98yomV0hP3C85Q5V8sg/view?usp=sharing)
4. Create Virtual Environment with following steps:
    (i) Make sure you have install python3.9
    (ii) Upgrade pip: `python -m pip install --upgrade pip`
    (iii) Check pip version: `py -m pip --version`
    (iv) Install virtualenv package: `py -m pip install --user virtualenv`
    (v) Create virtualenv: `py -m venv env`
    (vi) Activate venv: `.\env\Scripts\activate`
    (vii) Deactivate venv (Optional if not want to use this venv): `.\env\Scripts\activate`
4. Extract the dataset into `data/` folder
5. Install dependencies using `pip install -r requirements.txt`
6. Train MultiClassifier module using command: `python train.py`
7. Evaluate Model: `python evaluation.py -d ./data/valid_data.csv`

### Model Architecture
<img src="./notebooks/model.png" width="100%" alt="model_architecture">


#### Training Loss Curve
<img src="./notebooks/epoch_loss.svg" width="100%" alt="loss_curve">

Ongoing Project...
