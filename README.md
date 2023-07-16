## Multi-Labeled Text Classifier
#### Natural Language Processing and Deep Neural Network

### Setting Up this Project on system:
1. Clone this repo `git clone https://github.com/rawat999/MultiLabelTextClassification.git`
2. Change your directory into `cd MultiLabelTextClassification` 
3. Create Virtual Environment with following steps:
    - Make sure you have installed python3.9
    - Upgrade pip: `python -m pip install --upgrade pip`
    - Check pip version: `python -m pip --version`
    - Install virtualenv package: `python -m pip install --user virtualenv`
    - Create virtualenv: `python -m venv env`
    - Activate venv: `.\env\Scripts\activate`
    - Deactivate venv (Optional if not want to use this venv): `deactivate`
4. Install dependencies using `pip install -r requirements.txt`
5. Download [dataset](https://drive.google.com/file/d/1slGtHKHYTtiuC98yomV0hP3C85Q5V8sg/view?usp=sharing) and Extract the dataset into `data/` folder
6. Train MultiClassifier module using command: `python train.py`
7. Evaluate Model: `python evaluation.py -d ./data/valid_data.csv`

### Model Architecture
<img src="./notebooks/model.png" width="100%" alt="model_architecture">


#### Training Loss Curve
<img src="./notebooks/epoch_loss.svg" width="100%" alt="loss_curve">

Ongoing Project...
