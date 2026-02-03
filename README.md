# redfish-nlp-log-classification

A service that will be used to classify the logs.


## NLP model
The dataset contains Redfish logs that may belong to 3 classes:
1. hardware: SEL logs related to hardware
2. security: logs related to users
3. power: logs related to power status

## Usage
### Training
```
source venv/bin/activate
cd training
python train.py
```
The model will be stored in `models` directory.

### Test
```
source venv/bin/activate
cd test
python test.py
```


## Resources
- [Logs Dataset](https://www.kaggle.com/code/adepvenugopal/logs-dataset)