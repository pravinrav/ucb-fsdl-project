# README

## Contents
- `data/get_data.sh` downloads the tiny-imagenet data into the data/tiny-imagenet-200 file
- `data/convert.py` converts the validation set into a format uniform with the train set
- `eval.csv` is a sample evaluation file
- `final_report.pdf` contains the final report for this project
- `final.pt` contains the final Pytorch model
- `model.py` contains the Wide ResNet-50 architecture that was finetuned via transfer learning
- `requirements.txt` is the python requirements necessary to run this project
- `test_submission.py` takes in a CSV as input and outputs predictions 'eval_classified.csv’
- `train_sample_torch.py` trains Wide ResNet-50 models on the baseline and augmented data set.
- `train_adversarial.py` trains ResNets on adversarially generated images
- `train_distinguish_adv.py` trains ResNets to classify images as natural, FGSM, or BIM-perturbed
- `validate.py` runs the model on the validation datasets

## Setup (run all commands from top level directory)
```
pip install -r requirements.txt
cd data && ./get_data.sh
mv convert.py data/tiny-imagenet-200/val/convert.py && cd data/tiny-imagenet-200/val &&  python3 convert.py
git clone https://github.com/IBM/adversarial-robustness-toolbox.git
cp model.py train_adversarial.py validate_adversarial.py adversarial-robustness-toolbox/
```

## Training
```
python3 train_sample_torch.py
cd adversarial-robustness-toolbox/ && python3 train_adversarial.py
python3 validate.py
```

## Test Set Evaluation
```
python3 test_submission.py path/to/csv
```
