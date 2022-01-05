# cern-cms-detect-exp
Training a PyTorch based Sequential (linear) model on CERN CMS Particle Collision Detection Data.

## Introduction
This notebook is based on a Kaggle Contest [TrackML](https://www.kaggle.com/c/trackml-particle-identification).

## The model
The model has the following architecure. fs or feature size = 10 since we take 5 per hit, and hits are in pair.

`        nn.Linear(fs, 800),`\
`        nn.ReLU(),`\
        `nn.Linear(800, 400),`\
        `nn.ReLU(),`\
        `nn.Linear(400, 400),`\
        `nn.ReLU(),`\
        `nn.Linear(400, 400),`\
       ` nn.ReLU(),`\
       ` nn.Linear(400, 200),`\
       ` nn.ReLU(),`\
      `  nn.Linear(200, 1),`\
     `   nn.Sigmoid()`\

## Accuracy
98.88% after 4 epochs in total 486.61s or roughly 8 min.

We also tried Hard Negative Mining but it didn't yield any greater results - they gave us a reduced accuracy of 97.47% and even taking 1.02% more time from what we originally got without hard negative mining.
