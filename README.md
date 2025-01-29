# SpendWatch
SpendTrack is a machine learning tool that classifies credit card transactions using a fine-tuned BERT model. This allows users to gain insights into their spending patterns automatically.

## Features
- Train a BERT model on labeled credit card statements to classify transactions.
- Use the provided Python script to run the trained model on new credit card statements.
- Gain insights into spending habits based on transaction categories.
- Generate visualizations of spending distribution by category using Plotly.

## Installation 
```
pip install -r requirements.txt
```

## Usage
### Running the Model on a Statement
To classify transactions in a new credit card statement, run this in your terminal:

'''
python SpendWatch.py
'''

### Steps
1. The script will prompt for a CSV file containing transaction data.
2. Users must specify which columns correspond to transaction descriptions and amounts.
3. The model will process and classify transactions into predefined categories.
4. A visual breakdown of spending will be displayed as a pie chart.

## Model Details
- The model is based on bert-base-uncased and fine-tuned on transaction descriptions.
- Categories include: Dining, Entertainment, Gas, Grocery, Mobility, Other, Pharmaceutical, Retail, Subscription, Transportation, and Travel.
- The trained model weights are stored in ```model_epoch_5.pt```
