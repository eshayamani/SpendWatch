import pandas as pd
import torch
import plotly.express as px
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()


## PROCESS CSV FILE AND INFO
def process_csv():
    #get the file path 
    while True:
        file_path = input("Enter the path to your CSV file: ")
        try:
            df = pd.read_csv(file_path)
            print("CSV loaded successfully!")
            break
        except Exception as e:
            print(f"Error reading the CSV file: {e}")
            print("Please try again.")

    print("\nColumns in the file:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    
    #get the column number for descriptions
    while True:
        try:
            description_col = int(input("\nEnter the column number for descriptions: "))
            if 0 <= description_col < len(df.columns):
                break
            else:
                print(f"Invalid column number. Please enter a number between 0 and {len(df.columns) - 1}.")
        except ValueError:
            print("Please enter a valid integer.")
    
    #get the column number for amounts        
    while True:
        try:
            amount_col = int(input("Enter the column number for transaction amounts: "))
            if 0 <= amount_col < len(df.columns):
                break
            else:
                print(f"Invalid column number. Please enter a number between 0 and {len(df.columns) - 1}.")
        except ValueError:
            print("Please enter a valid integer.")

    #get yes or no for negative amounts
    while True:
        negative_input = input("\nAre transactions posted as negative amounts? (yes/no): ").strip().lower()
        if negative_input in ['yes', 'no']:
            negative_amounts = negative_input == 'yes'
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    return df, description_col, amount_col, negative_amounts


## TEST PROCESSING FUNCTIONS
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def text_processing(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_text(texts, tokenizer, max_length=128):
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded


## TESTING FUNCTION
def test_function(test_df, description_col, amount_col, tokenizer, model, neg=False):
    #rename columns
    test_df.rename(
        columns={test_df.columns[description_col]: 'Description', test_df.columns[amount_col]: 'Amount'},
        inplace=True
    )
    
    #create new dataframe
    filtered_df = test_df[['Description', 'Amount']].copy()
    
    #remove empty rows
    filtered_df.dropna(inplace=True)

    #if negative, convert to positive
    if neg:
        filtered_df['Amount'] = filtered_df['Amount'] * -1
        
    #filter to remove any user payments
    filtered_df = filtered_df[filtered_df['Amount'] >= 0]

    #apply text processing
    filtered_df['Description'] = filtered_df['Description'].apply(text_processing)
    all_texts = filtered_df['Description'].tolist()
    encoded = preprocess_text(all_texts, tokenizer)
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    #predicting spending categories
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
    
    label_mapping = {0: 'Dining', 1: 'Entertainment', 2: 'Gas', 3: 'Grocery', 4: 'Mobility', 5: 'Other', 6: 'Pharmaceutical', 7: 'Retail', 8: 'Subscription', 9: 'Transportation', 10: 'Travel'}

    filtered_df['predictions'] = predictions.numpy()
    filtered_df['Spending Category'] = filtered_df['predictions'].map(label_mapping)

    #grouping by categories and visualizing
    spending_by_category = filtered_df.groupby('Spending Category')['Amount'].sum().reset_index()

    fig = px.pie(
        spending_by_category,
        names='Spending Category',
        values='Amount',
        title="Proportion of Total Spending by Category",
        color='Spending Category',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.3
    )

    #highlight the largest category
    fig.update_traces(
        textinfo='percent+label',
        pull=[0.1 if i == spending_by_category['Amount'].idxmax() else 0 for i in range(len(spending_by_category))]
    )

    fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
    fig.show()

    return filtered_df

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = "bert-base-uncased"
    model_path = "model_epoch_5.pt"
    num_labels = 11
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
    
    try:
        df, description_col, amount_col, neg_amts = process_csv()
        print("Running the test function...")
        processed_df = test_function(df, description_col, amount_col, tokenizer, model, neg=neg_amts)

        print("\nProcessed DataFrame:")
        print(processed_df.head())
    except Exception as e:
        print(f"An error occurred: {e}")