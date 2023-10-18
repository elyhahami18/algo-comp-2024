#!/usr/bin/env python3
import json
import sys
import os

import numpy as np
from numpy.linalg import norm
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = '/workspaces/algo-comp-2024/assignment1/testdata.json' # Constant variables are usually in ALL CAPS

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class User:
    def __init__(self, name, gender, preferences, grad_year, responses):
        self.name = name
        self.gender = gender
        self.preferences = preferences
        self.grad_year = grad_year
        self.responses = responses

# Function that, hypothetically given bios, return similarity via BERTModel. 
# Note that in the testdata.json example, we are not given bios, so this function is entirely theoretical
def BERT_sentence_similarity(sentence1, sentence2):
    # Tokenize and convert to embeddings
    tokens1 = tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True)
    tokens2 = tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True)

    # Get BERT embeddings
    with torch.no_grad():
        output1 = model(**tokens1)
        output2 = model(**tokens2)

    # Use the [CLS] token embeddings as representations
    embedding1 = output1.last_hidden_state.mean(dim=1)
    embedding2 = output2.last_hidden_state.mean(dim=1)

    # Calculate cosine similarity
    return cosine_similarity(embedding1, embedding2)

# The logic here is that grad_year is a proxy for age, and closer differences in graduation years have higher compatibility
weights_for_grad_years = {0: 1, 
                          1: 0.9, 
                          2: 0.7, 
                          3: 0.5,
                          4: 0}
# Takes in two user objects and outputs a float denoting compatibility
def compute_score(user1, user2):
    age_diff_weight = weights_for_grad_years[abs(user1.grad_year - user2.grad_year)]

    # Implementing Cosine similarity on responses vector 
    cosine_sim = np.dot(user1.responses,user2.responses)/(norm(user1.responses)*norm(user2.responses))

    # Match preferences (ie. straight people with straight people ONLY, gay people with gay people ONLY)
    if (user1.gender == user2.preferences[0] and user2.gender == user1.preferences[0]):
        return (cosine_sim * age_diff_weight)
    else:
        return 0.0

if __name__ == '__main__':
    # Make sure input file is valid
    if not os.path.exists(INPUT_FILE):
        print('Input file not found')
        sys.exit(0)

    users = []
    with open(INPUT_FILE) as json_file:
        data = json.load(json_file)
        for user_obj in data['users']:
            new_user = User(user_obj['name'], user_obj['gender'],
                            user_obj['preferences'], user_obj['gradYear'],
                            user_obj['responses'])
            users.append(new_user)

    for i in range(len(users)-1):
        for j in range(i+1, len(users)):
            user1 = users[i]
            user2 = users[j]
            score = compute_score(user1, user2)
            print('Compatibility between {} and {}: {}'.format(user1.name, user2.name, score))
# print(BERT_sentence_similarity("Hello, I enjoy playing basktball and skiing", "I like to play sports!"))