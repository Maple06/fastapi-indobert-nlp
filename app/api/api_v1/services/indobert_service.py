import os
import torch
import torch.nn.functional as F

import platform

from ....core.logging import logger
from ...load_models import trainDataset, loadTokenizer, loadModel

CWD = os.getcwd()

PATH = f"{CWD}/ml-models"
CATEGORY_LIST = ['Lifestyle', 'Otomotif', 'Music', 'Beauty', 'Fashion', 'Traveling', 'Food', 'Finance', 'Parenting', 'Technology', 'Health', 'Gigs Worker', 'Homedecor', 'Gamers', 'Sport', 'Reviewer', 'Kpop', 'Politik']

# Module specific business logic (will be use for endpoints)
class NLPService:
    def __init__(self):
        pass

    def predict(self, username, texts):
        logger.info(f"API request received. Predicting {username} with {len(texts)} texts")
        
        loadModel.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        loadModel.to(device)
        if device == "cuda":
            print("Using device:", torch.cuda.get_device_name(0), flush=True)
        elif device == "cpu":
            print("Using device:", platform.processor(), flush=True)

        result = {
            'Lifestyle': 0.0, 
            'Otomotif': 0.0, 
            'Music': 0.0, 
            'Beauty': 0.0, 
            'Fashion': 0.0, 
            'Traveling': 0.0, 
            'Food': 0.0, 
            'Finance': 0.0, 
            'Parenting': 0.0, 
            'Technology': 0.0, 
            'Health': 0.0, 
            'Gigs Worker': 0.0, 
            'Homedecor': 0.0, 
            'Gamers': 0.0, 
            'Sport': 0.0, 
            'Reviewer': 0.0, 
            'Kpop': 0.0, 
            'Politik': 0.0
            }
        
        for text in texts:
            encoded_review = loadTokenizer.encode_plus(
                                                text,
                                                max_length=50,
                                                truncation=True,
                                                add_special_tokens=True,
                                                return_token_type_ids=False,
                                                padding=True,
                                                return_attention_mask=True,
                                                return_tensors='pt'
                                                )

            input_ids = encoded_review['input_ids'].to(device)
            attention_mask = encoded_review['attention_mask'].to(device)

            output = loadModel(input_ids, attention_mask)
            logits = output[0]
            probabilities = F.softmax(logits, dim=1)

            for i, category in enumerate(CATEGORY_LIST):
                updatedProba = result[category] + float(f"{probabilities[0][i].item()*100:.2f}")
                result.update({category: updatedProba})

        resultList = []

        result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

        for i in result:
            score = result[i] / len(texts)
            resultList.append({i: f"{score:.2f}%"})

        output = {
            'result': {
                'username': username,
                'prediction': list(resultList[0].keys())[0],
                'category': resultList
            }
        }

        return output