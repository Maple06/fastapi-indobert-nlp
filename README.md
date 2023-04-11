![Repo card](https://repository-images.githubusercontent.com/621719067/cca8448f-60dc-47f0-ac09-781ecb86dfb7)
# Text Classification using IndoBERT with FastAPI Implementation.

### Usage
- Using docker <br>
`docker compose up --build`
    - When container is running successfully, it will take several minutes until localhost is available and usable. Just wait until FastAPI shows "Application startup complete" in the logs.

- Native <br>
`uvicorn main:app --host 0.0.0.0 --port 2323`
    - This runs the app on localhost port 2323

Send a post request to the api route "/api/v1" (localhost:2323/api/v1) that includes 2 body requests, "username" which is a string and "list_content" which is the texts list.

#### Usage in postman: [Postman Collection](API-Postman-Collection.json)

### Input (RAW JSON)
```
{
    "username":"\<str\>"
    "list_content":\<list\> eg. ["text1", "text2", "text3"]
}
```

### Output
```
{'result': {
    'username': 'JohnDoe123',
    'category' : [
        {'Parenting' : 50.05%},
        {'Gigs Worker' : 23.3%},
        {'Health': 10.25%},
        {'Sport': ...},
        {'Lifestyle': ...},
        ...
        ]
    }
    'error-status': \<binary\>
}
```
(If error-status is 0, error-message is not outputted)

### This is a ready for deployment module by interns at PT Kazee Digital Indonesia for private company usage (PT. IAM Influencer Indonesia)
