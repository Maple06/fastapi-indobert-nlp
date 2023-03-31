# Text Classification using IndoBERT with FastAPI Implementation.

### Usage
- Using docker <br>
`docker compose up --build`
    - When container is running successfully, it will take several minutes until localhost is available and usable. Just wait until FastAPI shows "Application startup complete" in the logs.

- Native <br>
`uvicorn main:app --host 0.0.0.0 --port 2233`
    - This runs the app on localhost port 2233

Send a post request to the api route "/api/v1" (localhost:2233/api/v1) that includes 2 body requests, "username" which is a string and "list_content" which is the texts list.

### Input
username: \<str\> <br>
list_content: \<list\> '["text1", "text2", "text3"]'

### Output
```
{'result': {
    'username': 'JohnDoe123',
    'category' : [
        {'Parenting' : 20%},
        {'Gigs Worker' : 34%},
        {'Health': 50%},
        {'Sport': ...},
        {'Lifestyle': ...},
        ...
        ]
    }
    'error-status': \<binary\>
}
```
(If error-status is 0, error-message is not outputted)

### This is a semi-ready for deployment module by interns at PT Kazee Digital Indonesia for private company usage (PT. IAM Influencer Indonesia)
