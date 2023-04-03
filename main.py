import uvicorn
from fastapi import FastAPI
from app.api.router import routerv1

app = FastAPI()
app.include_router(routerv1)

# Default root path
@app.get('/')
async def root():

    message = {
        'message': 'This is IndoBERT Text Classification API v1.0'
    }

    return message

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2323)