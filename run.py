import uvicorn
import os

# SERVER CONFIGURATION
HOST = "0.0.0.0"
PORT = 8080
RELOAD = True

# APPLICATION ENTRY POINT
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host=HOST, 
        port=PORT, 
        reload=RELOAD
    )