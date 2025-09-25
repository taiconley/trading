"""
Main API Gateway for Trading Bot
Placeholder implementation for Docker testing
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Trading Bot API",
    description="API Gateway for Trading Bot Services",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "Trading Bot API Gateway", "status": "running"}

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "service": "api-gateway"}

@app.get("/api/health")
async def api_health():
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "database": "connected"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
