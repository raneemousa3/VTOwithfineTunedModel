

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Virtual Fitting Room API",
    description="API for AI-powered virtual fitting room for e-commerce",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import and include routers
# from app.api.routes import measurement_router, size_router, user_router
# app.include_router(measurement_router)
# app.include_router(size_router)
# app.include_router(user_router)

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "message": "Virtual Fitting Room API",
        "version": "0.1.0",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
