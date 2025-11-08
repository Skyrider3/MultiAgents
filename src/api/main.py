"""
Main entry point for the FastAPI application
"""

from src.api.app import create_app

# Create the FastAPI app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    from src.config import settings

    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.environment == "development"
    )