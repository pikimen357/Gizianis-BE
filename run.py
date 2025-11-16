# run.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Ini yang penting untuk diakses teman
        port=8000,
        reload=True
    )