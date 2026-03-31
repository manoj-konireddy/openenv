from server import app as fastapi_app
import uvicorn


def main():
    uvicorn.run("server:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()