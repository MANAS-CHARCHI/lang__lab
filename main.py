from dotenv import load_dotenv
import os


load_dotenv()
def main():
    print(os.environ.get("LANGSMITH_PROJECT"))
    print("Hello from lang-understand!")


if __name__ == "__main__":
    main()
