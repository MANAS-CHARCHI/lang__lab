from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic


def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query (str): The search query
    Returns:
        str: The search results
    """
    print(f"Searching for: {query}")
    return f"Results for '{query}'"


llm = ChatAnthropic(model="")

def main():
    print("Hello from lang-understand!")


if __name__ == "__main__":
    main()
