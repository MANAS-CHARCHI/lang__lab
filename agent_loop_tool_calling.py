from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10

load_dotenv() 


@tool
def get_product_price(product: str) -> str:
    """Look up the price of the product in the catalog"""
    print("  >> Executing the price of the product : ", product)
    prices = {"gold": 100, "platinum": 200, "diamond": 300}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the producd and return the final price
    Available discount tiers: gold, platinum, diamond
    """
    print("  >> Executing the discount of the product : ", discount_tier)
    discounts = {"gold": 0.2, "platinum": 0.3, "diamond": 0.4}
    return round(price * (1 - discounts.get(discount_tier, 0)))

#  Agent Loop ------------

@traceable(name="langchain agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name : t for t in tools}
    # print(tools_dict)
    llm = init_chat_model(model="claude-haiku-5-5")
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content="""
        You are a helpful assistant.
        you have access to the following tools:
        """ + "\n".join([f"{t.name}: {t.description}" for t in tools]) + """
        "STRICT RULE: DO NOT MAKE UP A RESPONSE.".
        "1 - Use tools to get your answer, if you don't know the answer, just say 'I don't know'."
        "2 - If asked for the price of a product, use the get_product_price tool."
        "3 - If asked for the discount of a product, use the apply_discount tool only after getting the product price from the get_product_price tool."
        """),
        HumanMessage(content=question),
    ]

    for i in range(1, MAX_ITERATIONS+1):
        print(" >> Iteration --- : ", i)
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        print(" >> Response: ", response)
        tool_calls = response.tool_calls
        if not tool_calls:
            print("Final Answer: ", response.content)
            return response.content
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(" >> Tool:", tool_name)
            print(" >> Arguments:", tool_args)

            tool_result = tools_dict[tool_name].invoke(tool_args)

            print(" >> Result:", tool_result)

            # Immediately provide the result for this tool call
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                )
            )
    print(" >> Max Iterations Reached")
    return


if __name__ == "__main__":
    print("Hello from lang-loop")
    print()
    result = run_agent("What is the price of a gold and diamond after appling the discount?")