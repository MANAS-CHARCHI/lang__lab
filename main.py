def main():
    print("Hello from lang-understand!")
    result = agent.invoke({"messages": HumanMessage(content="Find the best stock to buy in india?")})
    print(result)

if __name__ == "__main__":
    main()
