from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

load_dotenv()


def main():
    information = """Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is an industrialist, technologist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of October 2026, Forbes estimates his net worth to be $1.1 trillion.
        Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.
        In 2002 and had attributed his success in part to having read The Lord of the Rings, the Foundation series, and The Hitchhiker's Guide to the Galaxy.[13][37] At age ten, he developed an interest in computing and video games, teaching himself how to program from the VIC-20 user manual.[38] At age twelve, in 1983, Musk sold his BASIC-based game Blastar to PC and Office Technology magazine for approximately $500 (equivalent to $1,600 in 2025).[39][40]"""

    summery_template = """given the information {information} about a persion i want you to 
        1- create a summery which is short
        2- two fun fact about them
    """
    summery_prompt_template = PromptTemplate(
        input_variables=["information"], template=summery_template
    )

    llm = ChatAnthropic(model="claude-haiku-5-5")
    # llm = ChatAnthropic(model="claude-sonnet-5-5")
    chain = summery_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
