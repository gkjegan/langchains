from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from third_party.linkedin_scrape import scrape_linkedin_profile
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")
print("*******Da " + OPENAI_API_KEY)
if __name__ == "__main__":
    print("Hello LangChain!")

    summary_template = """
         given the Linkedin information {information} about a person from I want you to create:
         1. a short summary
         2. two interesting facts about them
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    linkedin_profile_url = "gkjegan"
    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY
    )

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    print(chain.run(information=linkedin_data))
