import streamlit as st 
#from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.llms import CTransformers

#from langchain.agents import create_csv_agent
#from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI  
from dotenv import load_dotenv
import pandas as pd 


def main():

    load_dotenv()

    st.set_page_config(page_title=" bills ",page_icon="😷😷")
    st.header("  bills ")

    user_csv = st.file_uploader ("upload ur bills ", type="csv")
    #if  a user uploads a csv file first we have to find it
    if user_csv is not None :
        user_question = st.text_input("ask regarding bills ")

        #below we have to choose our language model here we are choosin openai
        openai_api_key="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"
        #llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    #model_type="llama",
                    #max_new_tokens=512,

                    #temperature=0.5)
        llm = OpenAI(temperature=0) # 0 is non creative where as 10 is completly creative
        #llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    #model_type="llama",
                    #max_new_tokens=512,
                    #temperature=0.1)
        agent = create_csv_agent(llm,user_csv, handle_parsing_errors=True,verbose = True ) #verbose is sinply the thinking of model will be printed 
        #imporant its running code by itself it not asking permisiion from anyone


        if user_question is not None and user_question !="":
            response =agent.run(user_question)
                                


            st.write(response)






if __name__== "__main__":
    main()