import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import json

# llm intialisation

model_name = "gpt-4o-mini"
llm_deployment = "gpt-4o-mini"
api_version = "2024-12-01-preview"
embed_deployment = "text-embedding-3-small"

load_dotenv()


llm_client = AzureOpenAI(
                        api_version=api_version,
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        api_key=os.getenv("OPENAI_API_KEY"),
                        azure_deployment = llm_deployment,
                        )


class prompt_building:

    @staticmethod
    def build_prompt(context_example, value_format_prompt):
        context_example = """
                                    | --- | --- |
                                    | (19) Pethidine (meperidine) | 9230 |
                                    | (20) Pethidine-Intermediate-A, 4-cyano-1-methyl-4-phenylpiperidine | 9232 |
                                    | (21) Pethidine-Intermediate-B, ethyl-4-phenylpiperidine-4-carboxylate | 9233 |
                                    | (22) Pethidine-Intermediate-C, 1-methyl-4-phenylpiperidine-4-carboxylic acid | 9234 |
                                    | (23) Phenazocine | 9715 |
                                    | (24) Piminodine | 9730 |
                                    | (25) Racemethorphan | 9732 |
                                    | (26) Racemorphan | 9733 |
                                    | (27) Remifentanil | 9739 |
                                    | (28) Sufentanil | 9740 |
                                    | (29) Tapentadol | 9780 |
                                    | (30) Thiafentanil | 9729 |
                                    
                        """

        value_format_prompt = """ 
                                    [
                                        {
                                            "Chemical Name": "Pethidine",
                                            "type": "DEA Controlled Substances Code Number",
                                            "value": 9230,
                                            "remark" : "None",
                                            "listedunder": "None"
                                        },
                                        {
                                            "Chemical Name": "Tapentadol",
                                            "type": "DEA Controlled Substances Code Number",
                                            "value": 9780,
                                            "remark" : "None",
                                            "listedunder": "None"
                                        },
                                        {
                                            "Chemical Name": "Phenazocine",
                                            "type": "DEA Controlled Substances Code Number",
                                            "value": 9715,
                                            "remark" : "None",
                                            "listedunder": "None"
                                        },
                                    ]
                                    """

            # Example JSON format with corrected syntax
        value_extract_example = f"""Output format (in JSON): Please format the output strictly as valid JSON, using double quotes (") around all keys and string values.
                                    {value_format_prompt}
            
                                """

        # Combined value extraction section
        value_extraction = f"""Context: {context_example}
        Values to extract: {value_extract_example}
        """

        # Final prompt template
        prompt_template = f"""You are a Chemical relational values extraction expert.

            Instructions for Extracting Other Values:

            You will be provided with list of chemicals and a context, from the given context, extract other values for every chemical given in the list. 
            
            Take the example given below as a refference. If value not exit then return None for that variable

            {value_extraction}

            """ 
        return prompt_template
    

    @staticmethod
    def build_input_text(chunk_chemical, chunk_text):

        input_text = f""" 
        list of chemical for which values to be extracted:
        {chunk_chemical}

        chunk text from which values to be extracted
        {chunk_text}
        """
        return input_text
    
    @staticmethod
    def other_value_extractions_from_llm(prompt_template, input_text):

        response = llm_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt_template,
            },
            {
                "role": "user",
                "content": input_text,
            }
        ],
        max_tokens=4096,
        temperature=0.1,
        top_p=1.0,
        model=llm_deployment
                            )
        
        response_data = response.choices[0].message.content

        cleaned_response = response_data.replace("```json", "").replace("```", "").lstrip().rstrip()
        # Extra safety: Replace single quotes with double quotes (if clearly JSON-like)

        if cleaned_response.startswith("[{") and "'" in cleaned_response and '"' not in cleaned_response:
            cleaned_response = cleaned_response.replace("'", '"')
            
        chemicals_extracted = json.loads(cleaned_response)

        return chemicals_extracted