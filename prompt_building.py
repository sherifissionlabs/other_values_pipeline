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
                        PHENOL, including cresols and xylenols and any other homologue of phenol boiling below
                        220°C, when in animal feed additives containing 15% or less of such substances,
                        except in preparations containing 1% or less of phenol and in preparations containing
                        3% or less of cresols and xylenols and other homologues of phenol.

                        PHENYL METHYL KETONE except in preparations containing 25% or less of designated
                        solvents

                        PERMETHRIN (excluding preparations for human therapeutic use):

                        (a) in preparations containing 25% or less of permethrin; or

                        (b) in preparations for external use, for the treatment of dogs, containing 50% or less
                        of permethrin when packed in single use containers having a capacity of 4 mL or
                        less;

                        except in preparations containing 2% or less of permethrin.

                        PRALLETHRIN (cis:trans=20:80) in preparations containing 10% or less of prallethrin
                        except in insecticidal mats containing 1% or less of prallethrin.

                        POLIXETONIUM SALTS in preparations containing 60% or less of polixetonium salts
                        except in preparations containing 1% or less of polixetonium salts.
                        """

        value_format_prompt = """ 
                                [
                                    {
                                        "Chemical Name": "PHENOL",
                                        "value": "<=15",
                                        "unit" :"%",
                                        "type": "None", 
                                        "remark" : "including cresols and xylenols and any other homologue of phenol boiling below 220°C, except in preparations containing 1% or less of phenol and in preparations containing 3% or less of cresols and xylenols and other homologues of phenol.",
                                        "listedunder": "None",
                                    },
                                    {
                                        "Chemical Name": "PHENYL METHYL KETONE",
                                        "value": "None",
                                        "unit" : "None",
                                        "type": "None", 
                                        "remark" : "except in preparations containing 25% or less of designated solvents",
                                        "listedunder": "None",
                                    },
                                    {
                                        "Chemical Name": "PERMETHRIN",
                                        "value": "<=25",
                                        "unit" : "%",
                                        "type": "None", 
                                        "remark" : "excluding preparations for human therapeutic use, except in preparations containing 2% or less of permethrin.",
                                        "listedunder": "None",
                                    },
                                    {
                                        "Chemical Name": "PERMETHRIN",
                                        "value": "<=50",
                                        "unit" : "%",
                                        "type": "None", 
                                        "remark" : "excluding preparations for human therapeutic use, except in preparations containing 2% or less of permethrin. when packed in single use containers having a capacity of 4 mL or less",
                                        "listedunder": "None",
                                    },
                                    
                                    {
                                        "Chemical Name": "PRALLETHRIN",
                                        "value": "<=10", 
                                        "unit" : "%",
                                        "type": "None", 
                                        "remark" : "except in insecticidal mats containing 1% or less of prallethrin.",
                                        "listedunder": "None",
                                    },
                                    {
                                        "Chemical Name": "POLIXETONIUM SALTS",
                                        "value": "<=60",
                                        "unit" : "%",
                                        "type": "None", 
                                        "remark" : "except in preparations containing 1% or less of polixetonium salts.",
                                        "listedunder": "None",
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