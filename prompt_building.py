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
    def build_prompt_file_specific(context_example, value_format_prompt):
        context_example = """ 
                    SCHEDULE 2 CHEMICAL STANDARDS

                    | Item | Chemical Parameter | Standard (expressed as a maximum concentration in milligrams per litre) |
                    | 1. | Alachlor | 0.005 |
                    | 2. | Antimony | 0.006 |
                    | 3. | Arsenic | 0.01 |
                    | 4. | Atrazine + N-dealkylated metabolites | 0.005 |
                    | 5. | Azinphos-methyl | 0.02 |
                    | 6. | Barium | 1.0 |
                    | 7. | Benzene | 0.001 |
                    | 8. | Benzo(a)pyrene | 0.00001 |
                    | 9. | Boron | 5.0 |
                    | 10. | Bromate | 0.01 |
                    | 11. | Bromoxynil | 0.005 |
                    | 12. | Cadmium | 0.005 |
                    | 13. | Carbaryl | 0.09 |
                    | 14. | Carbofuran | 0.09 |
                    | 15. | Carbon Tetrachloride | 0.002 |
                    | 16. | Chloramines | 3.0 |

                    """

        value_format_prompt = """ 
                                [
                                    {
                                        "Chemical Name": "Alachlor",
                                        "value": "0.005",
                                        "unit" :"mg/L",
                                        "type": "MAC", 
                                        "remark" : "None",
                                        "reference number" : "1",
                                        "listedunder": "None",
                                    },
                                    {
                                        "Chemical Name": "Antimony",
                                        "value": "0.006",
                                        "unit" :"mg/L",
                                        "type": "MAC", 
                                        "remark" : "None",
                                        "reference number" : "2",
                                        "listedunder": "None",
                                    },
                                    {
                                        "Chemical Name": "Arsenic",
                                        "value": "0.01",
                                        "unit" :"mg/L",
                                        "type": "MAC", 
                                        "remark" : "None",
                                        "reference number" : "3",
                                        "listedunder": "None",
                                    },
                                    {
                                        "Chemical Name": "Azinphos-methyl",
                                        "value": "0.02",
                                        "unit" :"mg/L",
                                        "type": "MAC", 
                                        "remark" : "None",
                                        "reference number" : "5",
                                        "listedunder": "None",
                                    },
                                    {
                                        "Chemical Name": "Barium",
                                        "value": "1.0",
                                        "unit" :"mg/L",
                                        "type": "MAC", 
                                        "remark" : "None",
                                        "reference number" : "6",
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
    def build_prompt(compare_columns : list) -> str:

        prompt_template = """
            You are an information extraction assistant. Your task is to extract relevant information for a given list of chemical names from a provided text chunk.

            Instructions:
            - You will be given a list of chemical names and a chunk of text that may contain tabular data or unstructured text.
            - Your goal is to identify each chemical name within the chunk and extract all associated information presented in the same row, line, or sentence as that chemical.
            - The text chunk may vary in format, including structured tables, semi-structured lists, or plain text.
            - The columns associated with each chemical may vary, and their names are not predefined.
            - For each chemical name, extract its associated data and present the output as a list of JSON objects.
            - Each JSON object should contain the chemical name and all other extracted column names as key-value pairs.
            - Maintain the original text formatting of the values as closely as possible without modification.
            - Output format (in JSON): Format the output strictly as valid JSON, using double quotes (") around all keys and string values. Return the result as a list of JSON objects.
            - Include the key "Chemical Name" with its corresponding value in each JSON object.
            - If a chemical name is not present in the chunk, return None for all associated values.
            - If a chemical name is present multiple times in the chunk, extract all associated information for each occurrence.
            - If a chemical name is present in the chunk but no associated information is available, return "None" for all other values.
            - If the information is not found within the same row, line, or sentence as the chemical name, return "None" for that field.
            - Empty or missing values should be returned as the string "None".

            Column-specific guidance:
            - "value": A numeric or descriptive value (e.g., "Present", "≤15", "4000.0", "Banned as a pesticide in the group of plant protection products") representing concentration, regulatory status, or control code. Extract it from the same line or row as the chemical. In case of chemicals from the PIC regulation, extract descriptions like “Banned as a pesticide in the group of plant protection products” or “Banned as other pesticide including biocides”.
            - "unit": Extract the unit of measurement (e.g., %, mg/mL, Bq/L) if explicitly specified along with the value. If not found, return "None".
            - "type": Determine the classification or standard type if present. For regulatory lists, extract phrases such as "DEA Controlled Substances Code Number" or "No DEA Controlled Substances Code Number listed". For radiological standards, if a numerical limit is given under a table labeled with terms like “Standard” or “Maximum”, classify it as "MAC" (Maximum Allowable Concentration).
            - "remark": Any qualifying statements, conditions, exemptions, technical notes, or contextual comments associated with the chemical should be extracted here. For PIC contexts, extract phrases like "Chemical qualifying for PIC notification" or "Chemical subject or partially subject to the PIC procedure" when available.
            - "listedunder": The category or classification under which the chemical is grouped (e.g., "additive", "precursor", "explosive", "fuel", "NATURAL RADIONUCLIDES", etc.). If not explicitly stated, return "None".
            - "Index No": Capture any formal reference code or regulatory label that appears near the chemical name (e.g., "ML8.Note 1.p"). If not found, return "None".
            - "item number": Capture the item number or index used in the source table to identify the chemical. If the table is grouped into sections (like Table 1, Table 2), prefix the item with the table number (e.g., "1.01", "2.03"). If not found, return "None".
            - "Reference No.": If the chemical is listed with a regulatory subheading like "ML8.a.1", extract this full code as its Reference No. Match the chemical with its bullet/numbered point (e.g., “1.” for ML8.a.1) and convert it into the format “ML8.a.<number>”. If not found, return "None".
            """ + "other values to be extracted: " + str(compare_columns)


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