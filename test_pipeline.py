import pandas as pd
from pre_processing import Pre_processing
from prompt_building import prompt_building
import re
from other_value_evaluation import ValuesComparision
import numpy as np
from database_client import db_data_extraction

# === Config === #
# md_path = "data\\CND_Ontaria\\cnd_ontario_dw.md"

# extracted_chemical_path = "data\\CND_Ontaria\\list_6943_extracted_chemicals.xlsx"

# ground_truth_path = "data\\CND_Ontaria\\list_6943_gt.xlsx"

# compare_columns = ["value", "unit", "type", "remark", "item number", "listedunder"]

# # prompt building
# context_example = ""  # Example context for value format
# value_format_prompt = [{}]  # List of JSON examples for value format
# prompt_template = prompt_building.build_prompt(context_example, value_format_prompt)

def data_preparation(list_id, extracted_chemicals_df):

    chunks = Pre_processing.creat_chunks(list_id, chunk_size=2000)
    print(f"chunks created total chunks {len(chunks)}")

    # df = pd.read_excel(extracted_chemical_path, header=0)

    keyword_processor = Pre_processing.add_extracted_chemicas_to_flashtext(extracted_chemicals_df)
    print(f"added extrcated chemicals to keyword processor")

    return chunks, keyword_processor

def other_value_extraction_from_text(chunks, keyword_processor, prompt_template):
    
    # other value extraction 
    chemical_other_values = []

    for i, chunk in enumerate(chunks):
        
        chemicals_in_chunk = keyword_processor.extract_keywords(chunk)

        chemicals_in_chunk = list(set(chemicals_in_chunk))

        if not chemicals_in_chunk:
            print(f"#### NO values extracted for the chunk number {i}, number of chemicals in the chunk 0")
            continue

        input_text = prompt_building.build_input_text(chemicals_in_chunk, chunk)

        extracted_values = prompt_building.other_value_extractions_from_llm(prompt_template, input_text)
        
        print(f">>>> chunk number {i} >>>>extracted chemicals {len(extracted_values)}>>>>actual chemicals {len(chemicals_in_chunk)} ")
        print(f"################ >>>>>>> missed chemicals {len(chemicals_in_chunk)-len(extracted_values)}>>>>>>>>################ ")

        chemical_other_values.extend(extracted_values)

    return chemical_other_values

# post processing
def post_processing_data(chemical_other_values, extracted_chemicals_df):

    df = pd.DataFrame(chemical_other_values)
    # df.to_csv(extracted_csv_path, index=False)

    df.to_excel("data\\CND_Ontaria\\predicted_othervalues.xlsx",index=False)

    # df1 = pd.read_excel(extracted_chemical_path,header=0)
    df1 = extracted_chemicals_df

    # Standardize both to lowercase
    df1['Chemical Name'] = df1['Chemical Name'].str.lower()
    df['Chemical Name'] = df['Chemical Name'].str.lower()

    merged_df = pd.merge(df1, df, on='Chemical Name', how='left') # need to work arround it issues with mapping properly 

    merged_df.to_excel("data\\CND_Ontaria\\merged_base_othervalues.xlsx",index=False)

    mapped_df = Pre_processing.map_othervalues_to_RR_chemicals(merged_df)
    mapped_df.replace("None", np.nan, inplace=True)

    mapped_df.to_excel("data\\CND_Ontaria\\rr_mapped_other_values.xlsx", index=False)

    del(df)
    del(df1)
    del(merged_df)

    return mapped_df

# evaluation 
def evaluate_the_extracted_values(mapped_df, compare_columns, ground_truth_df):

    # ground_truth_df = pd.read_excel(ground_truth_path)

    ground_truth_df['Chemical Name'] = ground_truth_df['Chemical Name'].str.lower()

    evaluation_results = ValuesComparision.evaluate_extraction_accuracy(ground_truth_df, mapped_df, compare_columns)

    return evaluation_results

def get_columns_to_compare(ground_truth_df):

    columns_compare = list(ground_truth_df.columns[2:])

    if 'Formula' in columns_compare:
        columns_compare.remove('Formula')

    columns_compare

    return columns_compare

if __name__ == "__main__":

    # md_path = "data\\CND_Ontaria\\cnd_ontario_dw.md"

    # extracted_chemical_path = "data\\CND_Ontaria\\list_6944_extracted_chemicals.xlsx"
    
    # ground_truth_path = "data\\CND_Ontaria\\list_6944_gt.xlsx"

    # compare_columns = ["value", "unit", "type", "remark", "reference number", "listedunder"]
    
    # pipeline 
    list_id = 811

    ground_truth_df, extracted_chemicals_df = db_data_extraction.get_ground_truth_data(list_id)

    compare_columns = get_columns_to_compare(ground_truth_df)

    chunks, keyword_processor = data_preparation(list_id, extracted_chemicals_df)

    prompt_template = prompt_building.build_prompt(compare_columns)

    chemical_other_values = other_value_extraction_from_text(chunks, keyword_processor, prompt_template)

    mapped_df = post_processing_data(chemical_other_values, extracted_chemicals_df)

    evaluation_results = evaluate_the_extracted_values(mapped_df, compare_columns, ground_truth_df)

    print(evaluation_results)

