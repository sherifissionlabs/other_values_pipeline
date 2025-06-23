import pandas as pd
from pre_processing import Pre_processing
from prompt_building import prompt_building
import re
from other_value_evaluation import ValuesComparision

# === Config === #
md_path = "data\\A_SUSMP\\A_SUSMP_S5.md"

extracted_chemical_path = "data\\A_SUSMP\\A_susmp_2215_extracted_chemicals.xlsx"

ground_truth_path = "data\\A_SUSMP\\A_susmp_2215_gt.xlsx"

compare_columns = ["value", "unit", "type", "remark", "listedunder"]

# prompt building
context_example = ""  # Example context for value format
value_format_prompt = [{}]  # List of JSON examples for value format
prompt_template = prompt_building.build_prompt(context_example, value_format_prompt)


chunks = Pre_processing.creat_chunks(md_path, chunk_size=2000)
print(f"chunks created total chunks {len(chunks)}")

df = pd.read_excel(extracted_chemical_path, header=0)

keyword_processor = Pre_processing.add_extracted_chemicas_to_flashtext(extracted_chemical_path)
print(f"added extrcated chemicals to keyword processor")


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

# post processing 
df = pd.DataFrame(chemical_other_values)
# df.to_csv(extracted_csv_path, index=False)

df.to_excel("data\\A_SUSMP\\predicted_othervalues.xlsx",index=False)

df1 = pd.read_excel(extracted_chemical_path,header=0)

merged_df = pd.merge(df1, df, on='Chemical Name', how='left') # need to work arround it issues with mapping properly 

merged_df.to_excel("data\\A_SUSMP\\merged_base_othervalues.xlsx",index=False)

mapped_df = Pre_processing.map_othervalues_to_RR_chemicals(merged_df)

mapped_df.to_excel("data\\A_SUSMP\\rr_mapped_other_values.xlsx", index=False)


# evaluation 
ground_truth_df = pd.read_excel(ground_truth_path)

evaluation_results = ValuesComparision.evaluate_extraction_accuracy(ground_truth_df, mapped_df, compare_columns)


if __name__ == "__main__":

    pass 