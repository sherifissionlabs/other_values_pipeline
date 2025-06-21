# main pipelines 

import pandas as pd
from pre_processing import Pre_processing
from prompt_building import prompt_building
from other_value_evaluation import ValuesComparision
import re

# === Config === #
md_path = "data\\US_DEA_Source_file.md"
chemicals_xlsx_path = "data\\US_DEA_chemicals.xlsx"

# Inputs from user/UI
context_example = ""  # Example context for value format
value_format_prompt = [{}]  # List of JSON examples for value format

# === Pipeline === #

def map_othervalues_to_RR_chemicals(df):
    # Load input
    # df = pd.read_csv("inputrr1.csv")

    # Ensure consistent column order
    columns_to_copy = [col for col in df.columns if col not in ["CAS", "Chemical Name"]]

    # Split base and RR entries
    base_df = df[~df["CAS"].str.startswith("RR-")].copy()
    rr_df = df[df["CAS"].str.startswith("RR-")].copy()

    # Normalize names
    def normalize(name):
        return str(name).lower().strip()

    # Fill RR rows with corresponding base info using strict word-boundary matching
    for rr_idx, rr_row in rr_df.iterrows():
        rr_name = normalize(rr_row["Chemical Name"])
        
        for _, base_row in base_df.iterrows():
            base_name = normalize(base_row["Chemical Name"])
            
            # Use \b to ensure it matches whole words or with space boundaries
            if re.search(rf"\b{re.escape(base_name)}\b", rr_name):
                for col in columns_to_copy:
                    rr_df.at[rr_idx, col] = base_row.get(col, "")
                break  # Use first matching base chemical

    # Combine base + updated RR rows back
    final_df = pd.concat([base_df, rr_df], ignore_index=True)

    # # Save output
    # final_df.to_csv("output_with_base_data1.csv", index=False)
    # print(final_df)

    return final_df

def run_extraction_pipeline(md_path: str, chemicals_xlsx_path: str, prompt_template: str) -> str:


    chunks = Pre_processing.creat_chunks(md_path, chunk_size=2000)
    print(f"chunks created total chunks {len(chunks)}")

    keyword_processor = Pre_processing.add_extracted_chemicas_to_flashtext(chemicals_xlsx_path)
    print(f"added extrcated chemicals to keyword processor")

    chemical_other_values = []

    for i, chunk in enumerate(chunks):
        chemicals_in_chunk = keyword_processor.extract_keywords(chunk)

        if not chemicals_in_chunk:
            print(f"#### NO values extracted for the chunk number {i}, number of chemicals in the chunk 0")
            continue

        input_text = prompt_building.build_input_text(chemicals_in_chunk, chunk)

        extracted_values = prompt_building.other_value_extractions_from_llm(prompt_template, input_text)
        
        print(f">>>> chunk number {i} >>>>extracted chemicals {len(extracted_values)}>>>>actual chemicals {len(chemicals_in_chunk)} ")
        print(f"################ >>>>>>> missed chemicals {len(chemicals_in_chunk)-len(extracted_values)}>>>>>>>>################ ")

        chemical_other_values.extend(extracted_values)

    df = pd.DataFrame(chemical_other_values)
    # df.to_csv(extracted_csv_path, index=False)

    print("######################>>>>>>>>>>> extraction completed <<<<<<<<<<<<<####################")

    # merge the other values extracted with extracted chemicals
    df1 = pd.read_excel("data\\US_DEA_chemicals.xlsx",header=0)

    # df1.columns = df1.iloc[0]  # First row becomes header
    # df1 = df1[1:]              # Drop the first row from data
    # df1 = df1.reset_index(drop=True)

    merged_df = pd.merge(df1, df, on='Chemical Name', how='left')
    
    merged_df.to_excel("data\\merged_base_othervalues.xlsx",index=False)
    
    print("######################>>>>>>>>>>> mapping base chemical values to RR Chemicals <<<<<<<<<<<<<####################")

    mapped_df = map_othervalues_to_RR_chemicals(merged_df)
    
    mapped_df.to_excel("data\\rr_mapped_other_values.xlsx", index=False)

    print("######################>>>>>>>>>>> mapping base to RR Completed  <<<<<<<<<<<<<####################")
    
    mapped_df.to_excel("data\\rr_mapped_other_values.xlsx", index=False)
    
    # to free RAM memory 

    del(df)
    del(df1)
    del(merged_df)

    return mapped_df


def main():
    """
    Main orchestrator function for running the pipeline and evaluation.
    """
    # Build the prompt template
    prompt_template = prompt_building.build_prompt(context_example, value_format_prompt)

    # Run extraction pipeline
    mapped_df = run_extraction_pipeline(md_path, chemicals_xlsx_path, prompt_template)

    ground_truth_df = pd.read_excel("data\\US_DEA_chemicals_gt.xlsx")

    compare_columns = ["value", "type", "remark", "listedunder"]

    if ground_truth_df and mapped_df:

        print("######################>>>>>>>>>>> evaluation started  <<<<<<<<<<<<<####################")
        evaluation_results = ValuesComparision.evaluate_extraction_accuracy(ground_truth_df, mapped_df, compare_columns)

        print("######################>>>>>>>>>>> evaluation completed  <<<<<<<<<<<<<####################")

        print("[INFO] Evaluation results:", evaluation_results)


if __name__ == "__main__":
    

        # === Config === #
    md_path = "data\\US_DEA_Source_file.md"
    chemicals_xlsx_path = "data\\US_DEA_chemicals.xlsx"
    extracted_csv_path = "data\\other_value_extraction.csv"

    main()
 
 