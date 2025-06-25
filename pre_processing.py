import re
from markdownify import markdownify as md
from bs4 import BeautifulSoup
import logging
import traceback
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from flashtext import KeywordProcessor
from database_client import db_data_extraction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Pre_processing:

    @staticmethod
    def remove_html_comments(md_content):
        # This regex removes everything from <!-- to -->
        return re.sub(r'<!--.*?-->', '', md_content, flags=re.DOTALL)

    @staticmethod
    def convert_html_tables_to_markdowns(md_content):
        # Remove HTML comments
        md_content = Pre_processing.remove_html_comments(md_content)

        new_md_content = md_content
        try:
            soup = BeautifulSoup(md_content, "html.parser")
            for table in soup.find_all("table"):
                html_str = str(table)
                markdown_table = md(html_str).strip()

                if html_str in new_md_content:
                    new_md_content = new_md_content.replace(html_str, markdown_table)
                else:
                    start_index = new_md_content.find("<table")
                    end_index = new_md_content.find("</table>", start_index) + len("</table>")
                    if start_index != -1 and end_index != -1:
                        new_md_content = (
                            new_md_content[:start_index]
                            + markdown_table
                            + new_md_content[end_index:]
                        )
        except Exception as e:
            logger.error(f"Error converting HTML tables to markdown: {str(e)}")
            logger.error(traceback.format_exc())

        return new_md_content.lstrip().rstrip()

    @staticmethod
    def creat_chunks(list_id, chunk_size = 2000):
        # load the file and chunking it 

        # loader = TextLoader(md_path,encoding='utf-8')

        # md_content = loader.load()
        
        # md_content = md_content[0].page_content
        md_content = db_data_extraction.get_section_md_data_for_list_id(list_id)

        md_content = Pre_processing.convert_html_tables_to_markdowns(md_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap=200)

        chunks = text_splitter.split_text(md_content)

        return chunks
    
    @staticmethod
    def add_extracted_chemicas_to_flashtext(extracted_chemicals_df):

        # path_xlsx = "data\\US_DEA_chemicals.xlsx"

        keyword_processor = KeywordProcessor(case_sensitive=False)

        # df = pd.read_excel(path_xlsx, header=0)
        df = extracted_chemicals_df

        extracted_chemicals = df['Chemical Name'].values.tolist()

        for chemical in extracted_chemicals:
            keyword_processor.add_keyword( chemical )

        return keyword_processor 
    
    @staticmethod
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
        
        if rr_df.shape[0]>0:
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
        else:
            final_df = base_df

        # # Save output
        # final_df.to_csv("output_with_base_data1.csv", index=False)
        # print(final_df)

        return final_df

