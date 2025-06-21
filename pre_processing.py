import re
from markdownify import markdownify as md
from bs4 import BeautifulSoup
import logging
import traceback
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from flashtext import KeywordProcessor

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
    def creat_chunks(md_path, chunk_size = 2000):
        # load the file and chunking it 

        loader = TextLoader(md_path,encoding='utf-8')

        md_content = loader.load()
        
        md_content = md_content[0].page_content

        md_content = Pre_processing.convert_html_tables_to_markdowns(md_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap=200)

        chunks = text_splitter.split_text(md_content)

        return chunks
    
    @staticmethod
    def add_extracted_chemicas_to_flashtext(path_xlsx):

        # path_xlsx = "data\\US_DEA_chemicals.xlsx"

        keyword_processor = KeywordProcessor(case_sensitive=False)

        df = pd.read_excel(path_xlsx, header=0)

        extracted_chemicals = df['Chemical Name'].values.tolist()

        for chemical in extracted_chemicals:
            keyword_processor.add_keyword( chemical )

        return keyword_processor 
