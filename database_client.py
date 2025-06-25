from pymongo import MongoClient
from collections import defaultdict
import pprint
import pandas as pd


class db_data_extraction:

    def get_section_md_data_for_list_id(list_id):
        
        mongo_uri = "mongodb://172.203.242.247:27017"  # or from MongoDB Atlas
        client = MongoClient(mongo_uri) # Update with your URI

        db = client["chemadvisor-qa"]
        collection = db["section_data"]

        results = list(collection.find({
            "sections_metadata.ListID": list_id}))

        for doc in results:
            print(f"\n📄 File: {doc.get('file_name')}")
            matched_sections = [ sec for sec in doc.get("sections_metadata", []) if sec.get("ListID") == list_id ]

        md_data = matched_sections[0]['matched_section']
        
        return md_data


    def get_ground_truth_data(list_id):
        
        mongo_uri = "mongodb://172.203.242.247:27017"  # or from MongoDB Atlas
        client = MongoClient(mongo_uri) # Update with your URI

        db = client["chemadvisor-qa"]
        collection_ld = db["list_data"]

        results = list(collection_ld.find({"ListID": list_id}))

        df = pd.DataFrame(results[0]['list_chemicals'])

        base_df = df[~df["CAS"].str.startswith("RR-")].copy()
        extracted_chemicals = base_df[['CAS','Chemical Name']]

        return base_df, extracted_chemicals