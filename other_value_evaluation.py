import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

class ValuesComparision:

    # Normalize function
    @staticmethod
    def normalize(value):
        if pd.isnull(value):
            return ""
        value = str(value).lower().strip()
        value = value.replace("≤", "<=").replace("≥", ">=")
        value = value.replace("–", "-").replace("—", "-")
        value = re.sub(r"\s+", "", value)
        return value

    # Semantic similarity for remark
    @staticmethod
    def get_semantic_similarity(text1, text2):
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    @staticmethod
    def is_semantically_similar(text1, text2, threshold=0.65):
        return ValuesComparision.get_semantic_similarity(text1, text2) >= threshold

    # Deduplicate: pick best match for each (CAS, Chemical Name, value_ext)
    @staticmethod
    def get_best_matches(df):
        grouped = df.groupby(["CAS", "Chemical Name", "value_ext"])
        best_rows = []
        for _, group in grouped:
            if len(group) == 1:
                best_rows.append(group.iloc[0])
            else:
                best_row = max(
                    group.itertuples(index=False),
                    key=lambda row: ValuesComparision.get_semantic_similarity(str(getattr(row, "remark_ext", "")), str(getattr(row, "remark_gt", "")))
                )
                best_rows.append(pd.Series(best_row._asdict()))
        return pd.DataFrame(best_rows)

    # ✅ MAIN FUNCTION
    @staticmethod
    def evaluate_extraction_accuracy(ground_truth_df, extracted_df, compare_columns : list) -> dict:
        # compare_columns = ["value", "unit", "type", "remark", "listedunder"]

        # Load CSVs
        extracted_df = extracted_df
        ground_truth_df = ground_truth_df

        # Rename columns for merging
        extracted_df_renamed = extracted_df.rename(columns={col: f"{col}_ext" for col in compare_columns})
        ground_truth_df_renamed = ground_truth_df.rename(columns={col: f"{col}_gt" for col in compare_columns})

        # Merge
        merged_raw = pd.merge(
            extracted_df_renamed,
            ground_truth_df_renamed,
            left_on=["CAS", "Chemical Name"],
            right_on=["CAS", "Chemical Name"],
            how="inner"
        )

        # Deduplicate
        merged_df = ValuesComparision.get_best_matches(merged_raw)

        # Accuracy and mismatches
        mismatched_records = []
        accuracy_results = {}
        total = len(merged_df)

        # for col in compare_columns:
        #     correct = 0
        #     for _, row in merged_df.iterrows():
        #         val_ext = str(row.get(f"{col}_ext", "")).strip()
        #         val_gt = str(row.get(f"{col}_gt", "")).strip()

        #         is_match = (
        #             ValuesComparision.is_semantically_similar(val_ext, val_gt) if col == "remark"
        #             else ValuesComparision.normalize(val_ext) == ValuesComparision.normalize(val_gt)
        #         )

        #         if is_match:
        #             correct += 1
        #         else:
        #             mismatched_records.append({
        #                 "CAS": row["CAS"],
        #                 "Chemical Name": row["Chemical Name"],
        #                 f"{col}_ext": val_ext,
        #                 f"{col}_gt": val_gt
        #             })

        #     accuracy = (correct / total) * 100 if total > 0 else 0.0
        #     accuracy_results[col] = round(accuracy, 2)

        for col in compare_columns:
            correct = 0
            for _, row in merged_df.iterrows():
                val_ext = str(row.get(f"{col}_ext", "")).strip()
                val_gt = str(row.get(f"{col}_gt", "")).strip()

                if col == "remark":
                    is_match = ValuesComparision.is_semantically_similar(val_ext, val_gt)

                elif col == "value":
                    try:
                        int_ext = int(float(val_ext))
                        int_gt = int(float(val_gt))
                        is_match = int_ext == int_gt
                    except ValueError:
                        # fallback to normalized string comparison if conversion fails
                        is_match = ValuesComparision.normalize(val_ext) == ValuesComparision.normalize(val_gt)

                else:
                    is_match = ValuesComparision.normalize(val_ext) == ValuesComparision.normalize(val_gt)

                if is_match:
                    correct += 1
                else:
                    mismatched_records.append({
                        "CAS": row["CAS"],
                        # "Chemical Name": row["Chemical Name"],
                        f"{col}_ext": val_ext,
                        f"{col}_gt": val_gt
                    })


            accuracy = (correct / total) * 100 if total > 0 else 0.0
            accuracy_results[col] = round(accuracy, 2)


        # Save mismatches
        if mismatched_records:
            pd.DataFrame(mismatched_records).to_csv("mismatched_rows.csv", index=False)
            print("❌ Mismatched records saved to 'mismatched_rows.csv'")
        else:
            print("✅ No mismatches found.")

        # Save unmatched extracted rows
        matched_keys = merged_raw[["CAS", "Chemical Name", "value_ext"]].drop_duplicates()
        unmatched_df = extracted_df_renamed.merge(
            matched_keys,
            on=["CAS", "Chemical Name", "value_ext"],
            how="left",
            indicator=True
        ).query('_merge == "left_only"').drop(columns=['_merge'])

        if not unmatched_df.empty:
            unmatched_df.to_csv("unmatched_extracted_rows2.csv", index=False)
            print("📤 Unmatched extracted rows saved to 'unmatched_extracted_rows.csv'")
        else:
            print("✅ All extracted rows found a match.")

        # Print and return accuracy
        print("\n🔍 Per-column Accuracy (%):")
        for col, acc in accuracy_results.items():
            print(f"{col:12}: {acc}%")

        return accuracy_results


    # evaluate_extraction_accuracy("input.csv", "output.csv")
