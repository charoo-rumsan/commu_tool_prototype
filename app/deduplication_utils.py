import re
import pandas as pd
from fuzzywuzzy import fuzz
from typing import Dict, List, Tuple, Set


class Deduplicator:
    """Handles deduplication of rows based on specified columns."""

    def __init__(self, threshold: int = 85):
        # Similarity threshold for fuzzy match (0–100)
        self.threshold = threshold

    # -------------------------------------------------
    # 🔹 Normalization helpers
    # -------------------------------------------------
    def _normalize_phone_number(self, phone: str) -> str:
        """Normalize phone numbers (remove all non-digits)."""
        if not isinstance(phone, str):
            phone = str(phone)
        return re.sub(r"\D", "", phone)

    def _normalize_citizenship_number(self, number: str) -> str:
        """Normalize citizenship numbers (remove special chars, uppercase)."""
        if not isinstance(number, str):
            number = str(number)
        return re.sub(r"[^a-zA-Z0-9]", "", number).upper()

    # -------------------------------------------------
    # 🔹 Prefix extraction
    # -------------------------------------------------
    def _extract_prefix(self, value: str, length: int = 3) -> str:
        """Extract a prefix of given length for efficient grouping."""
        if not isinstance(value, str):
            value = str(value)
        return value[:length] if len(value) >= length else value

    # -------------------------------------------------
    # 🔹 Duplicate detection logic
    # -------------------------------------------------
    def _find_duplicates_in_batch(
        self, df: pd.DataFrame, column: str, column_type: str
    ) -> List[Set[int]]:
        """Find groups of duplicate values within a batch."""
        duplicate_groups = []
        processed = set()

        # Normalize column values
        if column_type == "phone":
            df[f"{column}_normalized"] = df[column].apply(self._normalize_phone_number)
        elif column_type == "citizenship":
            df[f"{column}_normalized"] = df[column].apply(self._normalize_citizenship_number)
        else:
            df[f"{column}_normalized"] = df[column].astype(str)

        # Group by exact matches first
        exact_duplicates = df[df[f"{column}_normalized"].duplicated(keep=False)]
        if not exact_duplicates.empty:
            for _, group in exact_duplicates.groupby(f"{column}_normalized"):
                duplicate_groups.append(set(group.index))
                processed.update(group.index)

        # Fuzzy match remaining unprocessed values
        remaining = df[~df.index.isin(processed)]
        n = len(remaining)

        for i in range(n):
            if remaining.index[i] in processed:
                continue

            current_group = {remaining.index[i]}
            current_val = remaining.iloc[i][f"{column}_normalized"]

            for j in range(i + 1, n):
                if remaining.index[j] in processed:
                    continue

                compare_val = remaining.iloc[j][f"{column}_normalized"]
                similarity = fuzz.ratio(current_val, compare_val)

                if similarity >= self.threshold:
                    current_group.add(remaining.index[j])

            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                processed.update(current_group)

        return duplicate_groups

    # -------------------------------------------------
    # 🔹 Main public function
    # -------------------------------------------------
    def deduplicate_dataframe(
        self, df: pd.DataFrame, columns: List[Tuple[str, str]], batch_size: int = 1000
    ) -> Dict:
        """
        Deduplicate a DataFrame based on given columns.

        Args:
            df: Pandas DataFrame containing the data.
            columns: List of (column_name, column_type) tuples.
                     Example: [("phone_number", "phone"), ("citizenship_number", "citizenship")]
            batch_size: Size of each processing batch (default 1000).

        Returns:
            Dict containing duplicate groups and deduplication statistics.
        """
        results = {}
        stats = {
            "total_rows": len(df),
            "duplicate_groups": {},
            "unique_values": {},
        }

        for column, column_type in columns:
            if column not in df.columns:
                continue

            # Normalize and prefix-group
            norm_col = f"{column}_normalized"
            if column_type == "phone":
                df[norm_col] = df[column].apply(self._normalize_phone_number)
            elif column_type == "citizenship":
                df[norm_col] = df[column].apply(self._normalize_citizenship_number)
            else:
                df[norm_col] = df[column].astype(str)

            prefix_col = f"{column}_prefix"
            df[prefix_col] = df[norm_col].apply(lambda x: self._extract_prefix(x, 3))

            duplicate_groups = []

            # Process each prefix group in batches
            for _, group_df in df.groupby(prefix_col):
                for start in range(0, len(group_df), batch_size):
                    end = min(start + batch_size, len(group_df))
                    batch_df = group_df.iloc[start:end].copy()
                    batch_duplicates = self._find_duplicates_in_batch(batch_df, column, column_type)

                    # Adjust indices relative to original DataFrame
                    adjusted_groups = [
                        {batch_df.index[idx] for idx in group}
                        for group in batch_duplicates
                    ]
                    duplicate_groups.extend(adjusted_groups)

            results[column] = duplicate_groups
            stats["duplicate_groups"][column] = len(duplicate_groups)
            stats["unique_values"][column] = len(df) - sum(len(g) - 1 for g in duplicate_groups)

        return {"results": results, "statistics": stats}
