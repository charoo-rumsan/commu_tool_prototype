import re
from typing import Dict, Optional, Tuple

import pandas as pd
import polars as pl
import phonenumbers
from phonenumbers import geocoder


class PhoneValidator:
    """Validate and enrich phone-number columns."""

    def __init__(self, phone_col: str):
        self.phone_col = phone_col

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.phone_col not in df.columns:
            raise ValueError(f"Column '{self.phone_col}' not found")

        def check_number(phone_str, default_region: str = "NP"):
            try:
                if phone_str is None or str(phone_str).strip() == "":
                    return {"valid": False, "country": None}

                cleaned = re.sub(r"[^0-9+]", "", str(phone_str))

                if cleaned.startswith("9") and not cleaned.startswith("+"):
                    cleaned = f"+977{cleaned}"

                num = phonenumbers.parse(cleaned, default_region)

                if not phonenumbers.is_valid_number(num):
                    return {"valid": False, "country": None}

                country = geocoder.description_for_number(num, "en")

                return {"valid": True, "country": country or "Unknown"}
            except Exception:
                return {"valid": False, "country": None}

        results = df[self.phone_col].fill_null("").map_elements(
            check_number,
            return_dtype=pl.Struct(
                [pl.Field("valid", pl.Boolean), pl.Field("country", pl.Utf8)]
            ),
        )

        df = df.with_columns(
            [
                results.struct.field("valid").alias("phone_valid"),
                results.struct.field("country").alias("phone_country"),
            ]
        )

        df = df.with_columns(
            pl.when(pl.col("phone_valid"))
            .then(
                pl.when(pl.col("phone_country") == "Nepal")
                .then(pl.lit("nepali"))
                .otherwise(pl.lit("international"))
            )
            .otherwise(pl.lit("invalid"))
            .alias("phone_type")
        )

        return df


class CitizenshipValidator:
    """Validate citizenship-number columns."""

    def __init__(self, citizenship_col: str):
        self.citizenship_col = citizenship_col

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.citizenship_col not in df.columns:
            raise ValueError(f"Column '{self.citizenship_col}' not found")

        def is_valid(num):
            if not num or str(num).strip() == "":
                return {"valid": False, "reason": "Empty or null"}

            num = str(num)
            num_normalized = num.replace("/", "-")

            if re.match(r"^\d{1,2}-\d{1,2}-\d{2,3}-\d{4,7}$", num_normalized):
                parts = num_normalized.split("-")
                try:
                    district = int(parts[0])
                    year = int(parts[2])
                    if 1 <= district <= 77 and 0 <= year <= 82:
                        return {"valid": True, "reason": "Valid standard format"}
                    return {"valid": False, "reason": "Invalid district/year"}
                except Exception:
                    return {"valid": False, "reason": "Invalid numeric parts"}

            if re.match(r"^\d{4,8}/\d{3,5}$", num):
                return {"valid": True, "reason": "Valid slash format"}

            if re.match(r"^\d{10,12}$", num):
                return {"valid": True, "reason": "Valid plain digits"}

            return {"valid": False, "reason": "Invalid format"}

        results = df[self.citizenship_col].fill_null("").map_elements(
            is_valid,
            return_dtype=pl.Struct(
                [pl.Field("valid", pl.Boolean), pl.Field("reason", pl.Utf8)]
            ),
        )

        df = df.with_columns(
            [
                results.struct.field("valid").alias("citizenship_valid"),
                results.struct.field("reason").alias("citizenship_reason"),
            ]
        )

        return df


def _summarize_phone(df: pl.DataFrame) -> Dict:
    total_rows = df.height
    valid_count = df.select(pl.col("phone_valid").sum().alias("valid_sum")).to_series()[0]
    valid_count = int(valid_count or 0)

    country_breakdown = df.group_by("phone_country", maintain_order=True).len()
    type_breakdown = df.group_by("phone_type", maintain_order=True).len()

    return {
        "total_rows": total_rows,
        "valid_rows": valid_count,
        "invalid_rows": total_rows - valid_count,
        "country_breakdown": country_breakdown.to_dicts(),
        "type_breakdown": type_breakdown.to_dicts(),
    }


def _summarize_citizenship(df: pl.DataFrame) -> Dict:
    total_rows = df.height
    valid_count = (
        df.select(pl.col("citizenship_valid").sum().alias("valid_sum")).to_series()[0]
    )
    valid_count = int(valid_count or 0)

    reason_breakdown = df.group_by("citizenship_reason", maintain_order=True).len()

    return {
        "total_rows": total_rows,
        "valid_rows": valid_count,
        "invalid_rows": total_rows - valid_count,
        "reason_breakdown": reason_breakdown.to_dicts(),
    }


def run_validations(
    df: pd.DataFrame,
    phone_col: Optional[str] = None,
    citizenship_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Run phone/citizenship validations on a pandas DataFrame.

    Returns:
        A tuple of (pandas_df_with_new_columns, summary_dict)
    """
    if not phone_col and not citizenship_col:
        return df, {}

    pl_df = pl.from_pandas(df)
    summary: Dict[str, Dict] = {}

    if phone_col and phone_col in pl_df.columns:
        phone_validator = PhoneValidator(phone_col)
        pl_df = phone_validator.transform(pl_df)
        summary["phone_validation"] = _summarize_phone(pl_df)

    if citizenship_col and citizenship_col in pl_df.columns:
        citizenship_validator = CitizenshipValidator(citizenship_col)
        pl_df = citizenship_validator.transform(pl_df)
        summary["citizenship_validation"] = _summarize_citizenship(pl_df)

    return pl_df.to_pandas(), summary

