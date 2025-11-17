import polars as pl
import re
import phonenumbers
from phonenumbers import geocoder

class PhoneValidator:
    """Validate phone numbers."""
    def __init__(self, phone_col):
        self.phone_col = phone_col

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.phone_col not in df.columns:
            raise ValueError(f"Column '{self.phone_col}' not found")

        def check_number(phone_str, default_region="NP"):
            try:
                if phone_str is None or str(phone_str).strip() == "":
                    return {"valid": False, "country": None}
                cleaned = re.sub(r'[^0-9+]', '', str(phone_str))
                if cleaned.startswith('9') and not cleaned.startswith('+'):
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
            return_dtype=pl.Struct([pl.Field("valid", pl.Boolean), pl.Field("country", pl.Utf8)]),
        )
        df = df.with_columns([
            results.struct.field("valid").alias("phone_valid"),
            results.struct.field("country").alias("phone_country"),
        ])
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
    """Validate citizenship numbers."""
    def __init__(self, citizenship_col):
        self.citizenship_col = citizenship_col

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.citizenship_col not in df.columns:
            raise ValueError(f"Column '{self.citizenship_col}' not found")

        def is_valid(num):
            if not num or str(num).strip() == "":
                return {"valid": False, "reason": "Empty or null"}
            num = str(num)
            num_normalized = num.replace('/', '-')
            if re.match(r'^\d{1,2}-\d{1,2}-\d{2,3}-\d{4,7}$', num_normalized):
                parts = num_normalized.split('-')
                try:
                    district = int(parts[0])
                    year = int(parts[2])
                    if 1 <= district <= 77 and 0 <= year <= 82:
                        return {"valid": True, "reason": "Valid standard format"}
                    else:
                        return {"valid": False, "reason": "Invalid district/year"}
                except Exception:
                    return {"valid": False, "reason": "Invalid numeric parts"}
            if re.match(r'^\d{4,8}/\d{3,5}$', num):
                return {"valid": True, "reason": "Valid slash format"}
            if re.match(r'^\d{10,12}$', num):
                return {"valid": True, "reason": "Valid plain digits"}
            return {"valid": False, "reason": "Invalid format"}

        results = df[self.citizenship_col].fill_null("").map_elements(
            is_valid,
            return_dtype=pl.Struct([pl.Field("valid", pl.Boolean), pl.Field("reason", pl.Utf8)]),
        )
        df = df.with_columns([
            results.struct.field("valid").alias("citizenship_valid"),
            results.struct.field("reason").alias("citizenship_reason"),
        ])
        return df
