import dspy
from typing import Iterable, List


# Configure DSPy with your remote lightweight LLM endpoint.
llm = dspy.LM(
    model="llama3.1:latest",
    base_url="https://jo3m4y06rnnwhaz.askbhunte.com",
    api_key="",
)

dspy.configure(lm=llm)


class StandardizeHeader(dspy.Signature):
    """Standardize a given CSV header name to a simpler, more usable format."""

    original_header = dspy.InputField(
        desc="The original, potentially complex, CSV header string"
    )
    standardized_header = dspy.OutputField(
        desc=(
            "A simplified and standardized version of the header. "
            "Examples: 'General Questions/Municipality and Ward Details/Name of Municipality (नगरपालिकाको नाम)' "
            "-> 'municipality_name', 'General Questions/_GPS Coordinates_latitude' -> 'latitude', 'name' -> 'name'"
        )
    )


class HeaderStandardizer(dspy.Module):
    """DSPy module that predicts a standardized header name."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(StandardizeHeader)

    def forward(self, original_header: str) -> str:
        prediction = self.predictor(original_header=original_header)
        return prediction.standardized_header


_header_standardizer = HeaderStandardizer()


def standardize_headers(headers: Iterable[str]) -> List[str]:
    """Standardize a list of headers using the global DSPy header standardizer."""
    non_empty_headers = [h for h in headers if h and str(h).strip()]
    result: List[str] = []

    for header in non_empty_headers:
        standardized_name = _header_standardizer(original_header=str(header))
        result.append(standardized_name)

    return result


