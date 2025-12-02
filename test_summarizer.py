from summarizer import load_summarizer_model, summarize

tok, mdl = load_summarizer_model("models/finalized_m2_reduced_lead3.sav")

text = """
Machine learning models are used to find patterns in data.
They require training data and validation.
These models can be applied to classification or prediction tasks.
"""

print("SUMMARY:", summarize(text, tok, mdl))
