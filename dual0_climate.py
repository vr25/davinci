import pandas as pd
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from datasets import load_dataset
from transformers import pipeline
import pandas as pd
import sys

# Load dataset (e.g., fever-symmetric or scifact)
dataset = load_dataset("rexarski/climate_fever_fixed", split="test")  # or "scifact"

dataset_len = len(dataset)
#print("dataset length: ", len(dataset))

# Define the label mapping
label_map = {
    0: "ENTAILMENT",
    1: "CONTRADICTION",
    2: "NEUTRAL"
}

# Add the new string-based label column
dataset = dataset.map(lambda x: {"label_text": label_map[x["label"]]})

# Remove the class label feature encoding
dataset = dataset.remove_columns("label")

# Rename the column: for example, 'label_text' → 'label'
dataset = dataset.rename_column("label_text", "label")

#print(dataset[0])


# Baseline: Verification-only using entailment
verifier = pipeline("text-classification", model="microsoft/deberta-large-mnli") 

#verifier = pipeline("text-classification", model="FacebookAI/roberta-large-mnli") 

### lower the labels for this model
#verifier = pipeline("text-classification", model="facebook/bart-large-mnli") 

### lower the labels for this model
#verifier = pipeline("text-classification", model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")

def baseline_verification(claim, evidence):
    input_text = f"Claim: {claim} Evidence: {evidence}"
    result = verifier(input_text)[0]
    return result['label'], result['score']

# DAV Framework: Attribution + Verification
retriever = pipeline("question-answering", model="deepset/roberta-base-squad2-distilled")


#Span Evidence
'''
def dav_framework(claim, evidence):
    # Attribution: retrieve relevant evidence (simulated here with provided evidence)
    attributed_evidence = retriever(question=claim, context=evidence)
    
    # Verification: classify claim against attributed evidence
    input_text = input_text = f"Claim: {claim} Evidence: {attributed_evidence['answer']}"
    result = verifier(input_text)[0]
    return result['label'], result['score'], attributed_evidence['answer']
'''

#Full Evidence
'''
def dav_framework(claim, evidence):
    input_text = f"{claim} [SEP] {evidence}"
    result = verifier(input_text)[0]
    return result['label'], result['score'], evidence
'''



# DAV Framework: Attribution (full evidence) + Verification with recalibration

def dav_framework(claim, evidence, threshold=0.7):
    input_text = f"{claim} [SEP] {evidence}"
    result = verifier(input_text)[0]
    label = result['label']
    score = result['score']

    # Recalibrate: if confidence is low, default to NEI
    if score < threshold:
        label = "NEUTRAL" #"neutral"    
    return label, score, evidence



# Run experiments
results = []
for example in dataset.select(range(dataset_len)):  # small batch for demo
    claim = example['claim']
    evidence = " ".join(example['evidence']) if isinstance(example['evidence'], list) else example['evidence']
    label = example['label'] #.lower()

    base_label, base_score = baseline_verification(claim, evidence)
    dav_label, dav_score, dav_evidence = dav_framework(claim, evidence)

    results.append({
        "claim": claim,
        "true_label": label,
        "baseline_label": base_label,
        "baseline_score": base_score,
        "dav_label": dav_label,
        "dav_score": dav_score,
        "dav_evidence": dav_evidence
    })

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(df.head())
df.to_csv("dav_results_climate.csv")

# Accuracy
baseline_accuracy = (df['baseline_label'] == df['true_label']).mean()
dav_accuracy = (df['dav_label'] == df['true_label']).mean()

# Confidence
baseline_conf = df['baseline_score'].mean()
dav_conf = df['dav_score'].mean()

# Print results
print(f"\nBaseline Accuracy: {baseline_accuracy:.3f}")
print(f"DAV Accuracy: {dav_accuracy:.3f}")
print(f"Baseline Avg Confidence: {baseline_conf:.3f}")
print(f"DAV Avg Confidence: {dav_conf:.3f}")

# Classification Reports
print("\nBaseline Classification Report:")
print(classification_report(df['true_label'], df['baseline_label']))

print("\nDAV Classification Report:")
print(classification_report(df['true_label'], df['dav_label']))


'''
# Confusion Matrices
print("\nBaseline Confusion Matrix:")
print(confusion_matrix(df['true_label'], df['baseline_label'], digits=3))

print("\nDAV Confusion Matrix:")
print(confusion_matrix(df['true_label'], df['dav_label'], digits=3))

'''