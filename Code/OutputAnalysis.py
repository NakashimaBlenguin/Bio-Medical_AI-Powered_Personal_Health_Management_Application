import torch
import openai
import matplotlib.pyplot as plt
import json

# Assume we have a trained Transformer model named 'chronic_disease_model'
# The input for the model is a tensor representing the patient's features
# Here we simulate a patient's feature input as an example
patient_features = torch.randn(1, 10, 512)  # Example input tensor for a patient's data (batch_size=1, seq_length=10, feature_dim=512)

# Perform inference to get model output
# The output is a tensor representing probabilities for 5 diseases
model_output = chronic_disease_model(patient_features)  # Run the patient features through your Transformer model
model_output = torch.sigmoid(model_output.squeeze())  # Apply sigmoid to get probabilities and remove batch dimension if needed

# Patient's Electronic Health Record (EHR)
patient_ehr = {
    "patient_id": "patient-12345",
    "visit_history": [
        {"date": "2023-01-01", "blood_pressure": "130/85", "blood_sugar": 100, "bmi": 25.3, "diagnosis": "Hypertension"},
        {"date": "2023-03-01", "blood_pressure": "135/88", "blood_sugar": 110, "bmi": 26.1, "diagnosis": "Hypertension"},
    ],
    "medications": ["Lisinopril", "Metformin"]
}

# Set OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Prepare input data for explanation
input_data = {
    "model_output": model_output.tolist(),  # Use the actual model output here
    "patient_ehr": patient_ehr
}
input_prompt = f"Please explain the health condition of the following patient: {json.dumps(input_data)}"

# Call OpenAI API to get explanation
def get_openai_explanation(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

explanation = get_openai_explanation(input_prompt)
print("Explanation of the model output:")
print(explanation)

# Plot chart
probabilities = model_output.tolist()
diseases = ["Diabetes", "Hypertension", "Cardiovascular Disease", "COPD", "Chronic Kidney Disease"]

plt.figure(figsize=(10, 6))
plt.bar(diseases, probabilities, color='skyblue')
plt.xlabel('Chronic Diseases')
plt.ylabel('Predicted Probability')
plt.title('Predicted Risk for Chronic Diseases')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--')
plt.show()

# Get suggestions based on the explanation
suggestion_prompt = f"Based on the following explanation of the patient's health condition, provide reasonable health suggestions: {explanation}"
suggestion = get_openai_explanation(suggestion_prompt)
print("Health Suggestions:")
print(suggestion)
