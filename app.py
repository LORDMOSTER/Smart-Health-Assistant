import streamlit as st
import pandas as pd
import joblib
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import io
from sklearn.ensemble import GradientBoostingClassifier

# Load the pretrained model and disease info
data = {
    'symptoms': [
        'fever, cough, sore throat',
        'headache, nausea',
        'skin rash, itching',
        'fever, fatigue, joint pain'
    ],
    'disease': ['Flu', 'Migraine', 'Allergy', 'Dengue']
}

df = pd.DataFrame(data)
vectorizer = joblib.load("vectorizer.pkl")  # Use relative path
X = vectorizer.transform(df['symptoms'])
y = df['disease']

model = GradientBoostingClassifier(n_estimators=100)
model.fit(X, y)

with open("disease_info.json") as f:  # Use relative path
    disease_info = json.load(f)

# UI
st.set_page_config(page_title="Smart Health Assistant", layout="centered")
st.title("🤖 Smart Health Assistant")
st.write("Enter your details to get disease prediction and suggested medication.")

# Inputs
age = st.number_input("Age", min_value=0, max_value=120, step=1)
allergy = st.text_input("Known Allergies (comma separated)", "")
symptoms = st.text_input("Enter Symptoms (comma separated)", "")

if st.button("Predict"):
    if not symptoms:
        st.warning("Please enter symptoms.")
    else:
        symptoms_list = [s.strip().lower() for s in symptoms.split(",")]
        symptom_text = " ".join(symptoms_list)

        input_vect = vectorizer.transform([symptom_text])
        prediction = model.predict(input_vect)[0]
        st.success(f"🩺 Predicted Disease: **{prediction}**")

        allergy_list = [a.strip().lower() for a in allergy.split(",")]
        meds = disease_info.get(prediction, {}).get("medicines", {})

        # Always provide a default set of medicines
        recommended = meds.get("default", ["Paracetamol", "Ibuprofen"])  # Default medicines
        for allergen in allergy_list:
            if allergen in meds:
                recommended = meds[allergen]
                break

        st.info(f"💊 Recommended Medicines: {', '.join(recommended)}")

        # Generate PDF with enhanced UI
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("Medical Bill", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Patient Age: {age}", styles['Normal']))
        elements.append(Paragraph(f"Known Allergies: {allergy}", styles['Normal']))
        elements.append(Paragraph(f"Symptoms: {symptoms}", styles['Normal']))
        elements.append(Paragraph(f"Predicted Disease: {prediction}", styles['Normal']))
        elements.append(Spacer(1, 12))

        table_data = [["Medicine", "Dosage", "When to Take", "Before/After Meals"]]
        for med in recommended:
            table_data.append([med, "1 tablet", "3 times a day", "After meals"])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)

        doc.build(elements)

        st.download_button(
            "📄 Download Medical Bill",
            data=pdf_buffer.getvalue(),
            file_name="medical_bill.pdf",
            mime="application/pdf"
        )
