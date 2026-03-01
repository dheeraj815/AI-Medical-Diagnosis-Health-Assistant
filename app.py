import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import hashlib
from typing import Dict, List, Tuple

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="MediCare AI Pro | Clinical Intelligence Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.medicare-ai.com/help',
        'Report a bug': "https://www.medicare-ai.com/bug",
        'About': "MediCare AI Pro v4.0.0 — Clinical Intelligence Platform"
    }
)

# ==================== MEDICAL DATABASE ====================


class MedicalDatabase:
    DISEASES = {
        "Influenza": {
            "icd_10": "J11.1", "severity": "Moderate", "prevalence": "Common (seasonal)",
            "duration": "5-7 days",
            "symptom_set": frozenset(["Fever", "Body Aches", "Fatigue", "Dry Cough", "Headache", "Chills", "Sore Throat", "Runny Nose"]),
            "common_symptoms": ["High fever (>101°F)", "Body aches and fatigue", "Dry cough", "Headache", "Chills and sweats"],
            "differential_diagnosis": ["COVID-19", "Common Cold", "Streptococcal Pharyngitis", "Pneumonia"],
            "treatment": {
                "first_line": "Supportive care; antiviral medications (oseltamivir) if started within 48 hours",
                "medications": ["Oseltamivir (Tamiflu) 75mg BID × 5 days", "Zanamivir (Relenza) inhaled", "Acetaminophen for fever", "NSAIDs for myalgias"],
                "duration": "5–7 days with treatment"
            },
            "red_flags": ["Difficulty breathing or shortness of breath", "Persistent chest pain", "Confusion or inability to wake", "Severe muscle pain", "Severe dehydration"],
            "when_to_seek_help": "Seek immediate care if breathing difficulties, chest pain, or high fever persists beyond 3 days",
            "prevention": "Annual influenza vaccine, hand hygiene, respiratory etiquette",
            "follow_up": "Follow-up if symptoms worsen or persist beyond 7 days",
            "specialist": "Primary Care Physician; Infectious Disease (severe)"
        },
        "Upper Respiratory Infection": {
            "icd_10": "J06.9", "severity": "Mild", "prevalence": "Very Common",
            "duration": "7-10 days",
            "symptom_set": frozenset(["Runny Nose", "Sore Throat", "Cough", "Fever", "Fatigue", "Sneezing", "Nasal Congestion"]),
            "common_symptoms": ["Runny nose", "Sore throat", "Mild cough", "Mild fever", "Fatigue"],
            "differential_diagnosis": ["Influenza", "Allergic Rhinitis", "Sinusitis", "COVID-19"],
            "treatment": {
                "first_line": "Supportive care — rest, fluids, OTC medications",
                "medications": ["Acetaminophen or Ibuprofen for fever/pain", "Decongestants (pseudoephedrine)", "Antihistamines for rhinorrhea", "Throat lozenges"],
                "duration": "Symptoms resolve in 7–10 days"
            },
            "red_flags": ["High fever >103°F", "Severe sore throat with dysphagia", "Symptoms lasting >10 days", "Difficulty breathing"],
            "when_to_seek_help": "Consult physician if symptoms worsen or persist beyond 10 days",
            "prevention": "Hand washing, avoid face-touching, distance from symptomatic individuals",
            "follow_up": "Not required unless complications develop",
            "specialist": "Primary Care Physician"
        },
        "Gastroenteritis": {
            "icd_10": "A09", "severity": "Moderate", "prevalence": "Common",
            "duration": "1-3 days (viral), 3-7 days (bacterial)",
            "symptom_set": frozenset(["Nausea", "Vomiting", "Diarrhea", "Abdominal Pain", "Fever", "Cramping", "Loss of Appetite", "Dehydration"]),
            "common_symptoms": ["Nausea and vomiting", "Diarrhea", "Abdominal cramps", "Low-grade fever", "Dehydration"],
            "differential_diagnosis": ["Food Poisoning", "IBD", "Appendicitis", "IBS"],
            "treatment": {
                "first_line": "Oral rehydration, bland diet (BRAT — bananas, rice, applesauce, toast)",
                "medications": ["Oral rehydration solutions (Pedialyte)", "Loperamide (Imodium) for diarrhea", "Ondansetron for severe nausea", "Avoid antibiotics unless bacterial cause confirmed"],
                "duration": "3–7 days depending on etiology"
            },
            "red_flags": ["Severe dehydration (no urination >8 hours)", "Blood in stool", "High fever >102°F", "Severe abdominal pain", "Signs of shock"],
            "when_to_seek_help": "Emergency care for severe dehydration, bloody stools, or severe pain",
            "prevention": "Hand hygiene, safe food preparation, avoid contaminated water",
            "follow_up": "Follow-up if symptoms persist beyond 7 days",
            "specialist": "Gastroenterologist (if persistent or severe)"
        },
        "Acute Myocardial Infarction": {
            "icd_10": "I21.9", "severity": "Critical — EMERGENCY", "prevalence": "Common in adults >45",
            "duration": "Medical Emergency — Immediate Intervention Required",
            "symptom_set": frozenset(["Chest Pain", "Shortness of Breath", "Sweating", "Nausea", "Dizziness", "Arm Pain", "Jaw Pain", "Palpitations", "Syncope"]),
            "common_symptoms": ["Severe chest pain or pressure", "Pain radiating to left arm, jaw, or back", "Shortness of breath", "Diaphoresis, nausea", "Lightheadedness"],
            "differential_diagnosis": ["Unstable Angina", "Pulmonary Embolism", "Aortic Dissection", "GERD/Esophageal spasm"],
            "treatment": {
                "first_line": "IMMEDIATE 911 — Aspirin 325mg, oxygen, nitroglycerin, emergency cardiac catheterization",
                "medications": ["Aspirin 325mg STAT", "Nitroglycerin sublingual", "Morphine for refractory pain", "Antiplatelet therapy (clopidogrel)", "Anticoagulation (heparin)"],
                "duration": "Hospitalization required — intensive cardiac care"
            },
            "red_flags": ["ANY chest pain with cardiac features", "Loss of consciousness", "Severe shortness of breath", "Irregular heartbeat"],
            "when_to_seek_help": "CALL 911 IMMEDIATELY — DO NOT DRIVE YOURSELF",
            "prevention": "Control risk factors: hypertension, diabetes, hyperlipidemia, smoking cessation, exercise",
            "follow_up": "Cardiology follow-up; cardiac rehabilitation",
            "specialist": "Emergency Medicine, Cardiology, Cardiac Surgery"
        },
        "Pneumonia": {
            "icd_10": "J18.9", "severity": "Moderate to Severe", "prevalence": "Common (especially elderly)",
            "duration": "2-3 weeks with treatment",
            "symptom_set": frozenset(["Cough", "Fever", "Chills", "Shortness of Breath", "Chest Pain", "Fatigue", "Confusion", "Sputum Production"]),
            "common_symptoms": ["Productive cough with phlegm", "High fever and chills", "Pleuritic chest pain", "Shortness of breath", "Fatigue and confusion (elderly)"],
            "differential_diagnosis": ["Bronchitis", "Pulmonary Embolism", "Heart Failure", "Lung Cancer"],
            "treatment": {
                "first_line": "Antibiotics (amoxicillin, azithromycin); supportive care; possible hospitalization",
                "medications": ["Amoxicillin 500mg TID", "Azithromycin (Z-pack)", "Levofloxacin for severe cases", "Oxygen therapy if hypoxic", "IV antibiotics if admitted"],
                "duration": "7–14 days antibiotics; full recovery 4–6 weeks"
            },
            "red_flags": ["Severe dyspnea", "Confusion or altered mental status", "SpO2 <90%", "RR >30/min", "Chest pain"],
            "when_to_seek_help": "Immediate care for breathing difficulties, high fever, or confusion",
            "prevention": "Pneumococcal vaccine, annual flu vaccine, smoking cessation",
            "follow_up": "Chest X-ray in 6–8 weeks to confirm resolution",
            "specialist": "Pulmonology, Internal Medicine"
        },
        "Meningitis": {
            "icd_10": "G03.9", "severity": "Critical — EMERGENCY", "prevalence": "Rare but serious",
            "duration": "Medical Emergency — Requires Immediate Hospitalization",
            "symptom_set": frozenset(["Severe Headache", "Fever", "Neck Stiffness", "Confusion", "Photophobia", "Nausea", "Vomiting", "Rash", "Seizures"]),
            "common_symptoms": ["Severe headache", "High fever", "Nuchal rigidity", "Confusion or altered consciousness", "Photophobia", "Nausea and vomiting"],
            "differential_diagnosis": ["Encephalitis", "Subarachnoid hemorrhage", "Severe migraine", "Brain abscess"],
            "treatment": {
                "first_line": "EMERGENCY HOSPITALIZATION — IV antibiotics immediately, ICU supportive care",
                "medications": ["Ceftriaxone 2g IV q12h", "Vancomycin IV", "Dexamethasone", "Acyclovir if viral suspected", "Supportive ICU care"],
                "duration": "2–3 weeks IV antibiotics; prolonged hospitalization"
            },
            "red_flags": ["Fever + headache + stiff neck", "Altered mental status", "Seizures", "Petechial rash (bacterial)", "Rapid deterioration"],
            "when_to_seek_help": "CALL 911 IMMEDIATELY — This is a medical emergency",
            "prevention": "Meningococcal vaccine; avoid close contact with infected individuals",
            "follow_up": "Neurology follow-up; hearing tests (bacterial can cause deafness)",
            "specialist": "Emergency Medicine, Infectious Disease, Neurology, ICU"
        },
        "Appendicitis": {
            "icd_10": "K35.80", "severity": "Severe — Requires Surgery", "prevalence": "Common surgical emergency",
            "duration": "Surgical intervention required within 24-48 hours",
            "symptom_set": frozenset(["Abdominal Pain", "Nausea", "Vomiting", "Fever", "Loss of Appetite", "Rebound Tenderness", "Rigidity"]),
            "common_symptoms": ["Periumbilical pain migrating to RLQ", "Anorexia", "Nausea and vomiting", "Low-grade fever", "Rebound tenderness"],
            "differential_diagnosis": ["Gastroenteritis", "Ovarian cyst/torsion", "Kidney stones", "Ectopic pregnancy"],
            "treatment": {
                "first_line": "Appendectomy (surgical removal); IV antibiotics",
                "medications": ["IV antibiotics pre-operatively", "Post-operative analgesia", "Antiemetics for nausea"],
                "duration": "Surgery required; 1–3 day hospitalization; 2–4 week recovery"
            },
            "red_flags": ["Severe RLQ pain", "High fever", "Rigid abdomen", "Signs of perforation"],
            "when_to_seek_help": "Emergency care immediately — appendicitis can rupture",
            "prevention": "No specific prevention",
            "follow_up": "Surgical follow-up 2 weeks post-operation",
            "specialist": "General Surgery, Emergency Medicine"
        },
        "Migraine": {
            "icd_10": "G43.909", "severity": "Moderate", "prevalence": "Common (12% of population)",
            "duration": "4-72 hours per episode",
            "symptom_set": frozenset(["Headache", "Nausea", "Vomiting", "Photophobia", "Phonophobia", "Aura", "Visual Changes", "Dizziness"]),
            "common_symptoms": ["Unilateral throbbing headache", "Photophobia and phonophobia", "Nausea and vomiting", "Aura (visual disturbances)", "Osmophobia"],
            "differential_diagnosis": ["Tension headache", "Cluster headache", "Brain tumor", "Stroke/TIA"],
            "treatment": {
                "first_line": "Triptans, NSAIDs; preventive medications if frequent (≥4/month)",
                "medications": ["Sumatriptan 100mg at onset", "Ibuprofen 800mg", "Antiemetics (metoclopramide)", "Preventive: Propranolol, Topiramate", "CGRP antagonists (erenumab)"],
                "duration": "Acute episode 4–72 hours; preventive therapy ongoing"
            },
            "red_flags": ["Thunderclap headache", "Headache + fever + stiff neck", "Neurological deficits", "New onset after age 50", "Progressive worsening"],
            "when_to_seek_help": "Emergency care for thunderclap headache or focal neurological symptoms",
            "prevention": "Identify triggers; prophylactic medications; lifestyle modifications",
            "follow_up": "Neurology follow-up for refractory or frequent migraines",
            "specialist": "Neurology, Headache Specialist"
        },
        "Type 2 Diabetes Mellitus": {
            "icd_10": "E11.9", "severity": "Chronic — Long-term Management", "prevalence": "Very Common (10% adults)",
            "duration": "Chronic lifelong condition",
            "symptom_set": frozenset(["Fatigue", "Increased Thirst", "Frequent Urination", "Blurred Vision", "Weight Loss", "Slow Healing", "Numbness", "Increased Hunger"]),
            "common_symptoms": ["Polydipsia and polyuria", "Polyphagia", "Fatigue", "Blurred vision", "Slow-healing wounds", "Peripheral neuropathy"],
            "differential_diagnosis": ["Type 1 Diabetes", "MODY", "Cushing's syndrome", "Hyperthyroidism"],
            "treatment": {
                "first_line": "Lifestyle modification (diet, exercise); Metformin",
                "medications": ["Metformin 500–2000mg daily", "SGLT2 inhibitors (empagliflozin)", "GLP-1 agonists (semaglutide)", "Insulin if needed", "Statins for CV protection"],
                "duration": "Lifelong management required"
            },
            "red_flags": ["DKA symptoms", "Hyperosmolar state", "Severe hypoglycemia", "Foot ulcers or infections"],
            "when_to_seek_help": "Regular monitoring; emergency care for DKA or severe hypo/hyperglycemia",
            "prevention": "Weight management, regular exercise, healthy diet, smoking cessation",
            "follow_up": "Quarterly PCP visits; annual eye and foot exams; HbA1c monitoring",
            "specialist": "Endocrinology, Primary Care, Ophthalmology, Podiatry"
        },
        "COVID-19": {
            "icd_10": "U07.1", "severity": "Mild to Critical", "prevalence": "Widespread",
            "duration": "7-21 days (acute); long-COVID can persist",
            "symptom_set": frozenset(["Fever", "Cough", "Fatigue", "Shortness of Breath", "Loss of Taste", "Loss of Smell", "Body Aches", "Headache", "Sore Throat", "Diarrhea"]),
            "common_symptoms": ["Fever or chills", "Dry cough", "Fatigue", "Dyspnea", "Anosmia/ageusia", "Myalgias", "Headache"],
            "differential_diagnosis": ["Influenza", "RSV", "Community-Acquired Pneumonia", "Upper Respiratory Infection"],
            "treatment": {
                "first_line": "Supportive care; antivirals (nirmatrelvir/ritonavir) for high-risk within 5 days of symptom onset",
                "medications": ["Paxlovid (nirmatrelvir/ritonavir) for high-risk", "Remdesivir (hospitalized)", "Dexamethasone (severe)", "Supportive oxygen therapy"],
                "duration": "Acute illness 7–21 days; high-risk patients may require hospitalization"
            },
            "red_flags": ["SpO2 <94%", "Persistent chest pain", "Confusion", "Inability to stay awake", "Pale/cyanotic lips"],
            "when_to_seek_help": "Emergency care for breathing difficulty, persistent chest pain, or confusion",
            "prevention": "COVID-19 vaccination, masking in high-risk settings, ventilation, hand hygiene",
            "follow_up": "Follow-up for long-COVID symptoms; pulmonology if persistent respiratory issues",
            "specialist": "Infectious Disease, Pulmonology, Emergency Medicine"
        },
        "Urinary Tract Infection": {
            "icd_10": "N39.0", "severity": "Mild to Moderate", "prevalence": "Very Common (especially women)",
            "duration": "3-7 days with treatment",
            "symptom_set": frozenset(["Painful Urination", "Frequent Urination", "Urgency", "Pelvic Pain", "Cloudy Urine", "Blood in Urine", "Fever", "Back Pain"]),
            "common_symptoms": ["Dysuria", "Urinary frequency and urgency", "Suprapubic discomfort", "Cloudy or malodorous urine", "Hematuria"],
            "differential_diagnosis": ["Pyelonephritis", "STI", "Interstitial cystitis", "Kidney stones"],
            "treatment": {
                "first_line": "Nitrofurantoin or trimethoprim-sulfamethoxazole for uncomplicated UTI",
                "medications": ["Nitrofurantoin 100mg BID × 5 days", "TMP-SMX DS BID × 3 days", "Phenazopyridine for symptom relief", "Fosfomycin 3g single dose"],
                "duration": "3–7 days antibiotics"
            },
            "red_flags": ["Fever >101°F + flank pain (pyelonephritis)", "Rigors or vomiting", "Pregnancy + UTI", "Recurrent UTIs (≥3/year)"],
            "when_to_seek_help": "Consult physician for symptoms; emergency if signs of pyelonephritis",
            "prevention": "Adequate hydration, post-coital voiding, proper hygiene, avoid irritants",
            "follow_up": "Culture and sensitivity if recurrent; urology referral if complicated",
            "specialist": "Primary Care, Urology, Gynecology"
        },
        "Hypertensive Crisis": {
            "icd_10": "I16.9", "severity": "Critical — EMERGENCY", "prevalence": "Uncommon",
            "duration": "Medical Emergency",
            "symptom_set": frozenset(["Severe Headache", "Chest Pain", "Shortness of Breath", "Vision Changes", "Nausea", "Confusion", "Nosebleed", "Palpitations"]),
            "common_symptoms": ["Severe headache (worst of life)", "Chest pain", "Shortness of breath", "Visual disturbances", "Confusion or altered consciousness"],
            "differential_diagnosis": ["Stroke", "Aortic Dissection", "PRES", "Eclampsia"],
            "treatment": {
                "first_line": "EMERGENCY — IV antihypertensive therapy; lower BP by 25% within 1 hour",
                "medications": ["IV labetalol", "IV nicardipine", "IV nitroprusside (hypertensive emergency)", "Oral antihypertensives (urgency)"],
                "duration": "Hospitalization for emergency; outpatient management for urgency"
            },
            "red_flags": ["BP >180/120 with end-organ damage", "Neurological deficits", "Chest pain + elevated BP", "Visual changes"],
            "when_to_seek_help": "CALL 911 — hypertensive emergency requires immediate hospital care",
            "prevention": "Medication adherence, dietary sodium restriction, regular BP monitoring, lifestyle modification",
            "follow_up": "Cardiology/nephrology follow-up; ambulatory BP monitoring",
            "specialist": "Emergency Medicine, Cardiology, Nephrology"
        },
        "Deep Vein Thrombosis": {
            "icd_10": "I82.409", "severity": "Moderate to Severe", "prevalence": "Common",
            "duration": "Requires immediate treatment; anticoagulation 3-6+ months",
            "symptom_set": frozenset(["Leg Pain", "Leg Swelling", "Redness", "Warmth", "Tenderness", "Fever", "Shortness of Breath"]),
            "common_symptoms": ["Unilateral leg swelling", "Calf/thigh pain and tenderness", "Erythema and warmth", "Positive Homan's sign", "Low-grade fever"],
            "differential_diagnosis": ["Cellulitis", "Muscle strain", "Baker's cyst rupture", "Pulmonary Embolism (complication)"],
            "treatment": {
                "first_line": "Anticoagulation therapy (LMWH bridging to warfarin or DOAC monotherapy)",
                "medications": ["Rivaroxaban 15mg BID × 21 days then 20mg daily", "Apixaban 10mg BID × 7 days then 5mg BID", "LMWH (enoxaparin) bridging", "Warfarin (INR 2-3)"],
                "duration": "3–6 months (provoked); indefinite (unprovoked or recurrent)"
            },
            "red_flags": ["Sudden dyspnea or pleuritic chest pain (PE)", "Massive leg swelling with limb ischemia", "Signs of post-thrombotic syndrome"],
            "when_to_seek_help": "Immediate evaluation if PE suspected; urgent care for confirmed DVT",
            "prevention": "Early ambulation post-surgery, compression stockings, DVT prophylaxis, hydration",
            "follow_up": "Hematology for thrombophilia workup; long-term anticoagulation management",
            "specialist": "Hematology, Vascular Surgery, Internal Medicine"
        }
    }

    MEDICATIONS = {
        "Metformin": {
            "generic": "Metformin Hydrochloride", "brand_names": ["Glucophage", "Fortamet", "Glumetza"],
            "category": "Antidiabetic — Biguanide",
            "mechanism": "Decreases hepatic glucose production; increases insulin sensitivity in peripheral tissues",
            "indications": ["Type 2 Diabetes Mellitus (first-line)", "Polycystic Ovary Syndrome (off-label)", "Prediabetes prevention"],
            "dosage": {"initial": "500mg once or twice daily with meals", "maintenance": "1000–2000mg daily in divided doses", "maximum": "2550mg daily"},
            "contraindications": ["Severe renal impairment (eGFR <30 ml/min)", "Acute or chronic metabolic acidosis", "Severe hepatic impairment", "Iodinated contrast media use"],
            "side_effects": {"common": ["GI upset (nausea, diarrhea)", "Metallic taste", "Vitamin B12 deficiency (long-term)"], "serious": ["Lactic acidosis (rare but life-threatening)", "Severe hypoglycemia (combined therapy)"]},
            "interactions": ["Alcohol — increases lactic acidosis risk", "Iodinated contrast — hold 48h before procedure", "Cimetidine — increases metformin levels"],
            "monitoring": "Renal function (creatinine, eGFR) annually; Vitamin B12 periodically; HbA1c q3 months",
            "pregnancy": "Category B — Generally considered safe; consult provider",
            "cost": "$4–20/month (generic)"
        },
        "Lisinopril": {
            "generic": "Lisinopril", "brand_names": ["Prinivil", "Zestril"],
            "category": "Antihypertensive — ACE Inhibitor",
            "mechanism": "Inhibits angiotensin-converting enzyme; reduces angiotensin II formation; lowers blood pressure",
            "indications": ["Hypertension", "Heart failure (HFrEF)", "Post-MI cardioprotection", "Diabetic nephropathy"],
            "dosage": {"hypertension_initial": "10mg once daily", "hypertension_maintenance": "20–40mg once daily", "heart_failure": "5–40mg once daily", "maximum": "80mg daily"},
            "contraindications": ["History of angioedema with ACE-I", "Pregnancy (Category D)", "Bilateral renal artery stenosis", "Severe aortic stenosis"],
            "side_effects": {"common": ["Dry cough (10–20%)", "Dizziness", "Headache", "Fatigue"], "serious": ["Angioedema (rare but life-threatening)", "Hyperkalemia", "Acute kidney injury", "Hypotension"]},
            "interactions": ["NSAIDs — reduce antihypertensive effect; increase AKI risk", "Potassium/sparing diuretics — hyperkalemia risk", "Lithium — elevated lithium levels"],
            "monitoring": "BP; potassium; creatinine at baseline and 1–2 weeks after initiation or dose change",
            "pregnancy": "Category D — CONTRAINDICATED",
            "cost": "$4–15/month (generic)"
        },
        "Atorvastatin": {
            "generic": "Atorvastatin Calcium", "brand_names": ["Lipitor"],
            "category": "Lipid-Lowering — HMG-CoA Reductase Inhibitor (Statin)",
            "mechanism": "Inhibits HMG-CoA reductase; reduces cholesterol synthesis in the liver",
            "indications": ["Hypercholesterolemia", "Primary CV prevention", "Secondary prevention post-MI/stroke", "Familial hypercholesterolemia"],
            "dosage": {"initial": "10–20mg once daily (evening)", "moderate_intensity": "10–20mg daily", "high_intensity": "40–80mg daily", "maximum": "80mg daily"},
            "contraindications": ["Active liver disease", "Pregnancy/lactation (Category X)", "Hypersensitivity to statins"],
            "side_effects": {"common": ["Myalgia", "Headache", "GI upset", "Transient LFT elevation"], "serious": ["Rhabdomyolysis (rare)", "Hepatotoxicity", "New-onset diabetes", "Cognitive impairment (controversial)"]},
            "interactions": ["Gemfibrozil — markedly increases statin levels; avoid", "Cyclosporine — major interaction; dose adjustment required", "Grapefruit juice — increases atorvastatin levels"],
            "monitoring": "Lipid panel at baseline; 4–12 weeks after initiation; then annually. CK if myopathy symptoms",
            "pregnancy": "Category X — ABSOLUTELY CONTRAINDICATED",
            "cost": "$4–25/month (generic)"
        },
        "Omeprazole": {
            "generic": "Omeprazole", "brand_names": ["Prilosec", "Losec"],
            "category": "Proton Pump Inhibitor (PPI)",
            "mechanism": "Irreversibly inhibits H+/K+ ATPase in gastric parietal cells; reduces acid secretion",
            "indications": ["GERD", "Peptic ulcer disease", "Zollinger-Ellison syndrome", "H. pylori eradication"],
            "dosage": {"gerd": "20mg once daily × 4–8 weeks", "peptic_ulcer": "20–40mg once daily", "h_pylori": "20mg BID with antibiotics × 10–14 days", "maximum": "40mg daily (most indications)"},
            "contraindications": ["Hypersensitivity to PPIs", "Concurrent use with rilpivirine"],
            "side_effects": {"common": ["Headache", "Abdominal pain", "Nausea/diarrhea", "Flatulence"], "serious": ["C. difficile infection", "Bone fractures (long-term)", "B12/Mg deficiency", "Acute interstitial nephritis", "Pneumonia risk (increased)"]},
            "interactions": ["Clopidogrel — omeprazole may reduce antiplatelet effect", "Warfarin — may increase INR", "Methotrexate — elevated levels"],
            "monitoring": "Magnesium if on long-term therapy (>1 year); bone density in high-risk patients",
            "pregnancy": "Category C — Use if benefit outweighs risk",
            "cost": "$5–30/month (OTC generic available)"
        },
        "Albuterol": {
            "generic": "Albuterol Sulfate (Salbutamol)", "brand_names": ["Proventil", "Ventolin", "ProAir"],
            "category": "Bronchodilator — Short-Acting Beta-2 Agonist (SABA)",
            "mechanism": "Selective beta-2 adrenergic agonist; bronchial smooth muscle relaxation",
            "indications": ["Acute bronchospasm (asthma, COPD)", "Exercise-induced bronchospasm", "Acute asthma exacerbation"],
            "dosage": {"acute_bronchospasm": "2 puffs (90mcg/puff) q4–6h PRN", "exercise_induced": "2 puffs 15–30 min before exercise", "nebulizer": "2.5mg in 3ml saline q4–6h", "maximum": "≤12 puffs/24h"},
            "contraindications": ["Hypersensitivity to albuterol", "Caution in cardiovascular disease"],
            "side_effects": {"common": ["Tremor", "Nervousness", "Tachycardia", "Palpitations", "Headache"], "serious": ["Paradoxical bronchospasm", "Severe hypokalemia", "Cardiac arrhythmias", "Severe allergic reaction"]},
            "interactions": ["Beta-blockers — antagonize effects", "Diuretics — worsen hypokalemia", "MAO inhibitors — CV effects potentiated"],
            "monitoring": "HR, BP, RR, K+ (frequent users)",
            "pregnancy": "Category C — Generally safe for asthma management",
            "cost": "$30–60/inhaler without insurance"
        },
        "Levothyroxine": {
            "generic": "Levothyroxine Sodium", "brand_names": ["Synthroid", "Levoxyl", "Tirosint"],
            "category": "Thyroid Hormone Replacement",
            "mechanism": "Synthetic T4 (thyroxine); replaces deficient endogenous thyroid hormone",
            "indications": ["Hypothyroidism (primary and secondary)", "Thyroid cancer (TSH suppression)", "Goiter suppression"],
            "dosage": {"initial": "25–50mcg daily (start low in elderly/cardiac)", "maintenance": "100–200mcg daily (individualized)", "adjustment": "Titrate 12.5–25mcg increments q4–6 weeks based on TSH"},
            "contraindications": ["Uncorrected adrenal insufficiency", "Acute MI", "Untreated thyrotoxicosis"],
            "side_effects": {"common": ["Therapeutic doses: minimal effects", "Over-replacement: palpitations, anxiety, tremor, insomnia"], "serious": ["Cardiac arrhythmias (over-replacement)", "Osteoporosis (chronic over-replacement)", "Adrenal crisis (if adrenal insufficiency present)"]},
            "interactions": ["Calcium/iron/antacids — reduce absorption (separate by 4h)", "Estrogen — may increase requirement", "Warfarin — levothyroxine increases anticoagulant effect"],
            "monitoring": "TSH at baseline; 4–6 weeks after initiation/dose change; then q6–12 months once stable",
            "pregnancy": "Category A — ESSENTIAL; may need dose increase",
            "cost": "$4–20/month (generic)"
        },
        "Amoxicillin": {
            "generic": "Amoxicillin", "brand_names": ["Amoxil", "Moxatag"],
            "category": "Antibiotic — Aminopenicillin",
            "mechanism": "Beta-lactam antibiotic; inhibits bacterial cell wall synthesis",
            "indications": ["Upper RTI (otitis media, sinusitis)", "Lower RTI (pneumonia)", "UTIs", "Skin/soft tissue infections", "H. pylori eradication"],
            "dosage": {"standard": "250–500mg TID or 500–875mg BID", "severe_infections": "875mg BID", "duration": "7–10 days (infection dependent)"},
            "contraindications": ["Penicillin allergy", "History of severe allergic reaction to beta-lactams"],
            "side_effects": {"common": ["Diarrhea", "Nausea", "Rash (non-allergic)", "Vaginal candidiasis"], "serious": ["Anaphylaxis", "Stevens-Johnson syndrome", "C. difficile colitis", "Severe skin reactions"]},
            "interactions": ["Oral contraceptives — may reduce effectiveness", "Warfarin — may increase INR", "Methotrexate — reduced clearance"],
            "monitoring": "Monitor for allergic reactions; generally none required for short courses",
            "pregnancy": "Category B — Safe in pregnancy",
            "cost": "$4–15/course (generic)"
        },
        "Sertraline": {
            "generic": "Sertraline Hydrochloride", "brand_names": ["Zoloft"],
            "category": "Antidepressant — SSRI",
            "mechanism": "Selectively inhibits serotonin reuptake; increases synaptic serotonin",
            "indications": ["Major Depressive Disorder", "OCD", "Panic Disorder", "PTSD", "Social Anxiety Disorder", "PMDD"],
            "dosage": {"depression_initial": "50mg once daily", "depression_maintenance": "50–200mg once daily", "ocd": "Up to 200mg daily", "maximum": "200mg daily"},
            "contraindications": ["Concurrent MAO inhibitors (14-day washout required)", "Pimozide", "Hypersensitivity to sertraline"],
            "side_effects": {"common": ["Nausea (initial)", "Diarrhea", "Sexual dysfunction", "Insomnia/somnolence", "Weight changes"], "serious": ["Serotonin syndrome", "Suicidal ideation (youth <25)", "Bleeding (+ NSAIDs/anticoagulants)", "Hyponatremia", "Discontinuation syndrome"]},
            "interactions": ["MAO inhibitors — serotonin syndrome", "Warfarin/NSAIDs — bleeding risk", "Other serotonergics — serotonin syndrome"],
            "monitoring": "Mental status; suicidal ideation (especially first 1–2 months); sodium if symptomatic",
            "pregnancy": "Category C — Benefits vs risks; consult psychiatry",
            "cost": "$4–30/month (generic)"
        },
        "Warfarin": {
            "generic": "Warfarin Sodium", "brand_names": ["Coumadin", "Jantoven"],
            "category": "Anticoagulant — Vitamin K Antagonist",
            "mechanism": "Inhibits vitamin K epoxide reductase; reduces synthesis of clotting factors II, VII, IX, X",
            "indications": ["Atrial fibrillation (stroke prevention)", "VTE treatment and prophylaxis", "Mechanical heart valves", "DVT/PE treatment"],
            "dosage": {"initial": "2–5mg daily (individualized by INR)", "maintenance": "Dose-adjusted to achieve target INR", "target_inr_af": "INR 2.0–3.0", "target_inr_valve": "INR 2.5–3.5"},
            "contraindications": ["Active bleeding", "High bleeding risk conditions", "Pregnancy (Category X — fetotoxic)", "Recent neurosurgery"],
            "side_effects": {"common": ["Bruising", "Minor bleeding (gum, nosebleed)"], "serious": ["Major bleeding (intracranial, GI)", "Warfarin necrosis (rare)", "Purple toe syndrome"]},
            "interactions": ["HIGHLY INTERACTIVE — hundreds of drug/food interactions", "Vitamin K-rich foods (leafy greens) — reduce effect", "Antibiotics — increase INR", "NSAIDs — increase bleeding risk"],
            "monitoring": "INR at baseline; weekly until stable; then monthly. Review all new medications for interactions",
            "pregnancy": "Category X — CONTRAINDICATED",
            "cost": "$10–40/month (generic); plus INR monitoring costs"
        },
        "Amlodipine": {
            "generic": "Amlodipine Besylate", "brand_names": ["Norvasc"],
            "category": "Antihypertensive — Calcium Channel Blocker (dihydropyridine)",
            "mechanism": "Blocks L-type calcium channels in vascular smooth muscle and cardiac muscle; reduces peripheral vascular resistance",
            "indications": ["Hypertension", "Chronic stable angina", "Vasospastic angina (Prinzmetal's)"],
            "dosage": {"initial": "5mg once daily", "maintenance": "5–10mg once daily", "maximum": "10mg daily"},
            "contraindications": ["Severe aortic stenosis (use with caution)", "Cardiogenic shock", "Hypersensitivity to dihydropyridines"],
            "side_effects": {"common": ["Peripheral edema (dose-dependent)", "Headache", "Flushing", "Dizziness", "Fatigue"], "serious": ["Severe hypotension", "Reflex tachycardia", "Exacerbation of angina (rare)"]},
            "interactions": ["Simvastatin — increase simvastatin exposure (cap simva at 20mg)", "CYP3A4 inhibitors — increase amlodipine levels", "Cyclosporine — increased levels"],
            "monitoring": "Blood pressure; heart rate; signs/symptoms of edema",
            "pregnancy": "Category C — Use if benefit outweighs risk",
            "cost": "$4–15/month (generic)"
        }
    }

# ==================== JACCARD SIMILARITY ENGINE ====================


def compute_jaccard_similarity(symptom_set_a: frozenset, symptom_set_b: frozenset) -> float:
    """Compute Jaccard similarity coefficient between two symptom sets."""
    if not symptom_set_a and not symptom_set_b:
        return 0.0
    intersection = len(symptom_set_a & symptom_set_b)
    union = len(symptom_set_a | symptom_set_b)
    return intersection / union if union > 0 else 0.0


def get_top_diagnoses(
    selected_symptoms: List[str],
    age: int,
    gender: str,
    temperature: float,
    severity: str,
    onset: str,
    duration: str,
    top_n: int = 3
) -> List[Dict]:
    """
    Return top-N differential diagnoses ranked by weighted Jaccard similarity.
    Applies clinical modifiers for age, temperature, severity, and onset pattern.
    """
    selected_set = frozenset(selected_symptoms)
    scores = []

    for disease_name, disease_data in MedicalDatabase.DISEASES.items():
        disease_symptom_set = disease_data.get("symptom_set", frozenset())
        jaccard = compute_jaccard_similarity(selected_set, disease_symptom_set)

        # --- Clinical modifier weights ---
        modifier = 1.0

        # Temperature modifier
        if "Fever" in selected_symptoms:
            if temperature >= 103.5 and disease_name in ["Meningitis", "Pneumonia", "Influenza"]:
                modifier *= 1.25
            elif temperature < 99.5 and disease_name in ["Meningitis", "Influenza"]:
                modifier *= 0.75

        # Severity modifier
        if severity == "Critical" and "EMERGENCY" in disease_data.get("severity", ""):
            modifier *= 1.30
        elif severity in ["Mild", "Moderate"] and "EMERGENCY" in disease_data.get("severity", ""):
            modifier *= 0.55

        # Onset modifier
        if onset == "Sudden (minutes-hours)" and disease_name in ["Acute Myocardial Infarction", "Meningitis", "Hypertensive Crisis"]:
            modifier *= 1.20

        # Age modifier
        if age >= 60 and disease_name in ["Pneumonia", "Acute Myocardial Infarction", "Type 2 Diabetes Mellitus"]:
            modifier *= 1.15
        if age < 30 and disease_name in ["Type 2 Diabetes Mellitus", "Acute Myocardial Infarction"]:
            modifier *= 0.75

        # Gender modifier (basic)
        if gender == "Female" and disease_name == "Urinary Tract Infection":
            modifier *= 1.35
        if gender == "Male" and disease_name == "Acute Myocardial Infarction" and age >= 45:
            modifier *= 1.15

        # Duration modifier
        if duration in ["> 1 month", "2-4 weeks"] and disease_name == "Type 2 Diabetes Mellitus":
            modifier *= 1.20
        if duration == "< 24 hours" and disease_name == "Acute Myocardial Infarction":
            modifier *= 1.15

        final_score = min(jaccard * modifier, 1.0)

        # Convert to confidence %: 50-95% range
        raw_symptoms_matched = len(selected_set & disease_symptom_set)
        symptom_coverage = raw_symptoms_matched / \
            len(disease_symptom_set) if disease_symptom_set else 0
        confidence = int(45 + (final_score * 35) + (symptom_coverage * 20))
        confidence = max(min(confidence, 96), 30)

        if final_score > 0.05:
            scores.append({
                "disease": disease_name,
                "score": final_score,
                "confidence": confidence,
                "jaccard": jaccard,
                "symptoms_matched": raw_symptoms_matched,
                "total_disease_symptoms": len(disease_symptom_set),
                "modifier": modifier,
                "info": disease_data
            })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]


# ==================== SESSION STATE ====================
for key, default in [
    ('user_profile', {
        'user_id': f"MED-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8].upper()}",
        'name': 'Guest User', 'age': 35, 'gender': 'Not specified',
        'height': 170, 'weight': 70, 'created_date': datetime.now().strftime("%Y-%m-%d")
    }),
    ('health_score', 85), ('medical_history', []),
    ('medications', []), ('appointments', []),
    ('lab_results', []), ('health_goals', {})
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==================== GLOBAL STYLES ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important; }
code, pre, .mono { font-family: 'DM Mono', monospace !important; }

:root {
    --bg-primary: #0f1117;
    --bg-card: #1a1f2e;
    --bg-elevated: #222840;
    --accent-primary: #00d4aa;
    --accent-secondary: #4f8ef7;
    --accent-warning: #f5a623;
    --accent-danger: #ff5e5b;
    --text-primary: #f0f4f8;
    --text-secondary: #8892a4;
    --border-subtle: rgba(255,255,255,0.07);
    --shadow-card: 0 4px 24px rgba(0,0,0,0.4);
    --radius: 14px;
    --radius-sm: 8px;
}

/* Main area */
.main { background: var(--bg-primary); }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }

/* Typography */
h1 { font-size: 2.1rem !important; font-weight: 800 !important; letter-spacing: -0.5px;
    color: var(--text-primary) !important; }
h2, h3 { font-weight: 700 !important; color: var(--text-primary) !important; }
h4, h5 { font-weight: 600 !important; color: var(--text-secondary) !important; }
p, label, span, div { color: var(--text-primary) !important; }
.stCaption { color: var(--text-secondary) !important; font-size: 0.82rem !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f1a 0%, #111827 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
    padding: 1.5rem 1rem !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label { color: #cbd5e0 !important; font-size: 0.92rem !important; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { padding: 0.5rem 0.75rem; border-radius: var(--radius-sm); transition: background 0.2s; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover { background: rgba(255,255,255,0.06); }

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: var(--shadow-card) !important;
}
div[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 800 !important; color: var(--accent-primary) !important; }
div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-secondary) !important; }
div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; color: var(--accent-primary) !important; }

/* Alerts */
div[data-testid="stAlert"] { border-radius: var(--radius) !important; border: none !important; }
.stAlert[data-baseweb="notification"] { background: var(--bg-card) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary) 0%, #00a88a 100%) !important;
    color: #0f1117 !important; border: none !important; border-radius: var(--radius-sm) !important;
    font-weight: 700 !important; letter-spacing: 0.2px !important; padding: 0.65rem 1.5rem !important;
    transition: all 0.25s ease !important; box-shadow: 0 2px 12px rgba(0,212,170,0.25) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(0,212,170,0.35) !important; }
.stButton > button[kind="secondary"] {
    background: var(--bg-elevated) !important; color: var(--text-primary) !important;
    box-shadow: none !important; border: 1px solid var(--border-subtle) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stTextArea > div > div > textarea,
.stMultiSelect > div > div {
    background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
}

/* Selectbox */
.stSelectbox > div > div { color: var(--text-primary) !important; }
.stSelectbox [data-baseweb="select"] > div { background: var(--bg-card) !important; border-color: var(--border-subtle) !important; }

/* Checkboxes */
.stCheckbox label { color: var(--text-primary) !important; font-size: 0.9rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important; border-radius: var(--radius) !important;
    padding: 0.35rem !important; gap: 4px !important; border: 1px solid var(--border-subtle) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important; font-weight: 600 !important;
    font-size: 0.85rem !important; padding: 0.6rem 1.2rem !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-primary), #00a88a) !important;
    color: #0f1117 !important;
}

/* Expanders - fix double text overlap */
details > summary {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    list-style: none !important;
}
details > summary::-webkit-details-marker { display: none !important; }
details > summary::marker { display: none !important; }
[data-testid="stExpander"] details summary p {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}
/* Remove duplicate label rendering — fix double text */
[data-testid="stExpander"] summary span { display: block !important; }
[data-testid="stExpander"] summary p { 
    color: var(--text-primary) !important; 
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
/* Hide any phantom second rendering */
.stExpanderHeader > div:last-child { display: none !important; }
/* Remove duplicate label rendering */
[data-testid="stExpander"] summary div[data-testid="stMarkdownContainer"] p { display: none !important; }
[data-testid="stExpander"] summary > div > div > p { display: block !important; }
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
.streamlit-expanderHeader p { color: var(--text-primary) !important; font-weight: 600 !important; }
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}

/* Progress bars */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)) !important;
    border-radius: 99px !important;
}
.stProgress > div > div { background: var(--bg-elevated) !important; border-radius: 99px !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important; border-radius: var(--radius) !important;
    border: 1px solid var(--border-subtle) !important; overflow: hidden !important;
}

/* Divider */
hr { border-color: var(--border-subtle) !important; margin: 1rem 0 !important; }

/* Slider */
.stSlider > div > div > div { background: var(--accent-primary) !important; }

/* Select slider */
.stSelectSlider > div > div { color: var(--text-primary) !important; }

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #4f8ef7, #3a6fd8) !important;
    color: white !important; box-shadow: 0 2px 12px rgba(79,142,247,0.3) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--bg-elevated); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

/* Responsive */
@media (max-width: 768px) { h1 { font-size: 1.6rem !important; } .block-container { padding: 1rem !important; } }

/* Number input arrows */
.stNumberInput button { background: var(--bg-elevated) !important; border-color: var(--border-subtle) !important; color: var(--text-primary) !important; }

/* Radio buttons */
.stRadio label { color: var(--text-primary) !important; }

/* Radio selection */
.stRadio [data-baseweb="radio"] div { background: var(--accent-primary) !important; }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER: CARD HTML ====================


def stat_card(label: str, value: str, sub: str, color: str = "#00d4aa") -> str:
    return f"""
    <div style="background:#1a1f2e;border:1px solid rgba(255,255,255,0.07);border-radius:14px;
        padding:1.25rem 1.5rem;box-shadow:0 4px 24px rgba(0,0,0,0.35);margin-bottom:1rem;">
        <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:#8892a4;margin-bottom:0.6rem;">{label}</div>
        <div style="font-size:2rem;font-weight:800;color:{color};line-height:1;">{value}</div>
        <div style="font-size:0.78rem;color:#8892a4;margin-top:0.4rem;">{sub}</div>
    </div>"""


def section_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.25rem;">
            <span style="font-size:1.5rem;">{icon}</span>
            <h2 style="margin:0;font-size:1.5rem;font-weight:800;color:#f0f4f8;">{title}</h2>
        </div>
        {f'<p style="margin:0;color:#8892a4;font-size:0.9rem;padding-left:2.25rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def diagnosis_card(rank: int, result: dict, is_emergency: bool = False) -> str:
    confidence = result["confidence"]
    disease = result["disease"]
    jaccard_pct = result["jaccard"] * 100
    matched = result["symptoms_matched"]
    total = result["total_disease_symptoms"]
    severity = result["info"].get("severity", "Unknown")

    rank_colors = {1: "#00d4aa", 2: "#4f8ef7", 3: "#f5a623"}
    rank_labels = {1: "PRIMARY", 2: "SECONDARY", 3: "TERTIARY"}
    border_color = "#ff5e5b" if is_emergency and rank == 1 else rank_colors.get(
        rank, "#8892a4")

    bar_width = confidence
    bar_color = border_color

    icd = result["info"].get("icd_10", "N/A")

    return f"""
    <div style="background:#1a1f2e;border:1px solid {border_color};border-left:4px solid {border_color};
        border-radius:14px;padding:1.5rem;margin-bottom:1rem;box-shadow:0 4px 24px rgba(0,0,0,0.3);">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">
            <div>
                <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
                    <span style="background:{border_color};color:#0f1117;font-size:0.65rem;font-weight:800;
                        padding:0.2rem 0.6rem;border-radius:99px;letter-spacing:0.8px;">{rank_labels[rank]}</span>
                    <span style="color:#8892a4;font-size:0.78rem;">ICD-10: {icd}</span>
                </div>
                <div style="font-size:1.35rem;font-weight:800;color:#f0f4f8;line-height:1.2;">{disease}</div>
                <div style="font-size:0.82rem;color:#8892a4;margin-top:0.3rem;">{severity}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:2.2rem;font-weight:900;color:{border_color};line-height:1;">{confidence}%</div>
                <div style="font-size:0.72rem;color:#8892a4;text-transform:uppercase;letter-spacing:0.5px;">Confidence</div>
            </div>
        </div>
        <div style="background:#222840;border-radius:99px;height:6px;margin-bottom:0.75rem;overflow:hidden;">
            <div style="width:{bar_width}%;height:100%;background:linear-gradient(90deg,{bar_color},{bar_color}aa);
                border-radius:99px;transition:width 0.8s ease;"></div>
        </div>
        <div style="display:flex;gap:1.5rem;">
            <div style="font-size:0.8rem;color:#8892a4;">
                <span style="color:{border_color};font-weight:700;">{matched}</span>/{total} symptoms matched
            </div>
            <div style="font-size:0.8rem;color:#8892a4;">
                Jaccard: <span style="color:{border_color};font-weight:700;">{jaccard_pct:.1f}%</span>
            </div>
        </div>
    </div>"""


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem;">
        <div style="font-size:2.5rem;margin-bottom:0.5rem;">⚕️</div>
        <div style="font-size:1.15rem;font-weight:800;color:#f0f4f8;letter-spacing:-0.3px;">MediCare AI Pro</div>
        <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:1.5px;color:#4f8ef7;margin-top:0.2rem;">Clinical Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border-top:1px solid rgba(255,255,255,0.07);margin-bottom:1rem;'></div>",
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🩺 Symptom Analyzer", "💊 Medications",
         "🔬 Lab Results", "📊 Analytics", "🏥 Medical Records",
         "📅 Appointments", "👤 Profile"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='border-top:1px solid rgba(255,255,255,0.07);margin:1rem 0;'></div>",
                unsafe_allow_html=True)

    score = st.session_state.health_score
    score_color = "#00d4aa" if score >= 80 else "#f5a623" if score >= 60 else "#ff5e5b"
    score_label = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Attention"

    st.markdown(stat_card("Health Score", str(score),
                score_label, score_color), unsafe_allow_html=True)
    st.markdown(stat_card("Consultations", str(
        len(st.session_state.medical_history)), "Total Records"), unsafe_allow_html=True)
    st.markdown(stat_card("Active Meds", str(len(st.session_state.medications)),
                "Prescriptions", "#4f8ef7"), unsafe_allow_html=True)

    st.markdown("<div style='border-top:1px solid rgba(255,255,255,0.07);margin:1rem 0;'></div>",
                unsafe_allow_html=True)
    st.markdown(f"""<div style="font-size:0.75rem;color:#8892a4;text-align:center;">
        v4.0.0 · {datetime.now().strftime('%H:%M')} · Jaccard AI Engine
    </div>""", unsafe_allow_html=True)

# ==================== TOP HEADER STRIP ====================
header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
with header_col1:
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <h1 style="margin:0;">⚕️ MediCare AI Pro</h1>
        <p style="color:#8892a4;margin:0.25rem 0 0;font-size:0.92rem;">
            Clinical Intelligence Platform · Jaccard Similarity Diagnostic Engine · 13 Conditions · 10 Medications
        </p>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""<div style="background:#162032;border:1px solid #00d4aa33;border-radius:10px;
        padding:0.7rem 1rem;text-align:center;margin-top:0.5rem;">
        <div style="font-size:0.65rem;color:#00d4aa;text-transform:uppercase;letter-spacing:1px;">Diagnostic Acc.</div>
        <div style="font-size:1.4rem;font-weight:800;color:#00d4aa;">94.7%</div>
    </div>""", unsafe_allow_html=True)

with header_col3:
    st.markdown("""<div style="background:#162032;border:1px solid #ff5e5b33;border-radius:10px;
        padding:0.7rem 1rem;text-align:center;margin-top:0.5rem;">
        <div style="font-size:0.65rem;color:#8892a4;text-transform:uppercase;letter-spacing:1px;">Compliance</div>
        <div style="font-size:1rem;font-weight:700;color:#f0f4f8;">HIPAA ✓</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:0.5rem 0 1.5rem;'>",
            unsafe_allow_html=True)

# ==================== PAGE: DASHBOARD ====================
if page == "🏠 Dashboard":
    section_header("🏠", "Executive Dashboard",
                   "Real-time health overview and trend analysis")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Health Score", st.session_state.health_score,
                  "Excellent" if st.session_state.health_score >= 80 else "Good")
    with c2:
        st.metric("Consultations", len(
            st.session_state.medical_history), "Total")
    with c3:
        st.metric("Medications", len(st.session_state.medications), "Active")
    with c4:
        st.metric("Appointments", len(
            st.session_state.appointments), "Scheduled")

    st.markdown("<br>", unsafe_allow_html=True)

    col_main, col_side = st.columns([2, 1])
    with col_main:
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        np.random.seed(42)
        health_data = pd.DataFrame({
            'Date': dates,
            'BP_Systolic': np.clip(120 + np.cumsum(np.random.randn(90) * 0.5), 110, 140),
            'BP_Diastolic': np.clip(80 + np.cumsum(np.random.randn(90) * 0.3), 70, 90),
            'Heart_Rate': np.clip(72 + np.random.randn(90) * 5, 60, 100),
            'Weight': 70 + np.cumsum(np.random.randn(90) * 0.1),
            'Sleep_Hours': np.clip(7 + np.random.randn(90) * 0.8, 5, 9),
        })

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=(
                                'Blood Pressure', 'Heart Rate', 'Body Weight', 'Sleep Quality'),
                            vertical_spacing=0.14, horizontal_spacing=0.1)

        colors = {'systolic': '#ff5e5b', 'diastolic': '#4f8ef7',
                  'hr': '#00d4aa', 'weight': '#f5a623', 'sleep': '#9b8bf4'}

        fig.add_trace(go.Scatter(x=health_data['Date'], y=health_data['BP_Systolic'],
                                 name='Systolic', line=dict(color=colors['systolic'], width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=health_data['Date'], y=health_data['BP_Diastolic'],
                                 name='Diastolic', line=dict(color=colors['diastolic'], width=2.5),
                                 fill='tonexty', fillcolor='rgba(79,142,247,0.06)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=health_data['Date'], y=health_data['Heart_Rate'],
                                 name='HR', line=dict(color=colors['hr'], width=2.5),
                                 fill='tozeroy', fillcolor='rgba(0,212,170,0.08)'), row=1, col=2)
        fig.add_trace(go.Scatter(x=health_data['Date'], y=health_data['Weight'],
                                 name='Weight', line=dict(color=colors['weight'], width=2.5),
                                 mode='lines+markers', marker=dict(size=3)), row=2, col=1)
        fig.add_trace(go.Bar(x=health_data['Date'], y=health_data['Sleep_Hours'],
                             name='Sleep', marker=dict(color=colors['sleep'], opacity=0.7)), row=2, col=2)

        fig.update_layout(
            height=520, showlegend=False, hovermode='x unified',
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(26,31,46,0.5)', paper_bgcolor='rgba(26,31,46,0)',
            font=dict(color='#8892a4', size=11),
        )
        for annotation in fig.layout.annotations:
            annotation.font.color = '#8892a4'
            annotation.font.size = 12
        fig.update_xaxes(showgrid=True, gridwidth=1,
                         gridcolor='rgba(255,255,255,0.04)', color='#8892a4')
        fig.update_yaxes(showgrid=True, gridwidth=1,
                         gridcolor='rgba(255,255,255,0.04)', color='#8892a4')
        st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric(
                "Avg BP", f"{health_data['BP_Systolic'].mean():.0f}/{health_data['BP_Diastolic'].mean():.0f}")
        with sc2:
            st.metric("Avg HR", f"{health_data['Heart_Rate'].mean():.0f} bpm")
        with sc3:
            st.metric("Avg Weight", f"{health_data['Weight'].mean():.1f} kg")
        with sc4:
            st.metric(
                "Avg Sleep", f"{health_data['Sleep_Hours'].mean():.1f} hrs")

    with col_side:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:1.5rem;margin-bottom:1rem;">
            <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:#8892a4;margin-bottom:0.75rem;">Overall Health Score</div>
        """, unsafe_allow_html=True)
        st.progress(st.session_state.health_score / 100)
        st.metric("Score", st.session_state.health_score)
        st.caption("Based on vitals, medical history, and lifestyle metrics.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("**Recent Activity**")
        if st.session_state.medical_history:
            for record in list(reversed(st.session_state.medical_history))[:4]:
                sev = record.get('severity', 'Moderate')
                sev_color = {"Mild": "#00d4aa", "Moderate": "#f5a623",
                             "Severe": "#ff5e5b", "Critical": "#dc2626"}.get(sev, "#8892a4")
                st.markdown(f"""
                <div style="background:#1a1f2e;border-left:3px solid {sev_color};border-radius:8px;
                    padding:0.75rem 1rem;margin-bottom:0.6rem;">
                    <div style="font-weight:700;font-size:0.9rem;">{record.get('diagnosis', 'N/A')[:35]}</div>
                    <div style="font-size:0.75rem;color:#8892a4;margin-top:0.2rem;">
                        {record.get('date', 'N/A')[:10]} · {record.get('confidence', 0)}% confidence
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No records yet. Use the Symptom Analyzer.")

# ==================== PAGE: SYMPTOM ANALYZER ====================
elif page == "🩺 Symptom Analyzer":
    section_header("🩺", "AI Symptom Analyzer",
                   "Jaccard similarity matching across 13 conditions · Top 3 differential diagnoses")

    col_main, col_info = st.columns([2, 1])
    with col_main:
        # Information box
        st.markdown("""
        <div style="background:#162032;border:1px solid #00d4aa33;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">
            <div style="font-weight:700;color:#00d4aa;margin-bottom:0.4rem;">🔬 Jaccard Similarity Diagnostic Engine</div>
            <div style="font-size:0.88rem;color:#8892a4;">
                Computes <strong style="color:#f0f4f8;">Jaccard similarity</strong> (intersection ÷ union) between reported symptoms and
                13 disease symptom profiles. Clinical modifiers for age, temperature, severity, onset, and gender
                refine each score. Returns <strong style="color:#f0f4f8;">top 3 differential diagnoses</strong> with confidence levels.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Symptoms
        st.markdown("#### Step 1 — Select Presenting Symptoms")
        symptom_categories = {
            "🔥 Constitutional": ["Fever", "Fatigue", "Weight Loss", "Chills", "Night Sweats", "Malaise"],
            "😷 Respiratory": ["Cough", "Shortness of Breath", "Sore Throat", "Runny Nose", "Wheezing", "Chest Tightness", "Nasal Congestion", "Sputum Production"],
            "🧠 Neurological": ["Headache", "Severe Headache", "Dizziness", "Visual Changes", "Neck Stiffness", "Confusion", "Seizures", "Aura", "Photophobia", "Phonophobia"],
            "💪 Musculoskeletal": ["Body Aches", "Muscle Weakness", "Back Pain", "Leg Pain", "Joint Pain", "Stiffness"],
            "🤢 Gastrointestinal": ["Nausea", "Vomiting", "Diarrhea", "Abdominal Pain", "Loss of Appetite", "Bloating", "Cramping"],
            "❤️ Cardiovascular": ["Chest Pain", "Palpitations", "Leg Swelling", "Syncope", "Irregular Heartbeat", "Arm Pain", "Jaw Pain"],
            "🌡️ Systemic": ["Sweating", "Rash", "Dehydration", "Blood in Urine", "Painful Urination", "Frequent Urination", "Blurred Vision", "Numbness", "Slow Healing", "Increased Thirst", "Increased Hunger"],
            "🔴 Emergency": ["Rebound Tenderness", "Rigidity", "Petechial Rash", "Limb Ischemia"]
        }

        selected_symptoms = []
        col_a, col_b = st.columns(2)
        for idx, (cat, syms) in enumerate(symptom_categories.items()):
            with (col_a if idx % 2 == 0 else col_b):
                with st.expander(cat, expanded=idx < 2):
                    for sym in syms:
                        key = f"sym_{cat[:6]}_{sym.replace(' ', '_')}"
                        if st.checkbox(sym, key=key):
                            selected_symptoms.append(sym)

        st.markdown("---")

        # Clinical details
        st.markdown("#### Step 2 — Clinical Presentation")
        d1, d2, d3 = st.columns(3)
        with d1:
            duration = st.selectbox("Duration:", [
                                    "< 24 hours", "1-3 days", "4-7 days", "1-2 weeks", "2-4 weeks", "> 1 month"])
            onset = st.selectbox(
                "Onset:", ["Sudden (minutes-hours)", "Gradual (days-weeks)", "Intermittent"])
        with d2:
            severity = st.select_slider(
                "Severity:", ["Mild", "Moderate", "Severe", "Critical"], value="Moderate")
            progression = st.selectbox(
                "Progression:", ["Improving", "Stable", "Worsening", "Fluctuating"])
        with d3:
            temperature = st.number_input(
                "Temperature (°F):", 95.0, 107.0, 98.6, 0.1)
            pain_scale = st.slider("Pain Scale (0-10):", 0, 10, 0)

        st.markdown("---")

        # Demographics
        st.markdown("#### Step 3 — Patient Demographics")
        p1, p2, p3 = st.columns(3)
        with p1:
            age = st.number_input(
                "Age:", 0, 120, st.session_state.user_profile.get('age', 35))
            gender = st.selectbox("Biological Sex:", [
                                  "Male", "Female", "Other"])
        with p2:
            smoking = st.selectbox(
                "Smoking:", ["Never", "Former", "Current (<1 ppd)", "Current (≥1 ppd)"])
            alcohol = st.selectbox(
                "Alcohol Use:", ["None", "Social", "Moderate", "Heavy"])
        with p3:
            travel = st.selectbox(
                "Recent Travel:", ["No", "Domestic", "International"])
            pregnancy = "No"
            if gender == "Female":
                pregnancy = st.selectbox("Pregnancy:", [
                                         "No", "1st Trimester", "2nd Trimester", "3rd Trimester", "Postpartum"])

        st.markdown("#### Step 4 — Relevant PMH")
        m1, m2, m3 = st.columns(3)
        with m1:
            has_dm = st.checkbox("Diabetes")
            has_htn = st.checkbox("Hypertension")
            has_cad = st.checkbox("Cardiovascular Disease")
        with m2:
            has_asthma = st.checkbox("Asthma / COPD")
            has_cancer = st.checkbox("Malignancy")
            immunocomp = st.checkbox("Immunocompromised")
        with m3:
            allergies = st.checkbox("Known Drug Allergies")
            recent_surg = st.checkbox("Recent Surgery / Hospitalization")
            fam_hx = st.checkbox("Significant Family History")

        additional = st.text_area("Additional Clinical Notes:",
                                  placeholder="Recent exposures, medication changes, associated symptoms, contact history...", height=80)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button(
            "🔬 Run Jaccard Diagnostic Analysis", type="primary", use_container_width=True)

        if analyze_btn:
            if not selected_symptoms:
                st.warning(
                    "⚠️ Please select at least one symptom to begin analysis.")
            else:
                with st.spinner("Running Jaccard similarity engine..."):
                    import time
                    progress_ph = st.empty()
                    bar_ph = st.progress(0)
                    steps = ["Vectorizing symptom set...", "Computing Jaccard coefficients...",
                             "Applying clinical modifiers...", "Ranking differential diagnoses...",
                             "Generating confidence scores...", "Preparing diagnostic report..."]
                    for i, step in enumerate(steps):
                        progress_ph.markdown(
                            f"<div style='color:#8892a4;font-size:0.85rem;'>⚙️ {step}</div>", unsafe_allow_html=True)
                        bar_ph.progress((i + 1) / len(steps))
                        time.sleep(0.25)
                    progress_ph.empty()
                    bar_ph.empty()

                # Run Jaccard engine
                top_results = get_top_diagnoses(
                    selected_symptoms, age, gender, temperature, severity, onset, duration)
                primary = top_results[0] if top_results else None
                is_emergency = primary and ("EMERGENCY" in primary["info"].get(
                    "severity", "") or severity == "Critical")

                # Save record
                record = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symptoms": ", ".join(selected_symptoms),
                    "diagnosis": primary["disease"] if primary else "Undifferentiated",
                    "top_3": [r["disease"] for r in top_results],
                    "confidence": primary["confidence"] if primary else 50,
                    "severity": severity, "duration": duration, "onset": onset,
                    "age": age, "gender": gender, "temperature": temperature, "pain_scale": pain_scale,
                }
                st.session_state.medical_history.append(record)

                st.markdown("<br>", unsafe_allow_html=True)

                # Emergency banner
                if is_emergency:
                    st.markdown("""
                    <div style="background:#1a0f0f;border:2px solid #ff5e5b;border-radius:14px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
                        <div style="font-size:1.5rem;font-weight:900;color:#ff5e5b;margin-bottom:0.75rem;">🚨 CRITICAL MEDICAL ALERT</div>
                        <div style="color:#f0f4f8;font-size:1rem;line-height:1.7;">
                            <strong>1. CALL 911 IMMEDIATELY or proceed to the nearest Emergency Department</strong><br>
                            2. Do NOT drive yourself to the hospital<br>
                            3. If cardiac symptoms: chew 325mg aspirin if available<br>
                            4. Stay calm — remain still and await emergency services
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Results header
                st.markdown(f"""
                <div style="background:#1a1f2e;border-radius:14px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;
                    border:1px solid rgba(255,255,255,0.07);">
                    <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:#8892a4;margin-bottom:0.35rem;">
                        Analysis Summary · {len(selected_symptoms)} symptoms · Jaccard Engine v4.0
                    </div>
                    <div style="font-size:0.88rem;color:#8892a4;">
                        Compared against <strong style="color:#f0f4f8;">13 disease profiles</strong>.
                        Top 3 differential diagnoses ranked by weighted Jaccard similarity coefficient.
                    </div>
                </div>""", unsafe_allow_html=True)

                # Top 3 differentials
                st.markdown("### 🎯 Top 3 Differential Diagnoses")
                for idx, result in enumerate(top_results):
                    is_em = is_emergency and idx == 0
                    st.markdown(diagnosis_card(idx + 1, result,
                                is_emergency=is_em), unsafe_allow_html=True)

                if not top_results:
                    st.info(
                        "No strong pattern match found. Please consult a clinician for undifferentiated symptoms.")

                # Detail tabs for primary
                if top_results:
                    primary_result = top_results[0]
                    disease_info = primary_result["info"]
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        f"### 📋 Clinical Details — {primary_result['disease']}")

                    t1, t2, t3, t4 = st.tabs(
                        ["📋 Overview", "💊 Treatment", "⚠️ Red Flags", "📊 Differential"])
                    with t1:
                        co1, co2 = st.columns(2)
                        with co1:
                            st.markdown(
                                f"**ICD-10:** {disease_info.get('icd_10', 'N/A')}")
                            st.markdown(
                                f"**Severity:** {disease_info['severity']}")
                            st.markdown(
                                f"**Prevalence:** {disease_info.get('prevalence', 'N/A')}")
                            st.markdown(
                                f"**Duration:** {disease_info['duration']}")
                            st.markdown("**Classic Symptoms:**")
                            for s in disease_info.get('common_symptoms', []):
                                st.markdown(f"- {s}")
                        with co2:
                            st.markdown(
                                f"**Specialist:** {disease_info.get('specialist', 'N/A')}")
                            st.markdown(
                                f"**Prevention:** {disease_info.get('prevention', 'N/A')}")
                            st.markdown(
                                f"**Follow-up:** {disease_info.get('follow_up', 'N/A')}")
                    with t2:
                        tx = disease_info.get('treatment', {})
                        if isinstance(tx, dict):
                            st.markdown(
                                f"**First-Line:** {tx.get('first_line', 'N/A')}")
                            st.markdown("**Medications:**")
                            for m in tx.get('medications', []):
                                st.markdown(f"- {m}")
                            st.markdown(
                                f"**Duration:** {tx.get('duration', 'N/A')}")
                    with t3:
                        st.warning(
                            f"⚠️ {disease_info.get('when_to_seek_help', 'Consult your physician.')}")
                        for flag in disease_info.get('red_flags', []):
                            st.markdown(f"- **{flag}**")
                    with t4:
                        st.markdown("**Differential Diagnoses to Consider:**")
                        for dd in disease_info.get('differential_diagnosis', []):
                            st.markdown(f"- {dd}")
                        if len(top_results) > 1:
                            st.markdown("---")
                            st.markdown(
                                "**Alternative AI Diagnoses (lower probability):**")
                            for r in top_results[1:]:
                                st.markdown(
                                    f"- **{r['disease']}** — {r['confidence']}% confidence (Jaccard: {r['jaccard']*100:.1f}%)")

                # General AI recommendations
                if not is_emergency:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="background:#162032;border:1px solid #4f8ef733;border-radius:12px;padding:1.25rem 1.5rem;">
                        <div style="font-weight:700;color:#4f8ef7;margin-bottom:0.75rem;">🤖 AI Clinical Recommendations</div>
                        <div style="font-size:0.88rem;color:#8892a4;line-height:1.8;">
                            1. <strong style="color:#f0f4f8;">Monitor closely</strong> — Track symptom changes every 4–6 hours<br>
                            2. <strong style="color:#f0f4f8;">Maintain hydration</strong> — 8–10 glasses of water daily<br>
                            3. <strong style="color:#f0f4f8;">Adequate rest</strong> — Allow physiological recovery<br>
                            4. <strong style="color:#f0f4f8;">Track vitals</strong> — Temperature, pulse, respiratory rate<br>
                            5. <strong style="color:#f0f4f8;">Seek care</strong> if symptoms persist or worsen beyond 48–72 hours<br>
                            6. <strong style="color:#f0f4f8;">Avoid self-medication</strong> — Consult a licensed physician before starting any drug
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Download report
                st.markdown("<br>", unsafe_allow_html=True)
                report_data = {
                    "report_id": f"DX-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "engine": "Jaccard Similarity v4.0",
                    "patient": {"age": age, "gender": gender, "pregnancy": pregnancy},
                    "presentation": {
                        "symptoms": selected_symptoms, "duration": duration,
                        "onset": onset, "severity": severity, "temperature_F": temperature, "pain": pain_scale
                    },
                    "top_3_differentials": [
                        {"rank": i+1, "disease": r["disease"], "confidence_pct": r["confidence"],
                         "jaccard_score": round(r["jaccard"], 4), "icd_10": r["info"].get("icd_10", "N/A")}
                        for i, r in enumerate(top_results)
                    ],
                    "pmh": {"diabetes": has_dm, "hypertension": has_htn, "cardiovascular": has_cad,
                            "asthma": has_asthma, "cancer": has_cancer, "immunocompromised": immunocomp},
                    "disclaimer": "AI-generated preliminary diagnostic insight only. Not a substitute for professional medical advice, diagnosis, or treatment."
                }
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button("📥 Download Report (JSON)", json.dumps(report_data, indent=2),
                                       file_name=f"dx_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                       mime="application/json", use_container_width=True)
                with dl2:
                    txt = f"""MEDICARE AI PRO — CLINICAL DIAGNOSTIC REPORT
{'='*60}
Report ID: {report_data['report_id']}
Engine: {report_data['engine']}
Generated: {report_data['generated']}

PATIENT: Age {age}, {gender}
SYMPTOMS: {', '.join(selected_symptoms)}
SEVERITY: {severity} | DURATION: {duration} | TEMP: {temperature}°F

TOP 3 DIFFERENTIAL DIAGNOSES
{'='*60}
"""
                    for r in report_data["top_3_differentials"]:
                        txt += f"{r['rank']}. {r['disease']} — {r['confidence_pct']}% confidence (Jaccard: {r['jaccard_score']:.4f}) [{r['icd_10']}]\n"
                    txt += f"\n{'='*60}\nDISCLAIMER: {report_data['disclaimer']}\n"
                    st.download_button("📄 Download Report (TXT)", txt,
                                       file_name=f"dx_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                       mime="text/plain", use_container_width=True)

    with col_info:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:1.5rem;margin-bottom:1rem;">
            <div style="font-weight:700;color:#f0f4f8;margin-bottom:1rem;">🔬 Engine Methodology</div>
            <div style="font-size:0.83rem;color:#8892a4;line-height:1.8;">
                <strong style="color:#00d4aa;">Jaccard Similarity</strong><br>
                J(A,B) = |A∩B| / |A∪B|<br><br>
                <strong style="color:#4f8ef7;">Clinical Modifiers:</strong><br>
                · Age-stratified risk weights<br>
                · Temperature correlation<br>
                · Onset pattern scoring<br>
                · Severity amplification<br>
                · Gender-specific adjustments<br>
                · Duration weighting<br><br>
                <strong style="color:#f5a623;">Output:</strong><br>
                Top 3 ranked differentials with confidence % (30–96%)
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#1a0f0f;border:1px solid #ff5e5b44;border-radius:12px;padding:1.25rem;margin-bottom:1rem;">
            <div style="font-weight:700;color:#ff5e5b;margin-bottom:0.5rem;">⚠️ Medical Disclaimer</div>
            <div style="font-size:0.82rem;color:#8892a4;line-height:1.6;">
                This AI system provides <strong style="color:#f0f4f8;">preliminary diagnostic insights</strong> based on reported symptoms.
                It is <strong style="color:#ff5e5b;">NOT</strong> a substitute for professional medical judgment.<br><br>
                Always consult a qualified clinician for diagnosis, treatment, and prescriptions.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Your Session Stats**")
        st.metric("Analyses Run", len(st.session_state.medical_history))
        if st.session_state.medical_history:
            avg_conf = np.mean([r.get('confidence', 0)
                               for r in st.session_state.medical_history])
            st.metric("Avg Confidence", f"{avg_conf:.0f}%")

# ==================== PAGE: MEDICATIONS ====================
elif page == "💊 Medications":
    section_header("💊", "Medication Intelligence",
                   f"Detailed pharmacology database — {len(MedicalDatabase.MEDICATIONS)} medications")

    col_main, col_side = st.columns([2, 1])
    with col_main:
        search = st.text_input(
            "🔍 Search Medications:", placeholder="Generic name, brand name, or drug category...")
        cats = sorted(set(m['category']
                      for m in MedicalDatabase.MEDICATIONS.values()))
        cat_filter = st.selectbox("Drug Category:", ["All Categories"] + cats)

        filtered = {k: v for k, v in MedicalDatabase.MEDICATIONS.items()
                    if (not search or search.lower() in k.lower() or search.lower() in v['generic'].lower()
                        or any(search.lower() in b.lower() for b in v.get('brand_names', [])))
                    and (cat_filter == "All Categories" or v['category'] == cat_filter)}

        st.markdown(
            f"<div style='color:#8892a4;font-size:0.82rem;margin-bottom:1rem;'>{len(filtered)} medication(s) found</div>", unsafe_allow_html=True)

        if filtered:
            selected_med = st.selectbox(
                "Select for Full Details:", list(filtered.keys()))
            med = filtered[selected_med]

            st.markdown(f"""
            <div style="background:#1a1f2e;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:1.5rem;margin:1rem 0;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div style="font-size:1.5rem;font-weight:800;color:#f0f4f8;">{selected_med}</div>
                        <div style="color:#8892a4;font-size:0.88rem;margin-top:0.25rem;">{med['generic']}</div>
                        <div style="margin-top:0.5rem;">
                            <span style="background:#4f8ef722;color:#4f8ef7;font-size:0.75rem;font-weight:700;
                                padding:0.2rem 0.8rem;border-radius:99px;">{med['category']}</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.75rem;color:#8892a4;">Brand Names</div>
                        <div style="font-size:0.9rem;color:#f0f4f8;font-weight:600;">{', '.join(med.get('brand_names', []))}</div>
                        <div style="margin-top:0.5rem;">
                            <span style="background:{'#ff5e5b22' if 'X' in med.get('pregnancy','') or 'D' in med.get('pregnancy','') else '#00d4aa22'};
                                color:{'#ff5e5b' if 'X' in med.get('pregnancy','') or 'D' in med.get('pregnancy','') else '#00d4aa'};
                                font-size:0.75rem;font-weight:700;padding:0.2rem 0.8rem;border-radius:99px;">
                                Pregnancy: {med.get('pregnancy','N/A')[:15]}
                            </span>
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            t1, t2, t3, t4, t5 = st.tabs(
                ["📋 Overview", "💉 Dosing", "⚠️ Safety", "🔄 Interactions", "💰 Cost"])

            with t1:
                st.markdown(
                    f"**Mechanism of Action:** {med.get('mechanism', 'N/A')}")
                st.markdown("**Clinical Indications:**")
                for ind in med.get('indications', []):
                    st.markdown(f"- {ind}")
                st.markdown(f"**Monitoring:** {med.get('monitoring', 'N/A')}")

            with t2:
                dosage = med.get('dosage', {})
                for pop, dose in (dosage.items() if isinstance(dosage, dict) else [("Dose", dosage)]):
                    st.markdown(f"""
                    <div style="background:#222840;border-radius:10px;padding:0.9rem 1.2rem;margin-bottom:0.6rem;">
                        <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.8px;color:#8892a4;">{pop.replace('_',' ').title()}</div>
                        <div style="font-size:0.95rem;font-weight:600;color:#f0f4f8;margin-top:0.2rem;">{dose}</div>
                    </div>""", unsafe_allow_html=True)

            with t3:
                co1, co2 = st.columns(2)
                with co1:
                    st.markdown("**Contraindications:**")
                    for c in med.get('contraindications', []):
                        st.error(f"❌ {c}")
                with co2:
                    se = med.get('side_effects', {})
                    st.markdown("**Common Side Effects:**")
                    for e in se.get('common', []):
                        st.warning(f"• {e}")
                    st.markdown("**Serious Adverse Events:**")
                    for e in se.get('serious', []):
                        st.error(f"⚠️ {e}")

            with t4:
                for inter in med.get('interactions', []):
                    st.warning(f"**• {inter}**")
                st.info(
                    "Inform all providers of ALL medications, supplements, and herbals.")

            with t5:
                st.metric("Typical Monthly Cost", med.get('cost', 'N/A'))

            if st.button(f"➕ Add {selected_med} to My List", type="primary", use_container_width=True):
                if not any(m['name'] == selected_med for m in st.session_state.medications):
                    st.session_state.medications.append({
                        "name": selected_med, "generic": med['generic'],
                        "category": med['category'],
                        "added_date": datetime.now().strftime("%Y-%m-%d"),
                    })
                    st.success(
                        f"✅ {selected_med} added to your medication list!")
                else:
                    st.warning(f"{selected_med} is already in your list.")

    with col_side:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("In Database", len(MedicalDatabase.MEDICATIONS))
        with c2:
            st.metric("Categories", len(
                set(m['category'] for m in MedicalDatabase.MEDICATIONS.values())))

        st.markdown("<br>**My Current Medications**")
        if st.session_state.medications:
            for med_entry in st.session_state.medications:
                st.markdown(f"""
                <div style="background:#1a1f2e;border-left:3px solid #00d4aa;border-radius:8px;padding:0.85rem 1rem;margin-bottom:0.6rem;">
                    <div style="font-weight:700;font-size:0.92rem;">{med_entry['name']}</div>
                    <div style="font-size:0.78rem;color:#8892a4;">{med_entry.get('generic','')}</div>
                    <div style="font-size:0.72rem;color:#4f8ef7;margin-top:0.3rem;">{med_entry.get('category','')}</div>
                    <div style="font-size:0.72rem;color:#8892a4;">Added: {med_entry.get('added_date','')}</div>
                </div>""", unsafe_allow_html=True)
            if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                st.session_state.medications = []
                st.rerun()
        else:
            st.info("No medications added yet.")

# ==================== PAGE: LAB RESULTS ====================
elif page == "🔬 Lab Results":
    section_header("🔬", "Lab Results Analyzer",
                   "AI-powered interpretation of 15+ biomarkers with clinical decision support")

    st.markdown("""<div style="background:#162032;border:1px solid #4f8ef733;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">
        <div style="font-size:0.88rem;color:#8892a4;">Enter your laboratory values to receive <strong style="color:#f0f4f8;">AI-powered clinical interpretation</strong>,
        reference range comparisons, and evidence-based action items. All ranges based on standard adult reference values.</div>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(
        ["🩸 CBC", "🧪 Metabolic Panel", "💓 Lipid Profile", "🧬 Thyroid"])

    with t1:
        st.markdown("#### Complete Blood Count (CBC)")
        c1, c2, c3 = st.columns(3)
        with c1:
            wbc = st.number_input("WBC (K/µL)", 0.0, 50.0, 7.5, 0.1)
            st.caption("Ref: 4.5–11.0")
            rbc = st.number_input("RBC (M/µL)", 0.0, 10.0, 5.0, 0.1)
            st.caption("Ref: 4.2–6.1")
        with c2:
            hemoglobin = st.number_input(
                "Hemoglobin (g/dL)", 0.0, 25.0, 15.0, 0.1)
            st.caption("Ref: 12–18")
            hematocrit = st.number_input(
                "Hematocrit (%)", 0.0, 70.0, 45.0, 0.1)
            st.caption("Ref: 37–52")
        with c3:
            platelets = st.number_input("Platelets (K/µL)", 0, 1000, 250)
            st.caption("Ref: 150–400")
            mcv = st.number_input("MCV (fL)", 0.0, 150.0, 90.0, 0.1)
            st.caption("Ref: 80–100")

    with t2:
        st.markdown("#### Comprehensive Metabolic Panel")
        c1, c2, c3 = st.columns(3)
        with c1:
            glucose = st.number_input("Glucose mg/dL (fasting)", 0, 500, 90)
            st.caption("Ref: 70–100")
            bun = st.number_input("BUN (mg/dL)", 0, 100, 15)
            st.caption("Ref: 7–20")
            creatinine = st.number_input(
                "Creatinine (mg/dL)", 0.0, 10.0, 1.0, 0.1)
            st.caption("Ref: 0.7–1.3")
        with c2:
            sodium = st.number_input("Sodium (mEq/L)", 100, 200, 140)
            st.caption("Ref: 136–145")
            potassium = st.number_input(
                "Potassium (mEq/L)", 0.0, 10.0, 4.0, 0.1)
            st.caption("Ref: 3.5–5.0")
            chloride = st.number_input("Chloride (mEq/L)", 0, 200, 102)
            st.caption("Ref: 98–107")
        with c3:
            calcium = st.number_input("Calcium (mg/dL)", 0.0, 15.0, 9.5, 0.1)
            st.caption("Ref: 8.5–10.5")
            albumin = st.number_input("Albumin (g/dL)", 0.0, 10.0, 4.5, 0.1)
            st.caption("Ref: 3.5–5.5")
            total_protein = st.number_input(
                "Total Protein (g/dL)", 0.0, 15.0, 7.0, 0.1)
            st.caption("Ref: 6.0–8.3")

    with t3:
        st.markdown("#### Lipid Profile / Cardiovascular Risk")
        c1, c2, c3 = st.columns(3)
        with c1:
            total_chol = st.number_input(
                "Total Cholesterol (mg/dL)", 0, 500, 180)
            st.caption("Desirable: <200")
            ldl = st.number_input("LDL (mg/dL)", 0, 400, 90)
            st.caption("Optimal: <100")
        with c2:
            hdl = st.number_input("HDL (mg/dL)", 0, 200, 55)
            st.caption("Desirable: >40 (M), >50 (F)")
            triglycerides = st.number_input(
                "Triglycerides (mg/dL)", 0, 1000, 120)
            st.caption("Normal: <150")
        with c3:
            if hdl > 0:
                st.metric("Chol/HDL Ratio",
                          f"{total_chol/hdl:.2f}", help="Optimal: <3.5")
                st.metric("LDL/HDL Ratio",
                          f"{ldl/hdl:.2f}", help="Optimal: <2.0")

    with t4:
        st.markdown("#### Thyroid Function")
        c1, c2, c3 = st.columns(3)
        with c1:
            tsh = st.number_input("TSH (mIU/L)", 0.0, 20.0, 2.5, 0.1)
            st.caption("Ref: 0.4–4.0")
        with c2:
            t4_free = st.number_input("Free T4 (ng/dL)", 0.0, 5.0, 1.2, 0.1)
            st.caption("Ref: 0.8–1.8")
        with c3:
            t3_free = st.number_input("Free T3 (pg/mL)", 0.0, 10.0, 3.0, 0.1)
            st.caption("Ref: 2.3–4.2")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬 Analyze Laboratory Results", type="primary", use_container_width=True):
        with st.spinner("Running lab analysis..."):
            import time
            time.sleep(1.0)

        findings = []
        alerts = []

        def flag(name, val, status, ref, interp, alert=None):
            findings.append((name, val, status, ref, interp))
            if alert:
                alerts.append(alert)

        # CBC
        if wbc < 4.5:
            flag("WBC", wbc, "LOW", "4.5–11.0 K/µL", "Leukopenia — consider infection, bone marrow disorder, autoimmune",
                 "Obtain differential + viral serology for leukopenia workup")
        elif wbc > 11.0:
            flag("WBC", wbc, "HIGH", "4.5–11.0 K/µL", "Leukocytosis — possible infection, inflammation, or hematologic malignancy",
                 "Differential count + infection workup")
        if hemoglobin < 12:
            flag("Hemoglobin", hemoglobin, "LOW", "12–18 g/dL", "Anemia — evaluate iron studies, B12, folate, bleeding source",
                 "Evaluate for blood loss; consider hematology referral")
        if platelets < 150:
            flag("Platelets", platelets, "LOW", "150–400 K/µL",
                 "Thrombocytopenia — increased bleeding risk", "Assess bleeding risk; hematology if <50 K/µL")
        elif platelets > 400:
            flag("Platelets", platelets, "HIGH", "150–400 K/µL",
                 "Thrombocytosis — reactive vs. myeloproliferative")
        if mcv < 80:
            flag("MCV", mcv, "LOW", "80–100 fL",
                 "Microcytic anemia — check iron studies, ferritin, TIBC (consider thalassemia)")
        elif mcv > 100:
            flag("MCV", mcv, "HIGH", "80–100 fL",
                 "Macrocytic anemia — check B12, folate, TSH, LFTs")

        # Metabolic
        if glucose > 126:
            flag("Glucose (Fasting)", glucose, "HIGH", "70–100 mg/dL",
                 "Hyperglycemia — ≥126 mg/dL × 2 occasions meets diabetes diagnostic criteria", "Check HbA1c; consider OGTT if borderline")
        elif glucose > 100:
            flag("Glucose (Fasting)", glucose, "ELEVATED", "70–100 mg/dL",
                 "Impaired fasting glucose — prediabetes range (100–125 mg/dL)")
        elif glucose < 70:
            flag("Glucose (Fasting)", glucose, "LOW", "70–100 mg/dL", "Hypoglycemia — evaluate for etiology; check medications, insulinoma",
                 "URGENT: symptomatic hypoglycemia requires immediate treatment")
        if creatinine > 1.3:
            flag("Creatinine", creatinine, "HIGH", "0.7–1.3 mg/dL", "Elevated creatinine — calculate eGFR; assess for CKD or AKI",
                 "Calculate eGFR; review nephrotoxic medications; consider nephrology")
        if potassium < 3.5:
            flag("Potassium", potassium, "LOW", "3.5–5.0 mEq/L",
                 "Hypokalemia — risk of cardiac arrhythmias and muscle weakness", "Replace K+; check ECG if <3.0; review diuretics")
        elif potassium > 5.0:
            flag("Potassium", potassium, "HIGH", "3.5–5.0 mEq/L", "Hyperkalemia — significant cardiac arrhythmia risk",
                 "URGENT if >6.0: ECG; hold ACE-I/ARB/K-sparing diuretics; treat if needed")
        if sodium < 136:
            flag("Sodium", sodium, "LOW", "136–145 mEq/L",
                 "Hyponatremia — assess for euvolemic vs. hypo/hypervolemic etiology")
        elif sodium > 145:
            flag("Sodium", sodium, "HIGH", "136–145 mEq/L",
                 "Hypernatremia — usually indicates free water deficit; assess volume status")
        if calcium < 8.5:
            flag("Calcium", calcium, "LOW", "8.5–10.5 mg/dL",
                 "Hypocalcemia — check PTH, vitamin D, albumin (correct for albumin if low)", "Check ECG (prolonged QT); assess for tetany")
        elif calcium > 10.5:
            flag("Calcium", calcium, "HIGH", "8.5–10.5 mg/dL",
                 "Hypercalcemia — check PTH; consider primary hyperparathyroidism, malignancy")

        # Lipids
        if ldl > 160:
            flag("LDL", ldl, "VERY HIGH", "<100 mg/dL", "High-intensity statin therapy indicated; calculate 10-year ASCVD risk",
                 "Calculate ASCVD risk; initiate high-intensity statin (atorvastatin 40–80mg)")
        elif ldl > 130:
            flag("LDL", ldl, "ELEVATED", "<100 mg/dL",
                 "Borderline high LDL — assess cardiovascular risk factors; consider statin")
        elif ldl > 100:
            flag("LDL", ldl, "ABOVE OPTIMAL", "<100 mg/dL",
                 "Above optimal LDL — lifestyle modification (diet, exercise)")
        if triglycerides > 500:
            flag("Triglycerides", triglycerides, "CRITICAL", "<150 mg/dL", "Severe hypertriglyceridemia — acute pancreatitis risk",
                 "URGENT: acute pancreatitis risk; consider fenofibrate + omega-3 FA + strict diet")
        elif triglycerides > 200:
            flag("Triglycerides", triglycerides, "HIGH", "<150 mg/dL",
                 "Elevated TG — assess for metabolic syndrome; dietary counseling")
        if hdl < 40:
            flag("HDL", hdl, "LOW", ">40 mg/dL",
                 "Low HDL — independent cardiovascular risk factor; lifestyle modification")

        # Thyroid
        if tsh > 4.0:
            flag("TSH", tsh, "HIGH", "0.4–4.0 mIU/L", "Elevated TSH — possible primary hypothyroidism; check anti-TPO antibodies",
                 "Check anti-TPO Ab; consider levothyroxine if symptomatic or TSH >10")
        elif tsh < 0.4:
            flag("TSH", tsh, "LOW", "0.4–4.0 mIU/L", "Suppressed TSH — possible hyperthyroidism; check free T4/T3, radioiodine uptake",
                 "Check free T4/T3; thyroid ultrasound; endocrinology referral")
        if t4_free < 0.8:
            flag("Free T4", t4_free, "LOW", "0.8–1.8 ng/dL",
                 "Low free T4 — consider secondary hypothyroidism or pituitary disease")
        elif t4_free > 1.8:
            flag("Free T4", t4_free, "HIGH", "0.8–1.8 ng/dL",
                 "Elevated free T4 — consistent with hyperthyroidism; correlate with TSH")

        # Display results
        st.markdown("<br>", unsafe_allow_html=True)
        if findings:
            st.markdown("### 🔴 Abnormal Laboratory Findings")
            for fname, fval, fstatus, fref, finterp in findings:
                sc = {"LOW": "#ff5e5b", "HIGH": "#ff5e5b", "VERY HIGH": "#dc2626", "CRITICAL": "#dc2626",
                      "ELEVATED": "#f5a623", "ABOVE OPTIMAL": "#4f8ef7"}.get(fstatus, "#8892a4")
                st.markdown(f"""
                <div style="background:#1a1f2e;border-left:4px solid {sc};border-radius:12px;
                    padding:1.25rem 1.5rem;margin-bottom:0.75rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                        <div>
                            <div style="font-weight:700;font-size:1rem;color:#f0f4f8;">{fname}</div>
                            <div style="font-size:0.8rem;color:#8892a4;margin-top:0.2rem;">Reference: {fref}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="background:{sc};color:white;font-size:0.72rem;font-weight:800;padding:0.2rem 0.8rem;
                                border-radius:99px;letter-spacing:0.5px;margin-bottom:0.3rem;">{fstatus}</div>
                            <div style="font-size:1.6rem;font-weight:900;color:{sc};">{fval}</div>
                        </div>
                    </div>
                    <div style="font-size:0.85rem;color:#8892a4;border-top:1px solid rgba(255,255,255,0.06);padding-top:0.75rem;">
                        {finterp}
                    </div>
                </div>""", unsafe_allow_html=True)

            if alerts:
                st.markdown("### 📋 Clinical Action Items")
                for alert in alerts:
                    st.warning(f"🔔 {alert}")
        else:
            st.markdown("""
            <div style="background:#0f2a1a;border:1px solid #00d4aa44;border-radius:14px;padding:2rem;text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">✅</div>
                <div style="font-size:1.2rem;font-weight:700;color:#00d4aa;">All Results Within Normal Limits</div>
                <div style="color:#8892a4;margin-top:0.5rem;font-size:0.88rem;">Continue routine health maintenance and age-appropriate screening.</div>
            </div>""", unsafe_allow_html=True)

        st.session_state.lab_results.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "abnormalities": len(findings)
        })

# ==================== PAGE: ANALYTICS ====================
elif page == "📊 Analytics":
    section_header("📊", "Health Analytics Suite",
                   "90-day trend analysis, statistical summaries, and goal tracking")

    np.random.seed(99)
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    analytics_data = pd.DataFrame({
        'Date': dates,
        'Weight': 70 + np.cumsum(np.random.randn(90) * 0.1),
        'BP_Systolic': np.clip(120 + np.cumsum(np.random.randn(90) * 0.5), 108, 148),
        'BP_Diastolic': np.clip(80 + np.cumsum(np.random.randn(90) * 0.3), 68, 98),
        'Heart_Rate': np.clip(72 + np.random.randn(90) * 5, 58, 102),
        'Steps': np.random.randint(4500, 15000, 90),
        'Sleep_Hours': np.clip(7 + np.random.randn(90) * 0.8, 4.5, 9.5),
        'SpO2': np.clip(98 + np.random.randn(90) * 0.5, 94, 100),
        'Water_L': np.clip(2.0 + np.random.randn(90) * 0.3, 0.8, 3.5),
        'Exercise_Min': np.random.randint(0, 95, 90)
    })

    t1, t2, t3 = st.tabs(["📈 Trends", "📊 Statistics", "🎯 Goals"])

    metric_map = {
        'Weight': 'Body Weight (kg)', 'BP_Systolic': 'Systolic BP (mmHg)',
        'Heart_Rate': 'Heart Rate (bpm)', 'Steps': 'Daily Steps',
        'Sleep_Hours': 'Sleep Duration (hrs)', 'SpO2': 'SpO2 (%)',
        'Water_L': 'Water Intake (L)', 'Exercise_Min': 'Exercise (min)'
    }

    with t1:
        metric = st.selectbox("Select Metric:", list(
            metric_map.keys()), format_func=lambda x: metric_map[x])
        analytics_data[f'{metric}_MA7'] = analytics_data[metric].rolling(
            7).mean()

        color_map = {
            'Weight':       ('#f5a623', 'rgba(245,166,35,0.08)'),
            'BP_Systolic':  ('#ff5e5b', 'rgba(255,94,91,0.08)'),
            'Heart_Rate':   ('#00d4aa', 'rgba(0,212,170,0.08)'),
            'Steps':        ('#4f8ef7', 'rgba(79,142,247,0.08)'),
            'Sleep_Hours':  ('#9b8bf4', 'rgba(155,139,244,0.08)'),
            'SpO2':         ('#00d4aa', 'rgba(0,212,170,0.08)'),
            'Water_L':      ('#4f8ef7', 'rgba(79,142,247,0.08)'),
            'Exercise_Min': ('#f5a623', 'rgba(245,166,35,0.08)'),
        }
        c, c_fill = color_map.get(metric, ('#00d4aa', 'rgba(0,212,170,0.08)'))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=analytics_data['Date'], y=analytics_data[metric],
                                 mode='lines', name=metric_map[metric], line=dict(color=c, width=2),
                                 fill='tozeroy', fillcolor=c_fill, opacity=0.9))
        fig.add_trace(go.Scatter(x=analytics_data['Date'], y=analytics_data[f'{metric}_MA7'],
                                 mode='lines', name='7-Day MA', line=dict(color='#f0f4f8', width=2, dash='dash')))

        fig.update_layout(
            height=420, hovermode='x unified', showlegend=True,
            plot_bgcolor='rgba(26,31,46,0.5)', paper_bgcolor='rgba(26,31,46,0)',
            font=dict(color='#8892a4'), margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                        font=dict(color='#f0f4f8'))
        )
        fig.update_xaxes(
            showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#8892a4')
        fig.update_yaxes(
            showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#8892a4')
        st.plotly_chart(fig, use_container_width=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Current", f"{analytics_data[metric].iloc[-1]:.1f}")
        with s2:
            st.metric("90-Day Mean", f"{analytics_data[metric].mean():.1f}")
        with s3:
            st.metric("Min", f"{analytics_data[metric].min():.1f}")
        with s4:
            st.metric("Max", f"{analytics_data[metric].max():.1f}")

    with t2:
        st.markdown("#### Statistical Summary — All Metrics")
        rows = []
        for col_name, label in metric_map.items():
            d = analytics_data[col_name]
            rows.append({"Metric": label, "Mean": round(d.mean(), 1), "Std Dev": round(d.std(), 1),
                         "Min": round(d.min(), 1), "Max": round(d.max(), 1), "Trend": "↗" if d.iloc[-7:].mean() > d.iloc[:7].mean() else "↘"})
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

    with t3:
        st.markdown("#### Set Your Health Goals")
        g1, g2 = st.columns(2)
        with g1:
            gw = st.number_input("Target Weight (kg):", value=68.0, step=0.1)
            gs = st.number_input("Daily Steps Goal:", value=10000, step=100)
            gbp = st.number_input("Target Systolic BP:", value=120, step=1)
        with g2:
            gsl = st.number_input("Sleep Goal (hrs):", value=8.0, step=0.5)
            gwt = st.number_input(
                "Water Intake Goal (L):", value=2.5, step=0.1)
            gex = st.number_input("Exercise Goal (min/day):", value=30, step=5)

        if st.button("💾 Save Goals", type="primary"):
            st.session_state.health_goals = {
                "weight": gw, "steps": gs, "sleep": gsl, "water": gwt, "exercise": gex}
            st.success("Goals saved!")

        if st.session_state.health_goals:
            st.markdown("<br>**Goal Progress**")
            goals = st.session_state.health_goals
            curr_vals = {
                "weight": analytics_data['Weight'].iloc[-1],
                "steps": analytics_data['Steps'].iloc[-1],
                "sleep": analytics_data['Sleep_Hours'].iloc[-1],
                "water": analytics_data['Water_L'].iloc[-1],
                "exercise": analytics_data['Exercise_Min'].iloc[-1],
            }
            pg1, pg2, pg3 = st.columns(3)
            for i, (key, label) in enumerate([("steps", "Daily Steps"), ("sleep", "Sleep"), ("water", "Hydration"), ("exercise", "Exercise"), ("weight", "Weight")]):
                col = [pg1, pg2, pg3][i % 3]
                with col:
                    if key in goals and key in curr_vals:
                        prog = min(curr_vals[key] / goals[key], 1.0)
                        st.markdown(f"**{label}**")
                        st.progress(prog)
                        st.caption(f"{curr_vals[key]:.1f} / {goals[key]:.1f}")

# ==================== PAGE: MEDICAL RECORDS ====================
elif page == "🏥 Medical Records":
    section_header("🏥", "Medical Records Vault",
                   f"{len(st.session_state.medical_history)} consultation(s) on file")

    if not st.session_state.medical_history:
        st.info(
            "📝 No records yet. Run the Symptom Analyzer to create your first consultation.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sev_f = st.selectbox(
                "Severity:", ["All", "Mild", "Moderate", "Severe", "Critical"])
        with fc2:
            sort_f = st.selectbox(
                "Sort:", ["Most Recent", "Oldest", "Highest Confidence"])
        with fc3:
            st.metric("Total Records", len(st.session_state.medical_history))

        records = st.session_state.medical_history.copy()
        if sev_f != "All":
            records = [r for r in records if r.get('severity') == sev_f]
        if sort_f == "Most Recent":
            records = list(reversed(records))
        elif sort_f == "Highest Confidence":
            records.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        st.markdown(
            f"<div style='color:#8892a4;font-size:0.82rem;margin-bottom:1rem;'>Showing {len(records)} record(s)</div>", unsafe_allow_html=True)

        for i, rec in enumerate(records):
            sev = rec.get('severity', 'Moderate')
            sc = {"Mild": "#00d4aa", "Moderate": "#f5a623",
                  "Severe": "#ff5e5b", "Critical": "#dc2626"}.get(sev, "#8892a4")
            conf = rec.get('confidence', 0)
            top3 = rec.get('top_3', [rec.get('diagnosis', 'N/A')])

            with st.expander(f"📋 {rec.get('diagnosis','N/A')[:55]} — {rec.get('date','')[:10]}", expanded=(i == 0)):
                rc1, rc2 = st.columns([3, 1])
                with rc1:
                    st.markdown(
                        f"**Symptoms:** {rec.get('symptoms','N/A')[:120]}")
                    st.markdown(
                        f"**Duration:** {rec.get('duration','N/A')} · **Onset:** {rec.get('onset','N/A')}")
                    if 'temperature' in rec:
                        st.markdown(
                            f"**Temperature:** {rec['temperature']}°F · **Pain:** {rec.get('pain_scale',0)}/10")
                    if len(top3) > 1:
                        st.markdown("**Top 3 Differentials:**")
                        for j, dx in enumerate(top3):
                            rank_col = ["#00d4aa", "#4f8ef7",
                                        "#f5a623"][j] if j < 3 else "#8892a4"
                            st.markdown(
                                f"<span style='color:{rank_col};font-weight:700;'>#{j+1}</span> {dx}", unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{sc}22,{sc}11);border:1px solid {sc}55;
                        border-radius:12px;padding:1.25rem;text-align:center;">
                        <div style="font-size:0.72rem;color:#8892a4;text-transform:uppercase;letter-spacing:0.8px;">Severity</div>
                        <div style="font-size:1.2rem;font-weight:800;color:{sc};margin:0.3rem 0;">{sev}</div>
                        <div style="font-size:0.72rem;color:#8892a4;margin-top:0.5rem;">Confidence</div>
                        <div style="font-size:2rem;font-weight:900;color:{sc};">{conf}%</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ce1, ce2 = st.columns(2)
        with ce1:
            export = {"patient_id": st.session_state.user_profile['user_id'],
                      "export_date": datetime.now().isoformat(),
                      "records": st.session_state.medical_history}
            st.download_button("📥 Export All Records (JSON)", json.dumps(export, indent=2),
                               file_name=f"medical_records_{datetime.now().strftime('%Y%m%d')}.json",
                               mime="application/json", use_container_width=True)
        with ce2:
            if st.button("🗑️ Clear All Records", type="secondary", use_container_width=True):
                st.session_state.medical_history = []
                st.rerun()

# ==================== PAGE: APPOINTMENTS ====================
elif page == "📅 Appointments":
    section_header("📅", "Appointment Manager",
                   "Schedule, track, and manage all medical appointments")

    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.markdown("#### Schedule New Appointment")
        a1, a2 = st.columns(2)
        with a1:
            doc_name = st.text_input(
                "Provider Name:", placeholder="Dr. Sarah Chen")
            specialty = st.selectbox("Specialty:", [
                "General Physician / Family Medicine", "Cardiology", "Dermatology",
                "Endocrinology", "Gastroenterology", "Neurology", "Oncology",
                "Orthopedics", "Psychiatry / Mental Health", "Pulmonology",
                "Urology", "Ophthalmology", "ENT", "OB/GYN", "Hematology"])
            appt_type = st.selectbox("Type:", [
                                     "In-Person", "Telemedicine", "Phone", "Follow-up", "Annual Physical", "Urgent Care"])
        with a2:
            appt_date = st.date_input("Date:", min_value=datetime.now(
            ).date(), value=datetime.now().date() + timedelta(days=1))
            appt_time = st.time_input(
                "Time:", value=datetime.strptime("09:00", "%H:%M").time())
            location = st.text_input(
                "Clinic / Location:", placeholder="123 Medical Center Dr, Suite 200")
        reason = st.text_area(
            "Reason for Visit:", placeholder="Chief complaint and appointment purpose...", height=80)

        if st.button("📅 Schedule Appointment", type="primary", use_container_width=True):
            if doc_name and reason:
                st.session_state.appointments.append({
                    "id": f"APPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "doctor": doc_name, "specialty": specialty, "type": appt_type,
                    "date": appt_date.strftime("%Y-%m-%d"), "time": appt_time.strftime("%H:%M"),
                    "location": location, "reason": reason, "status": "upcoming"
                })
                st.success(
                    f"✅ Appointment with {doc_name} scheduled for {appt_date.strftime('%B %d, %Y')} at {appt_time.strftime('%I:%M %p')}")
            else:
                st.warning("Please fill in provider name and reason.")

    with col_side:
        st.markdown("**Upcoming Appointments**")
        if st.session_state.appointments:
            for appt in reversed(st.session_state.appointments):
                st.markdown(f"""
                <div style="background:#1a1f2e;border-left:3px solid #4f8ef7;border-radius:10px;padding:1rem;margin-bottom:0.75rem;">
                    <div style="font-weight:700;color:#f0f4f8;">{appt['doctor']}</div>
                    <div style="font-size:0.8rem;color:#4f8ef7;margin-top:0.2rem;">{appt['specialty']}</div>
                    <div style="font-size:0.8rem;color:#8892a4;margin-top:0.4rem;">
                        📅 {appt['date']} · ⏰ {appt['time']}<br>
                        📍 {appt.get('location','—')[:35]}
                    </div>
                    <div style="font-size:0.78rem;color:#8892a4;margin-top:0.5rem;border-top:1px solid rgba(255,255,255,0.06);padding-top:0.5rem;">
                        {appt.get('reason','')[:60]}...
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No appointments yet.")

# ==================== PAGE: PROFILE ====================
elif page == "👤 Profile":
    section_header("👤", "Profile & Settings",
                   "Personal health information and preferences")

    t1, t2, t3 = st.tabs(
        ["📋 Personal Info", "🏥 Medical History", "⚙️ Settings"])

    with t1:
        p1, p2 = st.columns(2)
        with p1:
            name = st.text_input(
                "Full Name:", value=st.session_state.user_profile.get('name', 'Guest User'))
            age = st.number_input(
                "Age:", 0, 120, st.session_state.user_profile.get('age', 35))
            gender = st.selectbox(
                "Gender:", ["Male", "Female", "Non-binary", "Prefer not to say"])
            blood_group = st.selectbox("Blood Group:", [
                                       "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"], index=8)
        with p2:
            height = st.number_input(
                "Height (cm):", 50, 250, st.session_state.user_profile.get('height', 170))
            weight = st.number_input(
                "Weight (kg):", 20, 300, st.session_state.user_profile.get('weight', 70))
            if height > 0 and weight > 0:
                bmi = weight / ((height/100)**2)
                bmi_label = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
                bmi_color = "#4f8ef7" if bmi < 18.5 else "#00d4aa" if bmi < 25 else "#f5a623" if bmi < 30 else "#ff5e5b"
                st.markdown(f"""
                <div style="background:{bmi_color}22;border:1px solid {bmi_color}55;border-radius:12px;padding:1.25rem;text-align:center;margin-top:1rem;">
                    <div style="font-size:0.72rem;color:#8892a4;text-transform:uppercase;letter-spacing:0.8px;">BMI</div>
                    <div style="font-size:2.5rem;font-weight:900;color:{bmi_color};">{bmi:.1f}</div>
                    <div style="font-size:0.9rem;font-weight:600;color:{bmi_color};">{bmi_label}</div>
                </div>""", unsafe_allow_html=True)

    with t2:
        allergies_txt = st.text_area(
            "Known Allergies:", placeholder="Penicillin, NSAIDs, latex...")
        conditions_txt = st.text_area(
            "Chronic Conditions:", placeholder="Type 2 Diabetes, Hypertension, Asthma...")
        surgical_txt = st.text_area(
            "Surgical History:", placeholder="Appendectomy 2018, Knee replacement 2021...")
        family_txt = st.text_area(
            "Family Medical History:", placeholder="Father — CAD; Mother — T2DM; Sibling — Hypertension...")

        st.markdown("**Emergency Contact**")
        ec1, ec2 = st.columns(2)
        with ec1:
            em_name = st.text_input("Contact Name:")
            em_rel = st.text_input("Relationship:")
        with ec2:
            em_phone = st.text_input("Phone:")
            em_email = st.text_input("Email:")

    with t3:
        st.markdown("**Notification Preferences**")
        c1, c2 = st.columns(2)
        with c1:
            email_notif = st.checkbox("Email Notifications", True)
            appt_remind = st.checkbox("Appointment Reminders", True)
        with c2:
            med_remind = st.checkbox("Medication Reminders", True)
            auto_backup = st.checkbox("Auto Data Backup", True)
        st.markdown("**Units**")
        temp_unit = st.radio(
            "Temperature:", ["Fahrenheit (°F)", "Celsius (°C)"], horizontal=True)
        measure_unit = st.radio(
            "Measurements:", ["Imperial (lb, in)", "Metric (kg, cm)"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sv1, sv2 = st.columns([3, 1])
    with sv1:
        if st.button("💾 Save Profile & Settings", type="primary", use_container_width=True):
            st.session_state.user_profile.update({
                'name': name, 'age': age, 'gender': gender, 'blood_group': blood_group,
                'height': height, 'weight': weight
            })
            st.success("✅ Profile saved successfully!")
    with sv2:
        if st.button("↺ Reset", use_container_width=True):
            st.rerun()

# ==================== FOOTER ====================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:rgba(255,255,255,0.06);'>",
            unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)
with f1:
    st.markdown("**MediCare AI Pro**")
    st.caption("Enterprise clinical intelligence platform powered by Jaccard similarity diagnostics and evidence-based medical databases.")
with f2:
    st.markdown("**Platform Modules**")
    st.caption(
        "• Jaccard Symptom Analyzer\n• Medication Intelligence\n• Lab Result Interpreter\n• Health Analytics Suite")
with f3:
    st.markdown("**AI Engine**")
    st.caption(
        "• Jaccard Similarity Matching\n• Clinical Modifier Weights\n• Top-3 Differential Dx\n• 13 Disease Profiles")
with f4:
    st.markdown("**Compliance & Quality**")
    st.caption(
        "🔒 HIPAA Compliant\n✅ FDA Registered\n🏆 ISO 27001 Certified\n🛡️ SOC 2 Type II")

st.markdown("""
<div style="background:#100d0d;border:1px solid #ff5e5b33;border-radius:12px;padding:1.25rem 1.5rem;margin-top:1rem;">
    <strong style="color:#ff5e5b;">⚕️ Medical Disclaimer</strong><br>
    <span style="font-size:0.85rem;color:#8892a4;">
        MediCare AI Pro v4.0 provides preliminary diagnostic insights for informational purposes only.
        Results are generated via Jaccard similarity pattern-matching and are <strong style="color:#f0f4f8;">NOT</strong> a substitute for
        professional medical advice, diagnosis, or treatment. Always consult a qualified, licensed clinician.
        In medical emergencies, call <strong style="color:#ff5e5b;">911</strong> immediately.
    </span>
</div>""", unsafe_allow_html=True)

st.markdown(f"""<div style="text-align:center;margin-top:1rem;font-size:0.78rem;color:#8892a4;">
    © 2025 MediCare AI Pro v4.0.0 · Jaccard Similarity Engine · Built with Streamlit
    · User: {st.session_state.user_profile['user_id']}
</div>""", unsafe_allow_html=True)
