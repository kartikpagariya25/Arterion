import os
import json
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_model = None

def _get_model():
    global _model
    if _model is None:
        if not GEMINI_API_KEY:
            return None
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel("gemini-2.0-flash")
    return _model

def _lang_block(summary: str, what_it_means: str, suggestions: list) -> dict:
    return {"summary": summary, "what_it_means": what_it_means, "lifestyle_suggestions": suggestions}

def _fallback_report(analysis: dict) -> dict:
    lesions = analysis.get("lesions", [])
    disclaimer_en = "This is an AI-assisted screening result, not a medical diagnosis. Please consult a cardiologist for confirmation."
    disclaimer_hi = "यह एक AI-सहायता प्राप्त स्क्रीनिंग परिणाम है, चिकित्सीय निदान नहीं। कृपया पुष्टि के लिए हृदय रोग विशेषज्ञ से सलाह लें।"
    disclaimer_mr = "हा एक AI-सहाय्यित स्क्रीनिंग निकाल आहे, वैद्यकीय निदान नाही. कृपया खात्रीसाठी हृदयरोगतज्ञांचा सल्ला घ्या."

    if not lesions:
        return {
            "doctor_report": {
                "clinical_summary": "No stenotic lesions were identified above the model's detection threshold on the analyzed frame.",
                "findings": f"Vessel mask present: {analysis.get('vessel_mask_present', False)}. No regions met the criteria for stenosis classification.",
                "recommendation": "Clinical correlation is recommended. Further evaluation is advised if the patient remains symptomatic despite this screening result.",
            },
            "patient_report": {
                "english": _lang_block(
                    "No significant blockage was found in your scan.",
                    "Your coronary artery appears clear based on this AI analysis.",
                    ["Maintain a heart-healthy diet low in saturated fat", "Continue regular physical activity", "Keep routine cardiology check-ups"],
                ),
                "hindi": _lang_block(
                    "आपकी जांच में कोई महत्वपूर्ण रुकावट नहीं पाई गई।",
                    "इस AI विश्लेषण के अनुसार आपकी हृदय धमनी साफ दिखाई देती है।",
                    ["कम वसा वाला हृदय-स्वस्थ आहार लें", "नियमित शारीरिक गतिविधि जारी रखें", "नियमित हृदय जांच कराते रहें"],
                ),
                "marathi": _lang_block(
                    "तुमच्या तपासणीत कोणताही मोठा अडथळा आढळला नाही.",
                    "या AI विश्लेषणानुसार तुमची हृदयधमनी स्वच्छ दिसत आहे.",
                    ["कमी स्निग्धांश असलेला हृदय-निरोगी आहार घ्या", "नियमित शारीरिक हालचाल सुरू ठेवा", "नियमित हृदय तपासणी करत रहा"],
                ),
            },
            "disclaimer": {"english": disclaimer_en, "hindi": disclaimer_hi, "marathi": disclaimer_mr},
        }

    worst = max(lesions, key=lambda l: l["stenosis_percent"])
    pct = worst["stenosis_percent"]
    band = worst["severity_band"]
    confidence = worst.get("confidence")
    conf_clause_en = f" (model confidence: {confidence}%)" if confidence is not None else ""
    conf_clause_hi = f" (मॉडल विश्वास स्तर: {confidence}%)" if confidence is not None else ""
    conf_clause_mr = f" (मॉडेल विश्वास पातळी: {confidence}%)" if confidence is not None else ""

    return {
        "doctor_report": {
            "clinical_summary": f"{len(lesions)} stenotic lesion(s) identified on the analyzed frame. The most severe lesion demonstrates approximately {pct}% luminal narrowing, classified as {band} stenosis{conf_clause_en}.",
            "findings": f"Lesion count: {len(lesions)}. Peak stenosis: {pct}% ({band}), model confidence {confidence}%. Bounding box coordinates and per-lesion severity are provided in the structured output for review.",
            "recommendation": "Correlate with clinical presentation. Further diagnostic angiography or intervention planning may be warranted depending on the severity band and patient symptomatology.",
        },
        "patient_report": {
            "english": _lang_block(
                f"Your scan shows narrowing in {len(lesions)} area(s) of your heart's artery, with the most significant being about {pct}% narrowed ({band} severity){conf_clause_en}.",
                "This means part of your artery is narrower than normal, which can reduce blood flow to your heart muscle. The severity level helps your doctor decide next steps.",
                ["Consult a cardiologist promptly to confirm and discuss treatment options", "Reduce salt, fried food, and saturated fat intake", "Avoid smoking and limit alcohol", "Monitor blood pressure and cholesterol regularly", "Engage in doctor-approved light exercise like walking"],
            ),
            "hindi": _lang_block(
                f"आपकी जांच में हृदय की धमनी में {len(lesions)} जगह पर रुकावट पाई गई है, सबसे गंभीर रुकावट लगभग {pct}% है ({band} स्तर की){conf_clause_hi}।",
                "इसका मतलब है कि आपकी धमनी का एक हिस्सा सामान्य से संकरा है, जिससे हृदय तक रक्त प्रवाह कम हो सकता है। गंभीरता का स्तर आपके डॉक्टर को आगे का इलाज तय करने में मदद करता है।",
                ["तुरंत हृदय रोग विशेषज्ञ से सलाह लें", "नमक, तला-भुना और वसायुक्त भोजन कम करें", "धूम्रपान से बचें और शराब सीमित करें", "रक्तचाप और कोलेस्ट्रॉल की नियमित जांच कराएं", "डॉक्टर की सलाह अनुसार हल्का व्यायाम जैसे टहलना करें"],
            ),
            "marathi": _lang_block(
                f"तुमच्या तपासणीत हृदयाच्या धमनीत {len(lesions)} ठिकाणी अडथळा आढळला आहे, सर्वात गंभीर अडथळा सुमारे {pct}% आहे ({band} पातळीचा){conf_clause_mr}.",
                "याचा अर्थ तुमच्या धमनीचा एक भाग सामान्यपेक्षा अरुंद झाला आहे, ज्यामुळे हृदयाकडे रक्तप्रवाह कमी होऊ शकतो. तीव्रतेची पातळी तुमच्या डॉक्टरांना पुढील उपचार ठरवण्यास मदत करते.",
                ["लवकरात लवकर हृदयरोगतज्ञांचा सल्ला घ्या", "मीठ, तळलेले आणि स्निग्ध पदार्थ कमी करा", "धूम्रपान टाळा आणि मद्यपान मर्यादित करा", "रक्तदाब आणि कोलेस्टेरॉल नियमितपणे तपासा", "डॉक्टरांच्या सल्ल्यानुसार हलका व्यायाम जसे चालणे करा"],
            ),
        },
        "disclaimer": {"english": disclaimer_en, "hindi": disclaimer_hi, "marathi": disclaimer_mr},
    }

def generate_report(analysis: dict) -> dict:
    model = _get_model()
    if model is None:
        return _fallback_report(analysis)

    prompt = f"""You are a medical AI assistant generating a coronary angiogram screening report for two audiences. Write in a formal, professional clinical tone suitable for inclusion in a patient's medical record and for a non-medical reader respectively.

Structured model output:
{json.dumps(analysis, indent=2)}

Return ONLY a JSON object with exactly this structure (no extra text, no markdown fences):
{{
  "doctor_report": {{
    "clinical_summary": "1-2 sentences using precise clinical terminology (stenosis %, severity band, model confidence %, luminal narrowing)",
    "findings": "specific findings: lesion count, peak stenosis %, model confidence %, relevant coordinates, written formally as a radiology-style findings statement",
    "recommendation": "formal clinical next-step recommendation"
  }},
  "patient_report": {{
    "english": {{"summary": "2-3 simple sentences, no jargon, may mention the model's confidence in plain terms (e.g. 'the model is 85% confident in this finding')", "what_it_means": "1-2 sentences explaining the health implication in everyday language", "lifestyle_suggestions": ["4-6 short, concrete, actionable health/lifestyle tips relevant to the severity found"]}},
    "hindi": {{ same structure, fully translated and culturally natural in Hindi, not literal word-for-word translation }},
    "marathi": {{ same structure, fully translated and culturally natural in Marathi, not literal word-for-word translation }}
  }},
  "disclaimer": {{
    "english": "one sentence: AI screening aid, not diagnosis, consult a cardiologist",
    "hindi": "same disclaimer in Hindi",
    "marathi": "same disclaimer in Marathi"
  }}
}}

Base all severity and confidence language strictly on the provided data (each lesion object includes a "confidence" percentage — use the value for the most severe lesion when describing overall confidence). Do not invent findings not present in the structured output. If no lesions are present, say so honestly and give general heart-health maintenance suggestions instead of narrowing-specific ones, and omit confidence language entirely in that case."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception:
        return _fallback_report(analysis)