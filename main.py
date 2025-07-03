import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Gerekirse kendi WordPress sitenizin domain'ini ekleyin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class KeywordRequest(BaseModel):
    keyword: str

@app.post("/analyze")
async def analyze(req: KeywordRequest):
    keyword = req.keyword
    try:
        # 1) Sık sorulan sorular
        q_prompt = (
            f"'{keyword}' konusu hakkında kullanıcıların en sık sorduğu 10 soruyu "
            "madde madde sıralı liste halinde ver."
        )
        q_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": q_prompt}
            ],
            temperature=0.7
        )
        questions = q_resp.choices[0].message.content.strip()

        # 2) Semantik anahtar kelimeler
        s_prompt = (
            f"'{keyword}' için semantik olarak ilişkili 10 anahtar kelimeyi virgülle ayrılmış "
            "şekilde ver."
        )
        s_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": s_prompt}],
            temperature=0.7
        )
        semantic = s_resp.choices[0].message.content.strip()

        # 3) Entity önerileri
        e_prompt = (
            f"'{keyword}' ile ilgili 10 önemli entity (kişi, mekan, kavram) ve her biri için "
            "kısa bir açıklama, 'Entity: Açıklama' formatında ver."
        )
        e_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": e_prompt}],
            temperature=0.7
        )
        entities = e_resp.choices[0].message.content.strip()

        return {
            "questions": questions,
            "semantic": semantic,
            "entities": entities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
