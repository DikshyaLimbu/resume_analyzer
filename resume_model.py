import os
import pandas as pd
import fitz
import pdfplumber
import warnings
import numpy as np
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from pdfminer.high_level import extract_text
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, InputExample, util, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

class ResumeProcessor:
    def __init__(self, resume_dir: str = 'resumes/'):
        self.resume_dir = resume_dir
        self.resume_data = []

    def extract_resume_text(self, file_path: str) -> str:
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            if text.strip():
                return text
        except Exception as e:
            print(f"[fitz] Failed for {file_path}: {e}")

        try:
            with pdfplumber.open(file_path) as pdf:
                text = "\\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                return text
        except Exception as e:
            print(f"[pdfplumber] Failed for {file_path}: {e}")

        try:
            text = extract_text(file_path)
            if text.strip():
                return text
        except Exception as e:
            print(f"[pdfminer] Failed for {file_path}: {e}")

        return "[ERROR] Could not extract text from file."

    def extract_section(self, text: str, section_keywords: List[str], next_section_keywords: List[str]) -> str:
        pattern = r'(' + '|'.join(section_keywords) + r')\\s*:?[\\s\\S]*?(?=(' + '|'.join(next_section_keywords) + r')\\s*:?)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(0).strip() if match else ""

    def process_resumes(self) -> pd.DataFrame:
        skill_keywords = ['skills', 'technical skills']
        experience_keywords = ['experience', 'work experience', 'employment history']
        education_keywords = ['education', 'academic background', 'educational qualification']
        other_keywords = ['projects', 'certifications', 'languages', 'hobbies']

        for file in os.listdir(self.resume_dir):
            if file.endswith(".pdf"):
                path = os.path.join(self.resume_dir, file)
                text = self.extract_resume_text(path)

                clean_text = text.replace('\\n', ' ').replace('\\r', ' ').lower()

                skills = self.extract_section(clean_text, skill_keywords, experience_keywords + education_keywords + other_keywords)
                experience = self.extract_section(clean_text, experience_keywords, education_keywords + skill_keywords + other_keywords)
                education = self.extract_section(clean_text, education_keywords, skill_keywords + experience_keywords + other_keywords)

                self.resume_data.append({
                    'filename': file,
                    'Resume': text,
                    'Skills': skills,
                    'Experience': experience,
                    'Education': education
                })

        resume_dataframe = pd.DataFrame(self.resume_data)
        return resume_dataframe

    def save_results(self, df: pd.DataFrame, save_json: bool = True, save_excel: bool = True) -> None:
        if save_json:
            df.to_json("parsed_resumes.json", orient="records", indent=4)
        if save_excel:
            df.to_excel("parsed_resumes.xlsx", index=False)

class ResumeModel:
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.encoder = LabelEncoder()

    def prepare_training_data(self, df: pd.DataFrame) -> DataLoader:
        train_examples = []

        skills_embeddings = self.model.encode(df['Skills'].tolist())
        exp_embeddings = self.model.encode(df['Experience'].tolist())

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                skills_sim = util.cos_sim(skills_embeddings[i], skills_embeddings[j]).item()
                exp_sim = util.cos_sim(exp_embeddings[i], exp_embeddings[j]).item()

                if skills_sim > 0.7 and exp_sim > 0.7:
                    train_examples.append(InputExample(
                        texts=[df.iloc[i]['Resume'], df.iloc[j]['Resume']],
                        label=1.0
                    ))
                elif skills_sim < 0.3 and exp_sim < 0.3:
                    train_examples.append(InputExample(
                        texts=[df.iloc[i]['Resume'], df.iloc[j]['Resume']],
                        label=0.0
                    ))

        random.shuffle(train_examples)
        max_examples = min(1000, len(train_examples))
        train_examples = train_examples[:max_examples]

        return DataLoader(train_examples, shuffle=True, batch_size=16)

    def train(self, train_dataloader: DataLoader, epochs: int = 3) -> None:
        if not train_dataloader.dataset:
            raise ValueError("No training examples provided")
            
        try:
            train_loss = losses.ContrastiveLoss(self.model)

            warmup_steps = int(len(train_dataloader) * 0.1)

            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                show_progress_bar=True
            )
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

    def encode_resume(self, text: str) -> np.ndarray:
        try:
            return self.model.encode(text)
        except Exception as e:
            print(f"Encoding failed: {str(e)}")
            raise

    def save_model(self, output_path: str = 'model/fine_tuned_sbert_resume_model') -> None:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.model.save(output_path)
        except Exception as e:
            print(f"Model saving failed: {str(e)}")
            raise

def main():
    processor = ResumeProcessor()
    resume_df = processor.process_resumes()
    processor.save_results(resume_df)
    
    model = ResumeModel()
    train_dataloader = model.prepare_training_data(resume_df)
    model.train(train_dataloader)
    model.save_model()

if __name__ == "__main__":
    main()
