import os
import json
import sys
import fitz
from google import genai

def _configure_tesseract():
    import pytesseract

    env_path = os.getenv("TESSERACT_CMD")
    if env_path:
        pytesseract.pytesseract.tesseract_cmd = env_path
        return

    if sys.platform.startswith("win"):
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win_path):
            pytesseract.pytesseract.tesseract_cmd = default_win_path

def pick_file():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost",True)

    file_path = filedialog.askopenfilename(
        title="select your notes file",
        filetypes=[
            ("supported files","*.pdf *.docx *.pptx *.txt *.jpg *.jpeg *.png"),
            ("All files","*.*")
        ]
    )
    root.destroy()
    return file_path

def detect_file_type(file_path: str):
    _, extension = os.path.splitext(file_path)
    return extension.lower()

def extract_text_from_pdf(file_path:str):

    import pytesseract
    from PIL import Image
    import io

    _configure_tesseract()

    full_text = ""
    doc = fitz.open(file_path)
    print(f"PDF Opened.\n Total pages: {len(doc)}")

    for page_number,page in enumerate(doc,start=1):
        page_text = page.get_text()

        if page_text.strip():
            full_text += page_text +"\n"
        else:
            print(f"There is No Extractable text in page no: {page_number} - trying OCR...")
            pix = page.get_pixmap(dpi=400)
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(page_image,lang="eng+tam")
            if ocr_text.strip():
                full_text += ocr_text + "\n"
            else:
                print(f"OCR also found no text on page {page_number}")
    doc.close()
    return full_text

def extract_text_from_docx(file_path:str):
    from docx import Document

    doc = Document(file_path)
    full_text = ""
    for paragraph in doc.paragraphs:
        full_text += paragraph.text + "\n"
    return full_text

def extract_text_from_pptx(file_path:str):
    from pptx import Presentation

    prs = Presentation(file_path)
    full_text = ""
    for slide_number, slide in enumerate(prs.slides,start=1):
        full_text += f"\n--- slide{slide_number} --\n"
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        full_text += run.text + " "
                full_text += "\n"
    return full_text

def extract_text_from_txt(file_path:str):

    with open(file_path,"r", encoding="utf-8") as f:
        return f.read()
    
def extract_text_from_image(file_path: str):

    import pytesseract
    from PIL import Image

    _configure_tesseract()

    image = Image.open(file_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text
    
def extract_text(file_path:str):
    file_type = detect_file_type(file_path)
    print(f"Detected file type: {file_type}")

    if file_type == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == ".docx":
        return extract_text_from_docx(file_path)
    elif file_type == ".pptx":
        return extract_text_from_pptx(file_path)
    elif file_type == ".txt":
        return extract_text_from_txt(file_path)
    elif file_type in (".jpg",".jpeg",".png"):
        return extract_text_from_image(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{file_type}'."
                                                                          
        )

def ai_prompt(notes_text:str):

    prompt = f"""You are a veteran question-setter for Indian government exams: TNPSC (Group 1/2/4),
UPSC (Prelims/Mains), and TNTET.

Read the study notes below and generate exam-style questions in THREE difficulty
bands.Match the real exam patterns exactly:

- EASY  -> Direct factual recall. One correct fact, no reasoning needed.
           (Matches TNPSC Group 4 / TNTET difficulty.)
- MODERATE -> Requires connecting two related facts from the notes, or a
           "which of the following is/is not correct" style statement question.
           (Matches TNPSC Group 2 difficulty.)
- HARD  -> Requires analysis, comparison, or applying a concept to a new
           situation. Multi-statement or assertion-reason format.
           (Matches UPSC Prelims/Mains difficulty.)

Rules:
1. Generate exactly 30 questions per difficulty band.
2. Use multiple-choice format (4 options: A, B, C, D) wherever the real exam
   would use MCQs. Mark the correct option clearly.
3. Base every question ONLY on the notes provided below. Do not invent facts.
4. Write in the same dry, formal tone real question papers use - no casual language.

STUDY NOTES:
{notes_text}

Respond with ONLY valid JSON, no extra commentary, no markdown fences, in exactly
this structure:

{{
  "easy": [
    {{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A"}}
  ],
  "moderate": [ ... same structure ... ],
  "hard": [ ... same structure ... ]
}}
"""
    return prompt

def generate_questions(notes_text:str,api_key:str):
    client=genai.Client(api_key=api_key)
    
    prompt = ai_prompt(notes_text)

    response=client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    raw_reply = response.text
    return raw_reply

def parse_and_display(raw_reply:str):
    clean = raw_reply.strip()
    if clean.startswith("```"):
        clean=clean.strip("`")
        clean=clean.replace("json","",1).strip()

    data=json.loads(clean)

    for level in ["easy","moderate","hard"]:
        print(f"\n{'='*15}{level.upper()}{'='*15}")
        for i, qa in enumerate(data[level], start=1):
            print(f"\nQ{i}.{qa['question']}")
            for opt_key, opt_text in qa["options"].items():
                print(f"   {opt_key}. {opt_text}")
            print(f"  correct Answer: {qa['answer']}")
    return data 

def get_next_filename(base_name:str = "generated_questions",extension: str = "txt"):

    counter = 1
    while True:
        filename = f"{base_name}{counter}.{extension}"
        if not os.path.exists(filename):
            return filename
        counter += 1

def save_to_file(data:dict,filename:str):

    with open(filename,"w",encoding="utf-8") as f:
        for level in ["easy","moderate", "hard"]:
            f.write(f"\n{('='*15)} {level.upper()} {('='*15)}\n")
            for i,qa in enumerate(data[level],start=1):
                f.write(f"\nQ{i}.{qa['question']}\n")
                for opt_key, opt_text in qa["options"].items():
                    f.write(f"  {opt_key}. {opt_text}\n")
                f.write(f"   correct Answer: {qa['answer']}\n")

    print(f"\nThe Questions has been saved successfully in: {filename}")

def main():
    print("Opening file picker window...")
    file_path= pick_file()
    
    if not file_path:
        print("No File selected")
        return
    
    print(f"Selected file: {file_path}")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set your ANTHROPIC_API_KEY Environment first")
        return
    
    print("\nReading the File...")
    try:
        notes_text = extract_text(file_path)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if not notes_text.strip():
        print("No text can be Extracted from this File")
        return    
    
    if len(notes_text) > 25000:
        print("File is too large only first 25000 character are taken")
        notes_text = notes_text[:25000]

    print("Sending the File to AI")
    raw_reply = generate_questions(notes_text,api_key)

    print("Output is being customized and will be displayed soon ")
    data=parse_and_display(raw_reply)
    
    print("The File is being saved...")
    output = get_next_filename()
    save_to_file(data,output)

if __name__=="__main__":
    main()