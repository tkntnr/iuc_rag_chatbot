import fitz  # PyMuPDF kütüphanesi
import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_and_clean_pdf(pdf_path):
    print(f"[{pdf_path}] okunuyor...")
    
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # VERİ TEMİZLEME
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        full_text += text + " "
        
    print(f"Toplam {len(doc)} sayfa başarıyla okundu ve temizlendi.")
    return full_text.strip()

def split_text_into_chunks(text, chunk_size=512, chunk_overlap=50):
    print(f"\nMetin anlamsal parçalara (chunk) bölünüyor... (Boyut: {chunk_size}, Örtüşme: {chunk_overlap})")
    
    # LangChain'in akıllı bölücüsü
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    print(f"Toplam {len(chunks)} adet parça (chunk) oluşturuldu.")
    return chunks

if __name__ == "__main__":
    pdf_yolu = os.path.join("data", "yonetmelik.pdf") 
    
    if os.path.exists(pdf_yolu):
        # 1. Okuma ve Temizleme
        temiz_metin = extract_and_clean_pdf(pdf_yolu)
        
        # 2. Anlamsal Parçalama
        metin_parcalari = split_text_into_chunks(temiz_metin)
        
        # Kontrol için ilk 2 parçayı ekrana yazdıralım
        print("\n--- 1. Parça (Chunk) ---")
        print(metin_parcalari[0])
        print("\n--- 2. Parça (Chunk) ---")
        print(metin_parcalari[1])
    else:
        print(f"HATA: '{pdf_yolu}' bulunamadı!")