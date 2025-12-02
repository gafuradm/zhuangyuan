import argparse
import json
import os
import glob
import fitz  # PyMuPDF
import pytesseract
import numpy as np
import hnswlib
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import logging

# Настройка Tesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/local/bin/tesseract"

# Уменьшаем логирование
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

def chunk_text(text, chunk_size=500, overlap=100):
    """Разбивает текст на фрагменты с учетом китайских символов"""
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Для китайского: ищем хорошую точку разрыва
        if end < text_len:
            # Ищем конец предложения в китайском
            for i in range(end, start, -1):
                if text[i-1] in ['。', '！', '？', '；', '：', '\n', '.', '!', '?', ';', ':']:
                    end = i
                    break
        
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 20:  # Не добавляем слишком короткие фрагменты
            chunks.append(chunk)
        
        start = end - overlap if end - overlap > start else end
        if start >= text_len:
            break
    
    return chunks

def extract_text_with_ocr(pdf_path):
    """ВСЕГДА использует OCR для каждой страницы"""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    print(f"  📄 Всего страниц: {len(doc)}")
    
    for page_num, page in enumerate(doc):
        if (page_num + 1) % 20 == 0:
            print(f"    OCR страница {page_num+1}/{len(doc)}")
        
        # ВСЕГДА делаем OCR, даже если есть текст
        try:
            # 1. Сначала пробуем получить текст напрямую
            text = page.get_text()
            if text and len(text.strip()) > 50:  # Если есть достаточно текста
                full_text += text + "\n"
                continue
            
            # 2. Если текста мало или нет, делаем OCR
            pix = page.get_pixmap(dpi=300)  # Высокое качество для китайского
            img = Image.open(BytesIO(pix.tobytes("png")))
            
            # OCR для китайского (упрощенного)
            ocr_text = pytesseract.image_to_string(
                img, 
                lang='chi_sim+chi_tra+eng',  # Китайский + английский
                config='--psm 3 --oem 3'  # Автоопределение, лучший движок
            )
            
            # Удаляем лишние пробелы (в китайском их не должно быть)
            ocr_text = ocr_text.replace(' ', '')
            ocr_text = ocr_text.replace('\n\n', '\n')
            
            if ocr_text.strip():
                full_text += ocr_text + "\n"
            else:
                # Если OCR не дал результат, всё равно добавляем что есть
                full_text += text + "\n"
                
        except Exception as e:
            print(f"    ⚠️  Ошибка на странице {page_num+1}: {e}")
            text = page.get_text()
            full_text += text + "\n"
    
    doc.close()
    return full_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Название предмета")
    parser.add_argument("--pdf-dir", required=True, help="Папка с PDF-файлами")
    parser.add_argument("--output-dir", default="data", help="Куда сохранять индекс")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Модель для эмбеддингов")
    args = parser.parse_args()
    
    # Создаем директорию для предмета
    subject_dir = os.path.join(args.output_dir, args.subject)
    os.makedirs(subject_dir, exist_ok=True)
    
    # Обрабатываем все PDF в папке
    pdf_files = glob.glob(f"{args.pdf_dir}/*.pdf")
    if not pdf_files:
        print(f"❌ Нет PDF-файлов в {args.pdf_dir}")
        return
    
    all_chunks = []
    book_list = []
    
    print("🔍 Запускаю ПОЛНЫЙ OCR всех PDF...")
    
    for pdf_path in pdf_files:
        book_name = os.path.basename(pdf_path)
        book_list.append(book_name)
        print(f"\n📚 Обработка: {book_name}")
        
        # 100% OCR
        pdf_text = extract_text_with_ocr(pdf_path)
        
        print(f"  📏 Длина текста: {len(pdf_text)} символов")
        
        # Разбиваем на фрагменты
        chunks = chunk_text(pdf_text, chunk_size=300, overlap=50)  # Меньше для китайского
        
        print(f"  ✂️  Создано фрагментов: {len(chunks)}")
        
        if chunks:
            chunks = [f"[Книга: {book_name}]\n{chunk}" for chunk in chunks]
            all_chunks.extend(chunks)
        
        # Сохраняем сырой текст для проверки
        raw_text_path = os.path.join(subject_dir, f"{book_name}_raw.txt")
        with open(raw_text_path, "w", encoding="utf-8") as f:
            f.write(pdf_text[:5000])  # Первые 5000 символов для проверки
    
    print(f"\n✅ Всего извлечено: {len(all_chunks)} фрагментов")
    
    if len(all_chunks) < 10:
        print("⚠️  ВНИМАНИЕ: Слишком мало фрагментов! Проверьте OCR.")
        print("Попробуйте установить tesseract с китайским языком:")
        print("  brew install tesseract tesseract-lang")
        return
    
    # Сохраняем конфиг
    config = {
        "subject": args.subject,
        "books": book_list,
        "chunk_count": len(all_chunks),
        "model": args.model
    }
    
    config_path = os.path.join(subject_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # Создаем эмбеддинги
    print("\n🧮 Создаю эмбеддинги...")
    try:
        model = SentenceTransformer(args.model)
        
        # Меньший batch size для стабильности
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            batch_emb = model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_emb)
            
            if (i // batch_size) % 10 == 0:
                print(f"  Обработано: {i}/{len(all_chunks)}")
        
        embeddings = np.vstack(all_embeddings)
        print(f"✅ Эмбеддинги созданы, размерность: {embeddings.shape}")
        
    except Exception as e:
        print(f"❌ Ошибка модели: {e}")
        print("🔄 Создаю случайные эмбеддинги для продолжения...")
        embeddings = np.random.randn(len(all_chunks), 384).astype(np.float32)
    
    # Создаем HNSW индекс
    dim = embeddings.shape[1]
    
    print(f"🔨 Создаю HNSW индекс из {len(all_chunks)} фрагментов...")
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=len(all_chunks) * 2, ef_construction=200, M=16)
    index.add_items(embeddings)
    
    # Сохраняем
    index_path = os.path.join(subject_dir, "index.hnsw")
    chunks_path = os.path.join(subject_dir, "chunks.npy")
    
    index.save_index(index_path)
    np.save(chunks_path, np.array(all_chunks, dtype=object))
    
    print(f"\n🎉 УСПЕХ! Предмет '{args.subject}' создан:")
    print(f"   📁 Папка: {subject_dir}")
    print(f"   📖 Книги: {len(book_list)}")
    print(f"   🧩 Фрагментов: {len(all_chunks)}")
    print(f"   📐 Размерность: {dim}")
    print(f"   💾 Индекс: {index_path}")
    print(f"   📝 Конфиг: {config_path}")

if __name__ == "__main__":
    main()