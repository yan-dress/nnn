from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pandas as pd
from tqdm import tqdm
import time
import uuid
import shutil
import os
import torch

# 优化：设置 torch 线程数为 CPU 核心数
torch.set_num_threads(os.cpu_count())

class ProgressVectorDB:
    def __init__(self):
        self.model_name = 'all-mpnet-base-v2'
        
        # GPU 检测与配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        
        if self.device == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            torch.cuda.empty_cache()  # 清空 GPU 缓存
        else:
            print(f"CPU cores: {os.cpu_count()}")
        
        # 批处理大小根据设备调整
        self.batch_size = 256 if self.device == 'cuda' else 64  # GPU 可以用更大的 batch
    
    def build_vector_db(self):
        print("=" * 60)
        print("Building Vector Database")
        print("=" * 60)
        
        # Step 0: 彻底清除旧数据库（解决缓存问题）
        db_path = "vector_db"
        if os.path.exists(db_path):
            print(f"Removing old database at {db_path}")
            shutil.rmtree(db_path)
            print("✓ Old database removed")
        
        # Step 1: 加载模型到指定设备（GPU 或 CPU）
        print(f"Loading model '{self.model_name}' on {self.device}")
        model = SentenceTransformer(self.model_name, device=self.device)
        print("✓ Model loaded")
        
        # Step 2: 初始化数据库（不使用 embedding_function，我们手动控制）
        print("Initializing vector database")
        client = chromadb.PersistentClient(path=db_path)
        
        # 创建新集合（不配置 embedding_function，我们会手动传向量）
        collection = client.create_collection(
            name="docs",
            metadata={"model": self.model_name, "device": self.device}
        )

        # Step 3: 加载数据
        print("Loading data file")
        df = pd.read_csv("segments.txt", sep='\t')
        print(f"Total {len(df)} rows")
        
        # Step 4: 处理文档
        print("Processing document content")
        documents = []
        metadatas = []
        ids = []
        
        valid_count = 0
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
            if pd.notna(row['Content']) and len(str(row['Content'])) > 20:
                content = str(row['Content']).strip()
                documents.append(content)
                metadatas.append({
                    'source': row.get('Book_Name', 'unknown'),
                    'chapter': row.get('Chapter_Title', 'unknown')
                })
                # 使用 UUID 避免重复
                ids.append(f"doc_{uuid.uuid4().hex[:16]}")
                valid_count += 1
        
        print(f"Valid documents: {valid_count}/{len(df)}")
        
        # Step 5: 分批计算向量并添加到数据库
        print(f"Encoding documents with batch_size={self.batch_size} on {self.device}")
        
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(0, len(documents), self.batch_size), 
                             total=total_batches,
                             desc="Adding batches"):
            batch_end = min(batch_idx + self.batch_size, len(documents))
            
            batch_docs = documents[batch_idx:batch_end]
            batch_metas = metadatas[batch_idx:batch_end]
            batch_ids = ids[batch_idx:batch_end]
            
            # 在 GPU 上编码（convert_to_numpy=True 避免后续转换开销）
            embeddings = model.encode(
                batch_docs,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # 添加到集合，直接传 numpy 数组
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
                embeddings=embeddings
            )
        
        print(f"✓ Added {len(documents)} documents")
        
        # 清理 GPU 内存
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return collection

def main():
    """Main function"""
    start_time = time.time()
    
    # Create progress tracker
    progress_db = ProgressVectorDB()
    
    # Build database
    collection = progress_db.build_vector_db()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("Vector Database Build Complete!")
    print(f"Total time: {total_time:.2f} seconds")
       
    # Get document count
    doc_count = collection.count()
    print(f"\nFinal Statistics:")
    print(f"   - Total documents: {doc_count}")

if __name__ == "__main__":
    main()