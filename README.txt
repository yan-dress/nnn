vector_db - 向量数据库完整文件夹

文档数量：12,124 个文本块
向量数量：12,124 个嵌入向量
嵌入模型：all-mpnet-base-v2 (768 维度)
元数据字段：source(书籍标题), chapter(章节标题)
数据库类型：ChromaDB (开源向量数据库)
环境要求：pip install chromadb sentence-transformers pandas tqdm

在 "vector_db" 文件夹中：
chroma.sqlite3 - 存储文档内容、元数据和ID
.bin 文件 - 存储向量索引数据
这些文件共同表示 12,124 个文档 + 12,124 个向量

完整的 "vector_db" 文件夹包含：
- chroma.sqlite3 - 主数据库文件（存储文档、元数据、ID）
- data_level0.bin - 实际向量数据（嵌入向量）
- header.bin - 索引元数据和配置
- index_metadata.pickle - 附加索引元数据
- length.bin - 向量长度信息（用于归一化）
- link_lists.bin - 图索引连接信息

连接代码示例
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection(name="docs")
附简单test文件