"""Generate a test PDF with extractable Chinese text and an embedded image.

Uses reportlab's built-in STSong-Light CID font so extracted text is proper
Unicode (matplotlib PDFs embed CID-encoded fonts that don't round-trip).

Usage:
    python scripts/generate_test_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "documents"
OUT_PDF = OUT_DIR / "rag_technology_guide.pdf"
OUT_IMG = OUT_DIR / "rag_pipeline_architecture.png"

PAGES = [
    ("RAG 技术指南（一）：检索增强生成概述",
     "检索增强生成（Retrieval-Augmented Generation，RAG）是一种将信息检索系统与"
     "大语言模型生成能力相结合的技术范式。它的核心思想是：在生成回答之前，先从"
     "外部知识库中检索相关文档片段（chunk），再将检索结果作为上下文注入提示词，"
     "使模型能够基于最新、最准确的知识生成答案，从而缓解大模型幻觉问题。"
     "一个完整的 RAG 链路通常包含离线索引和在线检索两条流水线。离线索引负责将"
     "PDF 等原始文档解析为 Markdown、切分为语义块、通过 Embedding 模型向量化后"
     "写入向量数据库。在线检索则对用户查询进行向量化，在向量库中执行相似度搜索，"
     "召回最相关的候选文档，最后交给 LLM 生成带引用来源的回答。"),
    ("RAG 技术指南（二）：混合检索与重排序",
     "混合检索（Hybrid Search）是 RAG 系统提升召回质量的关键技术。它同时使用"
     "稠密检索（Dense Retrieval）和稀疏检索（Sparse Retrieval）两条通道。"
     "Dense 通道使用 Embedding 模型将文本映射到高维向量空间，通过余弦相似度"
     "捕捉语义层面的相似性，擅长处理同义词和改写表达。Sparse 通道基于 BM25 算法"
     "统计词频与逆文档频率，对专有名词、型号、编号等精确匹配场景表现优异。"
     "两条通道的召回结果通过 RRF（Reciprocal Rank Fusion）算法融合，按倒序排名"
     "加权合并，得到统一的候选列表。融合后的结果还可以送入 Cross-Encoder 或"
     "LLM 进行重排序（Rerank），实现精排，最终平衡查全率与查准率。"),
    ("RAG 技术指南（三）：MCP 协议与 DeepSeek 多模态",
     "MCP（Model Context Protocol）是一种开放协议，标准化了 AI 应用与外部工具、"
     "数据源之间的连接方式。通过 MCP Server，RAG 系统的检索能力可以以标准工具"
     "的形式暴露给 Claude、GitHub Copilot 等客户端，实现一次开发、处处可用。"
     "DeepSeek 于 2026 年 4 月发布 V4-Flash 模型，具备 1M Token 上下文窗口，"
     "支持文本生成与推理。2026 年 8 月，DeepSeek 进一步发布实验性多模态模型"
     "deepseek-v4-flash-vision-exp，正式开启视觉理解能力，支持 JPEG、PNG、GIF、"
     "WebP 格式的图片输入，图片可通过 Base64 内联或 URL 方式传入。"
     "下方图片展示了本项目的 RAG Pipeline 总体架构。"),
]


def main() -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    title_style = ParagraphStyle("TitleCn", fontName="STSong-Light",
                                 fontSize=16, leading=22, spaceAfter=12)
    body_style = ParagraphStyle("BodyCn", fontName="STSong-Light",
                                fontSize=11, leading=18, spaceAfter=8)

    OUT_DIR.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4,
                            leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.5 * cm, bottomMargin=2.5 * cm)

    story = []
    for i, (title, body) in enumerate(PAGES, 1):
        if i > 1:
            story.append(PageBreak())
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(body, body_style))
        if i == 3 and OUT_IMG.exists():
            img = Image(str(OUT_IMG), width=14 * cm, height=8.4 * cm)
            story.append(Spacer(1, 0.5 * cm))
            story.append(img)

    doc.build(story)
    print(f"[OK] Test PDF saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
