from typing import List
from langchain_community.document_loaders import UnstructuredFileLoader
import tqdm
import os, sys, io
from pathlib import Path
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# from rapidocr_paddle import  # GPU推理
from rapidocr_onnxruntime import RapidOCR  # COU推理引擎

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.3, 0.3)


def get_ocr():
    # ocr = RapidOCR(det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=True)
    ocr = RapidOCR()
    return ocr


class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def pdf2text(file_path):
            import pymupdf
            import numpy as np

            docs = pymupdf.open(file_path)
            # print(doc.metadata)
            b_unit = tqdm.tqdm(
                total=docs.page_count, desc="RapidOCRPDFLoader context page index: 0"
            )
            resp = ""
            for i, page in enumerate(docs):
                b_unit.set_description(
                    "RapidOCRPDFLoader context page index: {}".format(i)
                )
                b_unit.refresh()
                blocks = page.get_text(
                    option="blocks",
                    flags=4,
                    sort=True,
                )
                images = page.get_image_info(xrefs=True)
                image_texts = {}
                for img in images:
                    if xref := img.get("xref"):
                        block_num, bbox = img["number"], img["bbox"]
                        # 检查图片尺寸是否超过设定的阈值
                        if (bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[
                            0
                        ] or (bbox[3] - bbox[1]) / (
                            page.rect.height
                        ) < PDF_OCR_THRESHOLD[
                            1
                        ]:
                            continue
                        pix = pymupdf.Pixmap(docs, xref)
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                            pix.height, pix.width, -1
                        )
                        ocr = get_ocr()
                        result, _ = ocr(img_array)
                        # 获取当前图像的内容
                        img_resp = ""
                        if result:
                            ocr_result = [line[1] for line in result]
                            img_resp += " ".join(ocr_result)
                            image_texts[block_num] = img_resp
                # 更新进度
                b_unit.update(1)
                # 完整的页面内容按类似人类阅读的顺序输出
                for block in blocks:
                    if block[-1] == 1:
                        # print("图片内容", image_texts.get(block[-2], ""))
                        resp += image_texts.get(block[-2], "") + "\n"
                    else:
                        # print(block[-3])
                        resp += block[-3] + "\n"
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text

        return partition_text(text=text, **self.unstructured_kwargs)


class PDFTextLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def pdf2text(file_path):
            import pymupdf
            from text_splitter import re_construct_docs

            docs = pymupdf.open(file_path)
            # print(doc.metadata)
            b_unit = tqdm.tqdm(
                total=docs.page_count, desc="PDFTextLoader context page index: 0"
            )
            resp = ""
            for i, page in enumerate(docs):
                b_unit.set_description("PDFTextLoader context page index: {}".format(i))
                b_unit.refresh()
                # 只获取PDF中的文本内容
                page_content = page.get_text(
                    option="text",
                    sort=True,
                )
                # 更新进度
                b_unit.update(1)

                resp += page_content.replace("\n\n", "\n")
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text

        return partition_text(text=text, **self.unstructured_kwargs)


def extract_title(content: str) -> list[str]:
    import re

    res_titles = []
    for text in content:
        if len(text) > 30 or len(text) == 0:
            continue
        if text.endswith((",", ".", "，", "。")):
            continue
        if text.isnumeric():
            continue
        if text.isnumeric():
            continue

        # 若以标点符号结尾，说明不是标题
        ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
        ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
        if ENDS_IN_PUNCT_RE.search(text) is not None:
            continue

        START_IN_NUM_PATTERN = r"^(?:[一二三四五六七八九十]|\d){1}\s*[、.]\s*.+"
        START_IN_NUM_RE = re.compile(START_IN_NUM_PATTERN)
        if START_IN_NUM_RE.search(text) is None:
            continue

        res_titles.append(text)
    return res_titles


def optimize_content_retrieval(file_path):
    import pymupdf, json

    docs = pymupdf.open(file_path)
    tittle2text_all = []
    cur_title = ""
    # 如果刚好整个pdf都没有数字标题，不好处理
    for i, page in enumerate(docs):
        # 只获取PDF中的文本内容
        page_content = page.get_text(
            option="text",
            sort=True,
        )
        page_content = page_content.replace("\n\n", "\n").split("\n")
        titles = extract_title(page_content)
        # print(titles)
        # 需要解决内容跨页的问题
        if len(titles) == 0:
            # 跨页内容处理
            if cur_title is not None and tittle2text_all and tittle2text_all[-1]:
                last_content = tittle2text_all[-1]
                if cur_title in last_content:
                    last_content[cur_title] += " " + "".join(
                        page_content
                    )  # 使用空格连接文本
                else:
                    # 如果当前标题不在 last_content 中，初始化它
                    last_content[cur_title] = "".join(page_content)
            continue  # 如果没有标题，跳过剩余的页面内容处理

        title_idx = 0
        is_title_before = False
        title = None
        title2text = {}
        _text_ = ""
        cur_idx = 0
        # 处理跨页的内容，拼接到之前的标题后
        for text_line in page_content:
            if text_line.strip() not in titles and len(tittle2text_all) > 0:
                last_content = tittle2text_all[-1]
                if cur_title in last_content:
                    last_content[cur_title] += " " + text_line
                cur_idx += 1
                continue
            else:
                break

        for text_line in page_content[cur_idx:]:
            if title_idx < len(titles) and text_line.strip() == titles[title_idx]:
                # 识别为标题的内容，进行分级处理
                # 处理第一个title,注意可能有跨页的内容，要关联上一页的标题
                if not title:
                    title = text_line.strip()
                    _text_ = ""
                elif not is_title_before:
                    # 处理新的标题
                    is_title_before = True
                    title2text[title] = _text_.replace(" ", "")
                    # 切换新标题
                    title = text_line.strip()
                    # print("标题", title)
                    _text_ = ""
                else:
                    # 处理子标题内容，直接视为上一标题相关的内容
                    # print(f"下文关联标题：{title}", text_line)
                    _text_ += text_line
                title_idx += 1
            else:
                # 出现非识别为标题的内容，那关联当前标题
                is_title_before = False
                _text_ += text_line
                # print(f"{text_line}")
        # print(title, _text_)
        if _text_ != "" and title:
            title2text[title] = _text_.replace(" ", "")
            cur_title = title

        tittle2text_all.append(title2text)
        # print(title2text)

    with open("pdf_load.json", "w", encoding="utf-8") as json_file:
        json_data = json.dumps(tittle2text_all, ensure_ascii=False, indent=4)
        json_file.write(json_data)


# 待实现功能，就是将提取出来的内容，按照可能的小节信息进行分块；并存储到小节字典中，
if __name__ == "__main__":
    work_dir = Path(os.getcwd())
    data_dir = work_dir / "datas" / "pdfs"
    pdf_file = data_dir / "llm_qa.pdf"
    # metadata获取pdf的一些主要属性
    optimize_content_retrieval(pdf_file)
    # titles = re_construct_docs(page_content)
    # for title in titles:
    #     print(title)
