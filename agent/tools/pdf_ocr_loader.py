from typing import List
from langchain_community.document_loaders import UnstructuredFileLoader
import tqdm
import os, sys, io
from pathlib import Path

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
                            img_resp += "\n".join(ocr_result)
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


# 待实现功能，就是将提取出来的内容，按照可能的小节信息进行分块；并存储到小节字典中，
if __name__ == "__main__":
    work_dir = Path(os.getcwd())
    data_dir = work_dir / "datas" / "pdfs"
    pdf_file = data_dir / "llm_qa.pdf"
    loader = RapidOCRPDFLoader(file_path=pdf_file)
    docs = loader.load()
    # print(docs)
    for doc in docs:
        print(doc.page_content)
