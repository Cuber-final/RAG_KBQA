import ebooklib
from ebooklib import epub
import os
from pathlib import Path
import html
import re
from bs4 import BeautifulSoup
import sys
import io
import json

# 包装sys.stdout以确保使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class Book:
    def __init__(self, book_path):
        self.book = epub.read_epub(book_path)
        self.TITLE = self.book.get_metadata("DC", "title")
        self.AUTHORS = self.book.get_metadata("DC", "creator")
        self.CHAPTERS = {}
        self.chapters_nums = 0
        self.book_id = None

    def rerender_chapter_contents(self):
        book = self.book
        chapters = []
        chapter_id = 1
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content()
            soup = BeautifulSoup(html.unescape(content.decode("utf-8")), "html.parser")
            # 找到所有的标题<h1>、<h2>等
            titles = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            # 遍历所有标题
            for title in titles:
                # print(title.get_text(strip=True))
                _chapter = {}
                _chapter["chapter_id"] = chapter_id
                _chapter["title"] = title.get_text(strip=True)
                _chapter["title_tag"] = title.name
                _chapter["content"] = ""
                chapter_id += 1
                # 从标题的下一个兄弟节点开始，寻找所有<p>标签的直接文本内容
                next_sibling = title.find_next_sibling()
                while next_sibling and not next_sibling.name.startswith("h"):
                    if next_sibling.name == "p":
                        _chapter_contents = []
                        # 获取<p>标签的直接文本内容
                        # 如果是图片呢，又该如何提取？
                        if next_sibling.find("img"):
                            # 提取<img>标签的图片id或src属性
                            imgs = next_sibling.find_all("img")
                            for img in imgs:
                                # 提取<img>标签的src属性
                                image_src = img.get("src")
                                if image_src != "":
                                    # 添加当前章节对应的图片信息
                                    _chapter_contents.append(
                                        "#img_start" + image_src + "#img_end"
                                    )
                        pure_text = next_sibling.get_text(strip=True)
                        _chapter_contents.append(
                            pure_text
                        )  # 添加当前章节中所有p中的正文
                    next_sibling = next_sibling.find_next_sibling()
                # 把章节内容整合拼接
                _chapter["content"] = "".join(_chapter_contents)
                # print(_chapter)
            chapters.append(_chapter)  # 把新读取的章节内容存储到章节列表中

        self.CHAPTERS = chapters
        # print(chapters)
        json_data = json.dumps(chapters, ensure_ascii=False, indent=4)
        with open("chapter_log.json", "w", encoding="utf-8") as json_file:
            json_file.write(json_data)


if __name__ == "__main__":
    data_path = Path(os.getcwd(), "datas", "books")
    # epub_file = data_path / "week4hs.epub"
    epub_file = data_path / "meng.epub"
    # print(str(epub_file))
    book = Book(str(epub_file))
    book.rerender_chapter_contents()
    # book.load_chapter_titles()
    # print(len(book.CHAPTERS))
