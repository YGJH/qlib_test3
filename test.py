import markdown        # pip install markdown
import json

# 範例：先讀 JSON，再轉 Markdown -> HTML
with open('log.json', encoding='utf8') as file:
    data = file.read()
    md_text = "# 主標題\n\n" + "\n\n".join(f"## {item['title']}\n{item['content']}" for item in data)
    html_body = markdown.markdown(md_text)

    html = f"""<!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>自動生成</title></head>
    <body>{html_body}</body></html>"""
    with open('out.html','w',encoding='utf8') as f:
        f.write(html)