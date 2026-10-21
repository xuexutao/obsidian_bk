import os
import json
import re
from datetime import datetime

VAULT_ROOT = "/Users/eilonxtxue/Documents/obsidian/obsidian_bk"
OUTPUT_MANIFEST = "/Users/eilonxtxue/With/20261020/3qaq/vault-manifest.js"
OUTPUT_DOCS = "/Users/eilonxtxue/With/20261020/3qaq/vault-docs.js"

IGNORE_DIRS = {'.git', '.obsidian', '.claude', '.neeko', '__pycache__', 'node_modules'}

def parse_frontmatter_info(text):
    info = {'tags': []}
    if text.startswith('---\n') or text.startswith('---\r\n'):
        end_idx = text.find('\n---', 3)
        if end_idx != -1:
            fm_block = text[4:end_idx]
            in_tags = False
            for line in fm_block.splitlines():
                stripped = line.strip()
                if in_tags:
                    if stripped.startswith('- '):
                        info['tags'].append(stripped[2:].strip().strip('"\''))
                        continue
                    elif not stripped.startswith('#'):
                        in_tags = False
                m = re.match(r'^([a-zA-Z0-9_-]+):\s*(.*)$', stripped)
                if m:
                    k, v = m.group(1), m.group(2).strip().strip('"\'')
                    if k == 'tags':
                        if not v:
                            in_tags = True
                        elif v.startswith('[') and v.endswith(']'):
                            info['tags'] = [x.strip().strip('"\'') for x in v[1:-1].split(',') if x.strip()]
                        else:
                            info['tags'] = [v]
                    else:
                        info[k] = v
    return info

def build_tree_and_lists(root):
    md_files = []
    asset_map = {}
    docs_content = {}
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted([d for d in dirnames if not d.startswith('.') and d not in IGNORE_DIRS])
        for f in sorted(filenames):
            if f.startswith('.'):
                continue
            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, root).replace('\\', '/')
            ext = os.path.splitext(f)[1].lower()
            if ext == '.md':
                try:
                    stat = os.stat(full_path)
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as fp:
                        content = fp.read()
                    docs_content[rel_path] = content
                    head = content[:2500]
                    fm = parse_frontmatter_info(head)
                    title = fm.get('title', '')
                    if not title:
                        m_h1 = re.search(r'^#\s+(.+)$', head, re.M)
                        if m_h1:
                            title = m_h1.group(1).strip()
                    
                    block_math = len(re.findall(r'\$\$[\s\S]*?\$\$', content))
                    inline_math = len(re.findall(r'(?<!\$)\$(?!\$)[^\$\n]+?\$(?!\$)', content))
                    is_asset_doc = '/assets/' in ('/' + rel_path + '/') or f == 'sources.md'

                    md_files.append({
                        'path': rel_path,
                        'name': f[:-3],
                        'fileName': f,
                        'title': title or f[:-3],
                        'size': stat.st_size,
                        'mtime': int(stat.st_mtime),
                        'folder': os.path.dirname(rel_path).replace('\\', '/'),
                        'tags': fm.get('tags', [])[:6],
                        'arxiv': fm.get('arxiv', ''),
                        'year': fm.get('year', ''),
                        'status': fm.get('status', ''),
                        'mathCount': block_math + inline_math,
                        'isAssetDoc': is_asset_doc
                    })
                except Exception as e:
                    pass
            elif ext in {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.pdf', '.bmp'}:
                asset_map[rel_path] = rel_path
                parts = rel_path.split('/')
                for i in range(1, min(5, len(parts) + 1)):
                    sub_key = '/'.join(parts[-i:])
                    if sub_key not in asset_map:
                        asset_map[sub_key] = rel_path

    tree = {
        'name': 'obsidian_bk',
        'path': '',
        'type': 'folder',
        'children': [],
        'files': [],
        'mdCount': 0,
        'noteCount': 0,
        'isAssetFolder': False
    }
    folder_nodes = {'': tree}

    def get_folder_node(folder_path):
        if folder_path in folder_nodes:
            return folder_nodes[folder_path]
        parent_path = os.path.dirname(folder_path).replace('\\', '/')
        parent_node = get_folder_node(parent_path)
        folder_name = os.path.basename(folder_path)
        is_asset = folder_name == 'assets' or '/assets/' in ('/' + folder_path + '/') or folder_path.startswith('assets/')
        node = {
            'name': folder_name,
            'path': folder_path,
            'type': 'folder',
            'children': [],
            'files': [],
            'mdCount': 0,
            'noteCount': 0,
            'isAssetFolder': is_asset
        }
        parent_node['children'].append(node)
        folder_nodes[folder_path] = node
        return node

    for item in md_files:
        f_node = get_folder_node(item['folder'])
        f_node['files'].append(item)
        curr = item['folder']
        while True:
            folder_nodes[curr]['mdCount'] += 1
            if not item['isAssetDoc']:
                folder_nodes[curr]['noteCount'] += 1
            if curr == '':
                break
            curr = os.path.dirname(curr).replace('\\', '/')

    return tree, md_files, asset_map, docs_content

if __name__ == '__main__':
    tree, md_files, asset_map, docs_content = build_tree_and_lists(VAULT_ROOT)
    note_count = sum(1 for x in md_files if not x['isAssetDoc'])
    math_total = sum(x['mathCount'] for x in md_files)
    payload = {
        'vaultPath': VAULT_ROOT,
        'generatedAt': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'totalCount': len(md_files),
        'noteCount': note_count,
        'mathTotal': math_total,
        'tree': tree,
        'files': md_files,
        'assetMap': asset_map
    }
    with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as out:
        out.write('window.OBSIDIAN_VAULT_MANIFEST = ')
        json.dump(payload, out, ensure_ascii=False, separators=(',', ':'))
        out.write(';\n')
    with open(OUTPUT_DOCS, 'w', encoding='utf-8') as out:
        out.write('window.OBSIDIAN_VAULT_DOCS = ')
        json.dump(docs_content, out, ensure_ascii=False, separators=(',', ':'))
        out.write(';\n')
    print(f"Manifest generated: {len(md_files)} total md ({note_count} core notes), {math_total} formulas")
    print(f"Manifest size: {os.path.getsize(OUTPUT_MANIFEST)/1024:.1f} KB, Docs bundle size: {os.path.getsize(OUTPUT_DOCS)/(1024*1024):.2f} MB")
