
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {'__pycache__', '.git', 'test', '.idea', 'venv', '.venv',
             'node_modules', '.mypy_cache', '.pytest_cache', 'dist', 'build'}
SKIP_FILES = {'dump_Z1DB.py', 'dump_project.py'}
SKIP_PREFIXES = ('z1db_part', 'z1db_full')
MAX_CHARS_PER_PART = 5000000


def collect_files():
    """收集所有 .py 文件，跳过空 __init__.py，按路径排序。"""
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if not fn.endswith('.py'):
                continue
            if fn in SKIP_FILES:
                continue
            if any(fn.startswith(p) for p in SKIP_PREFIXES):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT).replace('\\', '/')
            try:
                with open(full, 'r', encoding='utf-8') as f:
                    content = f.read().rstrip()
            except Exception as e:
                content = f"# ERROR reading file: {e}"
            # 跳过几乎空的 __init__.py
            if fn == '__init__.py':
                stripped = content.strip()
                # 保留有实质内容的 (>2行 或 >80字符)
                line_count = stripped.count('\n') + 1
                if line_count <= 2 and len(stripped) < 100:
                    continue
            files.append((rel, content))
    return files


def make_entry(rel: str, content: str) -> str:
    line_count = content.count('\n') + 1
    return f"=== {rel} ({line_count}L) ===\n{content}\n"


def split_parts(files: list) -> list:
    """按大小分段，确保不拆分单个文件。"""
    parts = []
    current = []
    current_size = 0
    for rel, content in files:
        entry = make_entry(rel, content)
        entry_size = len(entry)
        if entry_size > MAX_CHARS_PER_PART:
            if current:
                parts.append(''.join(current))
                current = []
                current_size = 0
            parts.append(entry)
            continue
        if current_size + entry_size > MAX_CHARS_PER_PART:
            parts.append(''.join(current))
            current = []
            current_size = 0
        current.append(entry)
        current_size += entry_size
    if current:
        parts.append(''.join(current))
    return parts


def build_tree(files: list) -> str:
    """生成目录树。"""
    dirs = set()
    for rel, _ in files:
        parts = rel.split('/')
        for i in range(1, len(parts)):
            dirs.add('/'.join(parts[:i]))
    lines = ["目录结构:"]
    for d in sorted(dirs):
        depth = d.count('/')
        name = d.split('/')[-1]
        lines.append(f"  {'  ' * depth}{name}/")
    return '\n'.join(lines)


def count_stats(files: list) -> dict:
    """统计代码信息。"""
    total_lines = sum(c.count('\n') + 1 for _, c in files)
    total_bytes = sum(len(c.encode('utf-8')) for _, c in files)
    # 按目录分组统计
    dir_stats = {}
    for rel, content in files:
        top_dir = rel.split('/')[0] if '/' in rel else '(root)'
        if top_dir not in dir_stats:
            dir_stats[top_dir] = {'files': 0, 'lines': 0}
        dir_stats[top_dir]['files'] += 1
        dir_stats[top_dir]['lines'] += content.count('\n') + 1
    return {
        'files': len(files),
        'lines': total_lines,
        'bytes': total_bytes,
        'dirs': dir_stats,
    }


def main():
    files = collect_files()
    stats = count_stats(files)
    tree = build_tree(files)

    # 打印摘要
    print(f"Z1DB 项目摘要")
    print(f"{'=' * 50}")
    print(f"  文件数: {stats['files']}")
    print(f"  总行数: {stats['lines']:,}")
    print(f"  总大小: {stats['bytes']:,} 字节 ({stats['bytes']/1024:.1f} KB)")
    print()

    # 按目录统计
    print("各层统计:")
    for d in sorted(stats['dirs'].keys()):
        ds = stats['dirs'][d]
        print(f"  {d:<30s} {ds['files']:>3d} files  {ds['lines']:>5,d} lines")
    print()

    # 目录树
    print(tree)
    print()

    # 分段
    parts = split_parts(files)

    # 清理旧输出
    for fn in os.listdir(ROOT):
        if fn.startswith('z1db_part') and fn.endswith('.txt'):
            os.unlink(os.path.join(ROOT, fn))
        if fn == 'z1db_full.txt':
            os.unlink(os.path.join(ROOT, fn))

    # 写分段文件
    for i, part in enumerate(parts):
        fname = f"z1db_part{i + 1}.txt"
        header = (
            f"--- Z1DB 代码段 {i + 1}/{len(parts)} ---\n"
            f"--- 文件数: {stats['files']} | 总行数: {stats['lines']:,} ---\n\n"
        )
        with open(os.path.join(ROOT, fname), 'w', encoding='utf-8') as f:
            f.write(header + part)
        size_kb = len(part) / 1024
        file_count = part.count('=== ')

    # 完整版
    full = '\n'.join(make_entry(r, c) for r, c in files)
    full_header = (
        f"--- Z1DB 完整代码 ---\n"
        f"--- 文件数: {stats['files']} | 总行数: {stats['lines']:,} | "
        f"大小: {stats['bytes']:,} 字节 ---\n\n"
        f"{tree}\n\n"
    )
    with open(os.path.join(ROOT, 'z1db_full.txt'), 'w', encoding='utf-8') as f:
        f.write(full_header + full)
    full_kb = len(full) / 1024

if __name__ == '__main__':
    main()
