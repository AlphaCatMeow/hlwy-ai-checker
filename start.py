#!/usr/bin/env python3
"""
hlwy-ai-checker AI 模型鉴别器后端代理服务器
解决浏览器 CORS 限制，代理所有 API 请求
"""

import importlib.util
import subprocess
import sys


def _ensure_dependencies():
    """自动检查并安装第三方依赖。

    项目仅依赖 requests。若运行环境未安装，则尝试通过 pip 自动安装，
    默认源失败时回退到清华 PyPI 镜像，避免用户手动处理依赖。
    """
    if importlib.util.find_spec("requests") is not None:
        return

    print("[依赖检查] 未检测到 requests，正在自动安装 ...")
    requirement = "requests>=2.34.2"
    base_cmd = [sys.executable, "-m", "pip", "install", requirement]
    mirrors = (
        ([], "默认源"),
        (["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], "清华镜像"),
    )
    for extra, name in mirrors:
        try:
            subprocess.check_call(base_cmd + extra)
            print("[依赖检查] requests 安装完成。")
            return
        except Exception as e:
            if name == "默认源":
                print(f"[依赖检查] {name}安装失败（{e}），改用清华镜像重试 ...")
            else:
                print(f"[依赖检查] 自动安装失败：{e}")

    print("[依赖检查] 请手动执行：pip install requests")
    sys.exit(1)


_ensure_dependencies()

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import json
import re
import time
import requests as req_lib
import os
import webbrowser
import threading
import uuid
import platform

# ========================================
#  请求头伪装预设 (全小写 key，匹配真实 Node.js SDK)
# ========================================
_STAINLESS_OS = {
    'Darwin': 'MacOS', 'Linux': 'Linux', 'Windows': 'Windows'
}.get(platform.system(), f'Other:{platform.system()}')

_STAINLESS_ARCH = {
    'x86_64': 'x64', 'AMD64': 'x64', 'aarch64': 'arm64', 'arm64': 'arm64',
    'x86': 'x32', 'i386': 'x32', 'i686': 'x32',
}.get(platform.machine(), f'other:{platform.machine()}')

# Codex 安装 ID — 进程生命周期内固定
_CODEX_INSTALLATION_ID = str(uuid.uuid4())

HEADER_PRESETS = {
    'claude-code': {
        'accept': 'application/json',
        'accept-encoding': 'gzip, deflate, br',
        'connection': 'keep-alive',
        'user-agent': 'Anthropic/JS 0.109.0',
        'x-stainless-lang': 'js',
        'x-stainless-package-version': '0.109.0',
        'x-stainless-os': _STAINLESS_OS,
        'x-stainless-arch': _STAINLESS_ARCH,
        'x-stainless-runtime': 'node',
        'x-stainless-runtime-version': 'v22.13.1',
        'x-stainless-retry-count': '0',
    },
    'codex': {
        'accept': 'application/json',
        'accept-encoding': 'gzip, deflate, br',
        'connection': 'keep-alive',
        'user-agent': 'OpenAI/JS 6.45.0',
        'x-stainless-lang': 'js',
        'x-stainless-package-version': '6.45.0',
        'x-stainless-os': _STAINLESS_OS,
        'x-stainless-arch': _STAINLESS_ARCH,
        'x-stainless-runtime': 'node',
        'x-stainless-runtime-version': 'v22.13.1',
        'x-stainless-retry-count': '0',
        'openai-beta': 'responses_websockets=2026-02-06',
        'x-codex-installation-id': _CODEX_INSTALLATION_ID,
    },
}

# 默认浏览器伪装头
DEFAULT_BROWSER_HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'accept-encoding': 'gzip, deflate, br, zstd',
    'accept-language': 'en-US,en;q=0.9',
    'connection': 'keep-alive',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
}

# 创建全局 Session，清除默认头，避免泄漏 python-requests 指纹
_session = req_lib.Session()
_session.headers.clear()


# ========================================
#  一键鉴别 — 官方基准仓库
# ========================================
GITHUB_OWNER  = 'hanlinwenyuan'
GITHUB_REPO   = 'hlwy-ai-checker'
GITHUB_BRANCH = 'main'
BASELINE_DIR  = 'baselines'

# 缓存 TTL（秒）— 避免频繁请求 GitHub 触发匿名 API 限流 (60 次/小时)
BASELINE_LIST_TTL = 300     # 模型列表缓存 5 分钟
BASELINE_FILE_TTL = 1800    # 单个基准文件缓存 30 分钟

_baseline_cache = {'list': None, 'list_source': '', 'list_ts': 0.0, 'files': {}}
_baseline_lock  = threading.Lock()

# 合法基准文件名（防止路径穿越 / URL 注入）
_SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._\-]{0,80}$')

# 拉取 GitHub 用的独立 Session：不带 accept-encoding，避免 zstd 等压缩解码失败
_gh_session = req_lib.Session()
_gh_session.headers.clear()

GITHUB_HEADERS = {
    'accept': 'application/vnd.github+json, application/json, */*',
    'accept-language': 'en-US,en;q=0.9',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
}


def _gh_get(url, timeout=15):
    """带伪装头的 GET，返回解析后的 JSON"""
    resp = _gh_session.get(url, headers=GITHUB_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_baseline_list(force=False):
    """
    获取 baselines 目录下的全部模型。
    源 1: GitHub Contents API   源 2: jsDelivr（国内可用性更好）
    返回 (models, source)，models 形如 [{'id':..., 'file':..., 'size':...}]
    """
    now = time.time()
    with _baseline_lock:
        cached = _baseline_cache['list']
        if not force and cached and (now - _baseline_cache['list_ts']) < BASELINE_LIST_TTL:
            return cached, _baseline_cache['list_source'] + '(缓存)'

    errors = []

    # ---- 源 1: GitHub Contents API ----
    try:
        data = _gh_get(
            f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}'
            f'/contents/{BASELINE_DIR}?ref={GITHUB_BRANCH}'
        )
        models = [
            {'id': it['name'][:-5], 'file': it['name'], 'size': it.get('size', 0)}
            for it in data
            if it.get('type') == 'file'
            and it.get('name', '').endswith('.json')
            and _SAFE_NAME_RE.match(it['name'][:-5] or '')
        ]
        if models:
            return _store_list(models, 'GitHub')
        errors.append('GitHub API: baselines 目录为空')
    except Exception as e:
        errors.append(f'GitHub API: {e}')

    # ---- 源 2: jsDelivr ----
    try:
        data = _gh_get(
            f'https://data.jsdelivr.com/v1/packages/gh/{GITHUB_OWNER}/'
            f'{GITHUB_REPO}@{GITHUB_BRANCH}?structure=flat'
        )
        prefix = f'/{BASELINE_DIR}/'
        models = []
        for it in data.get('files', []):
            name = it.get('name', '')
            if not name.startswith(prefix) or not name.endswith('.json'):
                continue
            stem = name[len(prefix):-5]
            if '/' in stem or not _SAFE_NAME_RE.match(stem):
                continue
            models.append({'id': stem, 'file': stem + '.json', 'size': it.get('size', 0)})
        if models:
            return _store_list(models, 'jsDelivr')
        errors.append('jsDelivr: baselines 目录为空')
    except Exception as e:
        errors.append(f'jsDelivr: {e}')

    # 全部失败：如果有过期缓存，降级返回，总比什么都没有强
    with _baseline_lock:
        if _baseline_cache['list']:
            return _baseline_cache['list'], _baseline_cache['list_source'] + '(过期缓存)'

    raise RuntimeError('；'.join(errors))


def _store_list(models, source):
    models.sort(key=lambda m: m['id'].lower())
    with _baseline_lock:
        _baseline_cache['list']        = models
        _baseline_cache['list_source'] = source
        _baseline_cache['list_ts']     = time.time()
    return models, source


def fetch_baseline_file(name, force=False):
    """下载单个基准文件，raw.githubusercontent 失败时回落 jsDelivr。返回 (data, source)"""
    if not _SAFE_NAME_RE.match(name):
        raise ValueError('基准名称不合法')

    now = time.time()
    with _baseline_lock:
        hit = _baseline_cache['files'].get(name)
        if not force and hit and (now - hit['ts']) < BASELINE_FILE_TTL:
            return hit['data'], hit['source'] + '(缓存)'

    errors = []
    sources = [
        ('GitHub', f'https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}'
                   f'/{GITHUB_BRANCH}/{BASELINE_DIR}/{name}.json'),
        ('jsDelivr', f'https://cdn.jsdelivr.net/gh/{GITHUB_OWNER}/{GITHUB_REPO}'
                     f'@{GITHUB_BRANCH}/{BASELINE_DIR}/{name}.json'),
    ]

    for source, url in sources:
        try:
            data = _gh_get(url, timeout=20)
            with _baseline_lock:
                _baseline_cache['files'][name] = {'data': data, 'source': source, 'ts': time.time()}
            return data, source
        except Exception as e:
            errors.append(f'{source}: {e}')

    with _baseline_lock:
        hit = _baseline_cache['files'].get(name)
        if hit:
            return hit['data'], hit['source'] + '(过期缓存)'

    raise RuntimeError('；'.join(errors))


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ProxyHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        """处理 CORS 预检请求"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """处理 GET 请求 - 提供 HTML 文件、静态资源和官方基准"""
        parsed = urlparse(self.path)
        path   = parsed.path
        query  = parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self.serve_html()
        elif path == '/chart.js':
            self.serve_static('chart.js', 'application/javascript')
        elif path == '/api/baselines':
            self.serve_baseline_list(query)
        elif path == '/api/baseline':
            self.serve_baseline_file(query)
        else:
            self.send_error(404, "File not found")

    def serve_baseline_list(self, query):
        """返回 GitHub baselines 目录下的模型列表"""
        force = query.get('refresh', ['0'])[0] in ('1', 'true')
        try:
            models, source = fetch_baseline_list(force=force)
            self.send_json_response(200, {
                'models': models,
                'count': len(models),
                'source': source,
                'repo': f'{GITHUB_OWNER}/{GITHUB_REPO}',
            })
        except Exception as e:
            self.send_json_response(502, {
                'error': '无法获取官方基准列表',
                'detail': str(e),
            })

    def serve_baseline_file(self, query):
        """下载并返回单个官方基准"""
        name = (query.get('name', [''])[0] or '').strip()
        if not name:
            self.send_json_response(400, {'error': '缺少 name 参数'})
            return
        if not _SAFE_NAME_RE.match(name):
            self.send_json_response(400, {'error': '基准名称不合法'})
            return

        force = query.get('refresh', ['0'])[0] in ('1', 'true')
        try:
            data, source = fetch_baseline_file(name, force=force)
            self.send_json_response(200, {'name': name, 'source': source, 'data': data})
        except Exception as e:
            self.send_json_response(502, {
                'error': f'无法下载基准「{name}」',
                'detail': str(e),
            })

    def do_POST(self):
        """处理 POST 请求 - 代理 API 调用"""
        # 代理所有 OpenAI 和 Anthropic API 请求
        if '/chat/completions' in self.path or '/messages' in self.path or '/responses' in self.path:
            self.proxy_api_request()
        else:
            self.send_error(404, "Endpoint not found")

    def serve_html(self):
        """返回 HTML 文件"""
        try:
            with open('hlwy-ai-checker.html', 'r', encoding='utf-8') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except FileNotFoundError:
            self.send_error(404, "hlwy-ai-checker.html not found")

    def serve_static(self, filename, content_type):
        """返回静态文件"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except FileNotFoundError:
            self.send_error(404, f"{filename} not found")

    def proxy_api_request(self):
        """代理 API 请求到真实的 API 端点"""
        try:
            # 读取请求体
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            # 确定目标 URL
            if '/chat/completions' in self.path:
                # OpenAI Chat Completions API
                base_url = self.headers.get('X-Target-Base-URL', 'https://api.openai.com/v1')
                target_url = f"{base_url.rstrip('/')}/chat/completions"
            elif '/responses' in self.path:
                # OpenAI Responses API
                base_url = self.headers.get('X-Target-Base-URL', 'https://api.openai.com/v1')
                target_url = f"{base_url.rstrip('/')}/responses"
            elif '/messages' in self.path:
                # Anthropic API
                base_url = self.headers.get('X-Target-Base-URL', 'https://api.anthropic.com/v1')
                target_url = f"{base_url.rstrip('/')}/messages"
            else:
                self.send_json_response(400, {'error': '不支持的 API 端点'})
                return

            # 获取请求头伪装预设
            header_preset = self.headers.get('X-Header-Preset', 'default')

            # 构建代理请求头 (全小写 key)
            if header_preset in HEADER_PRESETS:
                headers = dict(HEADER_PRESETS[header_preset])
                # 每次请求动态生成唯一 request-id
                headers['x-request-id'] = f'req_{uuid.uuid4().hex}'
            else:
                headers = dict(DEFAULT_BROWSER_HEADERS)

            # 复制必要的业务请求头 (保持小写 key)
            header_map = {
                'Content-Type': 'content-type',
                'Authorization': 'authorization',
                'anthropic-version': 'anthropic-version',
                'x-api-key': 'x-api-key',
            }
            for src_key, dst_key in header_map.items():
                val = self.headers.get(src_key)
                if val:
                    headers[dst_key] = val

            # 移除 accept-encoding 避免收到压缩响应后原样转发导致浏览器解析失败
            headers.pop('accept-encoding', None)

            # 使用 requests 发送请求 (保留原始 header 大小写)
            try:
                resp = self._post_adaptive(target_url, body, headers)

                self.send_response(resp.status_code)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(resp.content)

            except req_lib.exceptions.ConnectionError as e:
                self.send_json_response(500, {'error': f'网络错误: {str(e)}'})
            except req_lib.exceptions.Timeout as e:
                self.send_json_response(504, {'error': f'请求超时: {str(e)}'})

        except Exception as e:
            self.send_json_response(500, {'error': f'服务器错误: {str(e)}'})

    def _post_adaptive(self, target_url, body, headers):
        """发送 POST，遇到 max_tokens / max_completion_tokens 参数不兼容时自动切换重试。

        新版模型（gpt-5 / o 系列等）只接受 max_completion_tokens，而旧模型/部分中转
        只接受 max_tokens，前端写死 max_tokens 会导致 400。这里根据报错自动降级切换。
        """
        resp = _session.post(target_url, data=body, headers=headers, timeout=30)
        if resp.status_code != 400 or '/chat/completions' not in self.path:
            return resp

        try:
            err_text = resp.text.lower()
        except Exception:
            return resp

        # 仅当报错确实指向这两个参数时才处理，避免误伤其他 400 错误
        if 'max_tokens' not in err_text and 'max_completion_tokens' not in err_text:
            return resp
        if 'unsupported_parameter' not in err_text and 'unknown parameter' not in err_text:
            return resp

        try:
            body_obj = json.loads(body)
        except Exception:
            return resp
        if not isinstance(body_obj, dict):
            return resp

        changed = False
        if 'max_tokens' in body_obj and 'max_completion_tokens' in err_text:
            body_obj['max_completion_tokens'] = body_obj.pop('max_tokens')
            changed = True
        elif 'max_completion_tokens' in body_obj and 'max_tokens' in err_text:
            body_obj['max_tokens'] = body_obj.pop('max_completion_tokens')
            changed = True

        if not changed:
            return resp
        return _session.post(target_url, data=json.dumps(body_obj), headers=headers, timeout=30)

    def send_json_response(self, status_code, data):
        """发送 JSON 响应"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def send_cors_headers(self):
        """添加 CORS 头"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, anthropic-version, x-api-key, X-Target-Base-URL, X-Header-Preset')

    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    HOST = 'localhost'
    PORT = 8000

    # 检查 HTML ��件是否存在
    if not os.path.exists('hlwy-ai-checker.html'):
        print("错误: 找不到 hlwy-ai-checker.html 文件")
        print("请确保在包含该文件的目录中运行此脚本")
        return

    server = ThreadingHTTPServer((HOST, PORT), ProxyHandler)
    url = f'http://{HOST}:{PORT}'
    print(f"""
╔════════════════════════════════════════════════════════╗
║      hlwy-ai-checker v2.5-pre1 - AI 模型鉴别器         ║
╚════════════════════════════════════════════════════════╝
本项目github地址：https://github.com/hanlinwenyuan/hlwy-ai-checker

🌐 前端访问地址: {url}
⚡ 默认页面「一键鉴别」会自动从 GitHub 下载官方基准，无需手动标定

按 Ctrl+C 停止
""")

    threading.Timer(0.5, webbrowser.open, args=[url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n已停止")
        server.shutdown()


if __name__ == '__main__':
    main()
