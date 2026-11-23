"""AICost text and image transport for Story Studio."""
import argparse
import base64
import http.client
import io
import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import urlopen

from studio_runtime import data_dir
from voice_settings import private_write, windows_protect


class ApiError(Exception):
    def __init__(self, kind, status=0):
        self.kind = kind
        self.status = status
        super().__init__(kind)


def error_kind(status, error):
    detail = json.dumps(error, ensure_ascii=False).lower()
    if status in (401, 402, 403) or any(term in detail for term in (
        'insufficient_quota', 'invalid_api_key', 'content_policy', 'content_filter',
        'moderation_blocked', '余额不足', '内容审核',
    )):
        return 'authorization_quota_or_policy'
    if status == 429 or any(term in detail for term in (
        'overloaded', 'overload', 'rate_limit', 'server_busy', 'service unavailable',
        'temporarily unavailable', 'too many requests', '过载', '负载过高', '繁忙', '限流',
    )):
        return 'temporary'
    if any(term in detail for term in (
        'model_not_found', 'model_not_available', 'model_unavailable', 'invalid_model',
        'unsupported_model', 'model_not_supported', 'model_error', 'model_disabled',
        'no available channel', '模型不存在', '模型不可用', '模型失败', '模型不支持',
    )):
        return 'model'
    if status in (408, 500, 502, 503, 504):
        return 'temporary'
    return 'request_failed'


def request(base, key, endpoint, body=None, content_type='application/json', timeout=420):
    address = urlsplit(base)
    connection = http.client.HTTPSConnection(address.hostname, address.port, timeout=timeout)
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': content_type}
    method = 'GET' if body is None else 'POST'
    if isinstance(body, dict):
        body = json.dumps(body, ensure_ascii=False).encode('utf-8')
    try:
        connection.request(method, address.path.rstrip('/') + endpoint, body, headers)
        response = connection.getresponse()
        status = response.status
        raw = response.read()
    except (OSError, http.client.HTTPException) as exc:
        # Without a response, an image submission may already have been accepted.
        kind = 'submission_unconfirmed' if method == 'POST' and endpoint.startswith('/images/') else 'temporary'
        raise ApiError(kind) from exc
    finally:
        connection.close()
    try:
        result = json.loads(raw)
    except (UnicodeError, ValueError):
        result = {}
    task_status = method == 'GET' and endpoint.startswith('/images/generations/') and result.get('status')
    if status >= 300 or (result.get('error') and not task_status):
        raise ApiError(error_kind(status, result.get('error') or result), status)
    return result


def route(models, invoke, attempts):
    for model in models:
        tried = [attempt for attempt in attempts if attempt['model'] == model]
        if tried and (tried[-1]['result'] == 'model' or len(tried) >= 2):
            continue
        for index in range(len(tried), 2):
            if index:
                time.sleep(3)
            try:
                result = invoke(model)
            except ApiError as exc:
                attempts.append({'model': model, 'result': exc.kind, 'http': exc.status})
                if exc.kind == 'model':
                    break
                if exc.kind != 'temporary':
                    raise
            else:
                attempts.append({'model': model, 'result': 'ok'})
                return model, result
    raise ApiError('models_exhausted')


def multipart(fields, references):
    boundary = uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for index, path in enumerate(references, 1):
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        filename = f'reference-{index}{mimetypes.guess_extension(mime) or ".bin"}'
        header = (f'--{boundary}\r\nContent-Disposition: form-data; name="image[]"; filename="{filename}"\r\n'
                  f'Content-Type: {mime}\r\n\r\n')
        chunks.extend((header.encode(), path.read_bytes(), b'\r\n'))
    chunks.append(f'--{boundary}--\r\n'.encode())
    return b''.join(chunks), f'multipart/form-data; boundary={boundary}'


def image_result(result):
    item = result.get('data') or result.get('output') or result.get('result') or result
    if isinstance(item, list):
        item = item[0]
    encoded = item.get('b64_json') or item.get('image_base64') or item.get('base64')
    if encoded:
        return base64.b64decode(encoded, validate=True)
    url = item.get('url') or item.get('image_url')
    if isinstance(url, dict):
        url = url.get('url')
    if url and urlsplit(url).scheme == 'https':
        with urlopen(url, timeout=90) as response:
            return response.read()
    raise ApiError('image_result_missing')


def run_image(args, send, models, prompt, attempts):
    from PIL import Image

    fields = {'prompt': prompt, 'n': 1, 'size': args.size, 'quality': 'auto',
              'output_format': 'png', 'moderation': 'auto'}

    def submit(model):
        payload = {'model': model, **fields}
        if args.reference:
            body, content_type = multipart(payload, args.reference)
            return send('/images/edits', body, content_type, timeout=3600)
        return send('/images/generations', payload, timeout=3600)

    receipt = args.output.with_name(args.output.name + '.aicost-task.json')
    pending_id = None
    if receipt.is_file():
        state = json.loads(receipt.read_text(encoding='utf-8'))
        model, pending_id = state['model'], state['task_id']
        attempts.extend(state['attempts'])
        result = send('/images/generations/' + quote(pending_id, safe=''))
        status = str(result.get('status', '')).lower()
        if status in ('pending', 'queued', 'running', 'processing', 'in_progress', 'not_start'):
            return {'status': 'pending', 'model': model, 'task_id': pending_id}
        if status in ('failed', 'failure', 'error'):
            kind = error_kind(0, result)
            if kind not in ('model', 'temporary'):
                raise ApiError(kind)
            attempts[-1]['result'] = kind
            model, result = route(models, submit, attempts)
        elif status in ('cancelled', 'canceled', 'rejected', 'content_filter'):
            raise ApiError('image_task_' + status)
    else:
        model, result = route(models, submit, attempts)
    task_id = result.get('task_id') if not result.get('data') else None
    if task_id and task_id != pending_id:
        receipt.write_text(json.dumps({'model': model, 'task_id': task_id, 'attempts': attempts}), encoding='utf-8')
        return {'status': 'pending', 'model': model, 'task_id': task_id}
    content = image_result(result)
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        image.save(args.output, format='PNG')
    receipt.unlink(missing_ok=True)
    return {'status': 'ok', 'model': model, 'output': str(args.output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    setup = commands.add_parser('configure')
    for argument in ('base', 'text-primary', 'text-backup', 'image-primary', 'image-backup'):
        setup.add_argument('--' + argument, required=True)
    setup.add_argument('--key-file', type=Path, required=True)
    commands.add_parser('check')
    for name in ('text', 'image'):
        job = commands.add_parser(name)
        job.add_argument('--prompt-file', type=Path, required=True)
        job.add_argument('--output', type=Path, required=True)
    image_job = commands.choices['image']
    image_job.add_argument('--size', default='864x1536')
    image_job.add_argument('--reference', action='append', type=Path, default=[])
    args = parser.parse_args()
    config_path = data_dir() / 'aicost.dpapi'
    attempts = []
    try:
        if args.action == 'configure':
            address = urlsplit(args.base)
            if address.scheme != 'https' or address.hostname not in ('aicost.me', 'www.aicost.me') or address.path.rstrip('/') != '/v1':
                raise ApiError('invalid_aicost_base')
            keys = re.findall(r'(?<![A-Za-z0-9_-])sk-[A-Za-z0-9._-]{12,}(?![A-Za-z0-9_-])',
                              args.key_file.read_text(encoding='utf-8-sig'))
            if len(keys) != 2:
                raise ApiError('expected_two_keys_in_text_image_order')
            config = {'base': args.base, 'text': {'key': keys[0], 'models': [args.text_primary, args.text_backup]},
                      'image': {'key': keys[1], 'models': [args.image_primary, args.image_backup]}}
            private_write(config_path, windows_protect(json.dumps(config).encode()))
            report = {'status': 'configured', 'text': config['text']['models'], 'image': config['image']['models']}
        else:
            config = json.loads(windows_protect(config_path.read_bytes(), decrypt=True))
            if args.action == 'check':
                report = {'status': 'ok'}
                for name in ('text', 'image'):
                    slot = config[name]
                    available = {item['id'] for item in request(config['base'], slot['key'], '/models')['data']}
                    report[name] = {model: model in available for model in slot['models']}
                if not all(all(report[name].values()) for name in ('text', 'image')):
                    report['status'] = 'model_missing'
            else:
                prompt = args.prompt_file.read_text(encoding='utf-8-sig').strip()
                if not prompt or args.output.exists():
                    raise ApiError('empty_prompt_or_output_exists')
                args.output.parent.mkdir(parents=True, exist_ok=True)
                slot = config[args.action]

                def send(endpoint, *parts, **options):
                    return request(config['base'], slot['key'], endpoint, *parts, **options)

                if args.action == 'image':
                    report = run_image(args, send, slot['models'], prompt, attempts)
                else:
                    def complete(model):
                        data = send('/chat/completions', {'model': model, 'messages': [{'role': 'user', 'content': prompt}]})
                        content = (data.get('choices') or [{}])[0].get('message', {}).get('content')
                        if not isinstance(content, str) or not content.strip():
                            raise ApiError('temporary')
                        return content

                    model, content = route(slot['models'], complete, attempts)
                    args.output.write_text(content, encoding='utf-8')
                    report = {'status': 'ok', 'model': model, 'output': str(args.output)}
    except ApiError as exc:
        report = {'status': exc.kind, 'http': exc.status}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {'status': 'local_or_response_error', 'class': type(exc).__name__}
    if attempts:
        report['attempts'] = attempts
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['status'] in ('ok', 'configured', 'pending') else 1


if __name__ == '__main__':
    raise SystemExit(main())
