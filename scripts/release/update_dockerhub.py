"""Synchronize the maintained public image overview after successful publication."""
import json
import os
from pathlib import Path
import urllib.request


def request(url, value, token=None, method='POST'):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = 'JWT ' + token
    message = urllib.request.Request(url, json.dumps(value).encode(), headers, method=method)
    with urllib.request.urlopen(message, timeout=45) as response:
        return json.load(response)


def main():
    login = request('https://hub.docker.com/v2/users/login/', {
        'username': 'bmaged23', 'password': os.environ['DOCKERHUB_TOKEN']})
    root = Path(__file__).resolve().parents[2]
    result = request('https://hub.docker.com/v2/repositories/bmaged23/linguabridge-en-de-sts/', {
        'description': 'English-German GPU speech translation with cloud LLM, memory and voice interruption.',
        'full_description': (root / 'deployment/dockerhub/README.md').read_text()},
        login['token'], method='PATCH')
    if result.get('is_private') is True:
        raise RuntimeError('Published repository is private; public release is not verified')
    print('Docker Hub overview synchronized with maintained documentation.')


if __name__ == '__main__':
    main()
