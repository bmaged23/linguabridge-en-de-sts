"""An unsuccessful smoke launch must never remove somebody else's container."""
import subprocess
from types import SimpleNamespace
import pytest
from scripts.release import smoke_container


def test_name_collision_preserves_existing_container(monkeypatch):
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        if args[1] == 'run':
            raise subprocess.CalledProcessError(125, args)
        if args[1] == 'inspect':
            return SimpleNamespace(returncode=0, stdout='different-owner\n')
        raise AssertionError('Foreign container must not be removed')
    monkeypatch.setattr(smoke_container.subprocess, 'run', command)
    monkeypatch.setattr('sys.argv', ['smoke', '--image', 'candidate', '--name', 'existing'])
    with pytest.raises(subprocess.CalledProcessError):
        smoke_container.main()
    assert not any(args[1] == 'rm' for args in calls)


def test_failed_launch_cleans_its_own_partial_container(monkeypatch):
    calls = []; owner = []
    def command(args, **kwargs):
        calls.append(args)
        if args[1] == 'run':
            owner.append(args[args.index('--label') + 1].split('=', 1)[1])
            raise subprocess.CalledProcessError(125, args)
        if args[1] == 'inspect':
            return SimpleNamespace(returncode=0, stdout=owner[0] + '\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(smoke_container.subprocess, 'run', command)
    monkeypatch.setattr('sys.argv', ['smoke', '--image', 'candidate'])
    with pytest.raises(subprocess.CalledProcessError):
        smoke_container.main()
    assert any(args[1] == 'rm' for args in calls)
