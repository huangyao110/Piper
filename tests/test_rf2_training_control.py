"""The long-run launcher preserves checkpoints and retries only native crashes."""
import importlib.util
import hashlib
import json
from pathlib import Path
import signal

import pytest

pytestmark = pytest.mark.integration


def controller_fixture(tmp_path, monkeypatch, exits):
    root = Path(__file__).resolve().parents[1]
    source = root / 'experiments/rf2ppi_v3_fresh_4epochs_20261024/control.py'
    if not source.is_file():
        pytest.skip('Requires the local archived run controller, excluded from the source release')
    spec = importlib.util.spec_from_file_location('rf2_training_control_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exp, storage = tmp_path / 'experiments/run', tmp_path / 'storage'
    exp.mkdir(parents=True)
    runtime = exp / 'runtime'
    runtime.mkdir()
    config = dict(epochs=4, steps=4216)
    checkpoint = storage / 'preflight_steps_000000_000002/checkpoints/step_000002'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text(json.dumps(config))
    (checkpoint / 'manifest.json').write_text(json.dumps(dict(step=2)))
    (checkpoint.parent.parent / 'latest.json').write_text(json.dumps(dict(checkpoint='checkpoints/step_000002')))
    (exp / 'config.json').write_text(json.dumps(config))
    (exp / 'runtime_sha256.json').write_text('{}')
    (exp / 'gpu_preflight.json').write_text('{"state":"passed"}')
    (tmp_path / 'experiments/active_training.json').write_text('{}')
    for name, value in dict(EXP=exp, ROOT=tmp_path, STORAGE=storage, RUNTIME=runtime,
                            PYTHON=tmp_path/'python').items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.pid = 100 + len(commands)
            self.code = exits[len(commands)-1]
            output = Path(command[command.index('--output')+1])
            output.mkdir()
            step = 4216 if self.code == 0 else 200 * len(commands)
            (output / 'status.json').write_text(json.dumps(dict(
                status='complete' if self.code == 0 else 'testing', step=step)))
            checkpoint = output / 'checkpoints' / f'step_{step:06d}'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'manifest.json').write_text(json.dumps(dict(step=step)))
            (output / 'latest.json').write_text(json.dumps(dict(checkpoint=f'checkpoints/step_{step:06d}')))

        def wait(self):
            return self.code

    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    return module, commands, exp


def test_controller_continues_from_verified_step_two(tmp_path, monkeypatch):
    module, commands, exp = controller_fixture(tmp_path, monkeypatch, [0])
    module.main()
    assert len(commands) == 1
    assert commands[0][commands[0].index('--resume')+1].endswith('step_000002')
    assert json.loads((exp / 'status.json').read_text())['phase'] == 'complete'


def test_controller_native_crash_restarts_at_last_saved_test_boundary(tmp_path, monkeypatch):
    module, commands, exp = controller_fixture(tmp_path, monkeypatch, [-signal.SIGSEGV, 0])
    module.main()
    assert len(commands) == 2
    assert commands[1][commands[1].index('--resume')+1].endswith('train_attempt_01/checkpoints/step_000200')
    status = json.loads((exp / 'status.json').read_text())
    assert status['phase'] == 'complete' and len(status['incidents']) == 1


@pytest.mark.parametrize('exits', [[1], [-signal.SIGSEGV] * 3])
def test_controller_has_bounded_retries_and_reports_failure(tmp_path, monkeypatch, exits):
    module, commands, exp = controller_fixture(tmp_path, monkeypatch, exits)
    with pytest.raises(RuntimeError, match='no further automatic retry'):
        module.main()
    assert len(commands) == len(exits)
    assert json.loads((exp / 'status.json').read_text())['phase'] == 'failed'
    assert json.loads((module.ROOT / 'experiments/active_training.json').read_text())['phase'] == 'failed'


def test_controller_requires_passed_gpu_preflight(tmp_path, monkeypatch):
    module, commands, exp = controller_fixture(tmp_path, monkeypatch, [0])
    (exp / 'gpu_preflight.json').write_text('{"state":"failed"}')
    with pytest.raises(AssertionError):
        module.main()
    assert not commands


def recovery_checkpoint(module, exp):
    checkpoint = module.STORAGE / 'train_attempt_01/checkpoints/step_000100'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text((exp / 'config.json').read_text())
    manifest = dict(step=100)
    for filename, key in (('trainable.safetensors', 'trainable_sha256'),
                          ('training_state.pt', 'training_state_sha256')):
        data = b'fixture for ' + filename.encode()
        (checkpoint / filename).write_bytes(data)
        manifest[key] = hashlib.sha256(data).hexdigest()
    (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
    (exp / 'status.json').write_text(json.dumps(dict(
        phase='interrupted_by_kernel_panic', attempts=[{'start_step': 2, 'through_step': 4216}],
        incidents=[])))
    return checkpoint


def test_recovery_dry_run_preserves_history_and_uses_step_100(tmp_path, monkeypatch):
    module, commands, exp = controller_fixture(tmp_path, monkeypatch, [])
    checkpoint = recovery_checkpoint(module, exp)
    before = (exp / 'status.json').read_bytes()
    monkeypatch.setattr(module, 'nohz_disabled', lambda: False)
    plan = module.main(checkpoint, require_nohz_off=True, dry_run=True)
    assert plan['start_step'] == 100 and plan['through_step'] == 4216
    assert plan['output'].endswith('train_attempt_02')
    assert not plan['nohz_off_active'] and not plan['training_started']
    assert not commands and (exp / 'status.json').read_bytes() == before
    assert not (module.STORAGE / 'train_attempt_02').exists()


def test_recovery_refuses_start_until_workaround_is_active(tmp_path, monkeypatch):
    module, commands, exp = controller_fixture(tmp_path, monkeypatch, [])
    checkpoint = recovery_checkpoint(module, exp)
    monkeypatch.setattr(module, 'nohz_disabled', lambda: False)
    with pytest.raises(RuntimeError, match='nohz=off is not active'):
        module.main(checkpoint, require_nohz_off=True)
    assert not commands
    assert json.loads((exp / 'status.json').read_text())['phase'] == 'interrupted_by_kernel_panic'


def test_recovery_preserves_attempt_history_and_resumes_exact_checkpoint(tmp_path, monkeypatch):
    module, commands, exp = controller_fixture(tmp_path, monkeypatch, [0])
    checkpoint = recovery_checkpoint(module, exp)
    monkeypatch.setattr(module, 'nohz_disabled', lambda: True)
    module.main(checkpoint, require_nohz_off=True)
    assert commands[0][commands[0].index('--resume')+1] == str(checkpoint)
    assert commands[0][commands[0].index('--output')+1].endswith('train_attempt_02')
    assert (checkpoint / 'training_state.pt').read_bytes() == b'fixture for training_state.pt'
    status = json.loads((exp / 'status.json').read_text())
    assert len(status['attempts']) == 2
    assert status['attempts'][0]['start_step'] == 2
    assert status['attempts'][1]['start_step'] == 100
    assert status['recovery']['nohz_off_active'] and status['phase'] == 'complete'


@pytest.mark.parametrize('damage', ['checksum', 'config'])
def test_recovery_rejects_modified_checkpoint(tmp_path, monkeypatch, damage):
    module, commands, exp = controller_fixture(tmp_path, monkeypatch, [])
    checkpoint = recovery_checkpoint(module, exp)
    if damage == 'checksum':
        (checkpoint / 'training_state.pt').write_bytes(b'corrupted')
    else:
        (checkpoint / 'config.json').write_text('{"steps": 99, "epochs": 1}')
    with pytest.raises((ValueError, AssertionError)):
        module.main(checkpoint, dry_run=True)
    assert not commands
