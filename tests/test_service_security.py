import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from prediction_service.config import Settings, _load_or_create_session_secret


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_secret_and_data_directory_are_private_even_with_permissive_umask(tmp_path):
    config = tmp_path / "config.ini"
    config.write_text("[服务]\n启用定时刷新 = 否\n", encoding="utf-8")
    previous = os.umask(0)
    try:
        settings = Settings.from_config(config)
    finally:
        os.umask(previous)
    assert settings.root_dir.stat().st_mode & 0o777 == 0o700
    assert (settings.root_dir / "session_secret.txt").stat().st_mode & 0o777 == 0o600
    assert Settings.from_config(config).session_secret == settings.session_secret


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_existing_secret_is_secured_without_changing_sessions(tmp_path):
    path = tmp_path / "session_secret.txt"
    path.write_text("existing-secret\n")
    path.chmod(0o666)
    assert _load_or_create_session_secret(tmp_path, None) == "existing-secret"
    assert path.stat().st_mode & 0o777 == 0o600


def test_concurrent_configuration_uses_one_persisted_secret(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: _load_or_create_session_secret(tmp_path, None), range(8)))
    assert len(set(values)) == 1
    assert (tmp_path / "session_secret.txt").read_text().strip() == values[0]


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink protection")
def test_session_secret_cannot_follow_a_symlink(tmp_path):
    target = tmp_path / "other-file"
    target.write_text("do-not-use-or-modify")
    (tmp_path / "session_secret.txt").symlink_to(target)
    with pytest.raises(OSError):
        _load_or_create_session_secret(tmp_path, None)
    assert target.read_text() == "do-not-use-or-modify"
