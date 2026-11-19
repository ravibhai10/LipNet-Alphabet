"""Create persistent local access credentials once; never rotate ownership on startup."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess


def _windows_acl_is_private(path: Path) -> bool:
    """Fail closed unless only the current account, SYSTEM and Administrators can read it."""
    if os.name != 'nt':
        return False
    script = r"""
$ErrorActionPreference = 'Stop'
try {
  $path = '__TARGET__'
  $acl = Get-Acl -LiteralPath $path
  $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
  $ownerSid = $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value
  $allowed = @($currentSid, 'S-1-5-18', 'S-1-5-32-544')
  $allowSids = @()
  $currentHasFullControl = $false
  foreach ($entry in $acl.Access) {
    if ($entry.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow) {
      $entrySid = $entry.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
      $allowSids += $entrySid
      if ($entrySid -eq $currentSid -and
          (($entry.FileSystemRights -band [Security.AccessControl.FileSystemRights]::FullControl) -eq
           [Security.AccessControl.FileSystemRights]::FullControl)) {
        $currentHasFullControl = $true
      }
    }
  }
  $unexpected = @($allowSids | Where-Object { $allowed -notcontains $_ })
  $result = @{ restricted = (($allowed -contains $ownerSid) -and $unexpected.Count -eq 0 -and $currentHasFullControl) }
  $result | ConvertTo-Json -Compress
} catch { exit 2 }
"""
    escaped = str(path.resolve()).replace("'", "''")
    command = script.replace('__TARGET__', escaped)
    encoded = base64.b64encode(command.encode('utf-16le')).decode('ascii')
    try:
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
            capture_output=True, text=True, timeout=10, check=False)
        if result.returncode != 0:
            return False
        value = json.loads(result.stdout.strip())
        return value.get('restricted') is True
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return False


def _credentials_are_ignored(directory: Path, paths: tuple[Path, ...]) -> bool:
    """Prevent credential files from being written to a trackable repository path."""
    repository = Path(__file__).resolve().parents[1]
    try:
        relative = directory.resolve().relative_to(repository)
    except ValueError:
        return True
    try:
        for name in (path.name for path in paths):
            candidate = (relative / name).as_posix()
            result = subprocess.run(
                ['git', 'check-ignore', '--quiet', '--', candidate], cwd=repository,
                capture_output=True, timeout=5, check=False)
            if result.returncode != 0:
                return False
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _write_new_files(directory: Path, output: Path, token_file: Path, principals: list[dict], tokens: dict):
    created = []
    try:
        with output.open('x', encoding='utf-8') as stream:
            created.append(output)
            json.dump({'principals': principals}, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        with token_file.open('x', encoding='utf-8') as stream:
            created.append(token_file)
            json.dump(tokens, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        if not all(_windows_acl_is_private(path) for path in (directory, output, token_file)):
            raise PermissionError('Private credential ACL verification failed.')
    except Exception as original_error:
        cleanup_failed = False
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                cleanup_failed = True
        if cleanup_failed:
            raise RuntimeError('Credential cleanup could not be verified.') from original_error
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('.access'))
    parser.add_argument('--context-test', action='store_true',
                        help='provision only the local runtime-test context_tester principal')
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().absolute()
    output = directory / 'access.json'
    token_file = directory / 'tokens.json'
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        parser.error('Credential directory could not be created.')
    if not directory.is_dir() or directory.is_symlink():
        parser.error('Credential path must be a regular directory.')
    if output.exists() or output.is_symlink() or token_file.exists() or token_file.is_symlink():
        parser.error('Credentials already exist; refusing to overwrite either credential file.')
    if not _windows_acl_is_private(directory):
        parser.error('Credential directory ACL is not verifiably restricted to the current account, SYSTEM and Administrators.')
    if not _credentials_are_ignored(directory, (output, token_file)):
        parser.error('Credential paths are inside the repository but are not ignored by Git.')

    names = (('runtime-test', 'context_tester'),) if args.context_test else (
        ('local-operator', 'reviewer'), ('local-collector', 'collector'))
    tokens = {name: secrets.token_urlsafe(48) for name, _ in names}
    principals = [{'id': name, 'role': role,
                   'token_sha256': [hashlib.sha256(tokens[name].encode()).hexdigest()]}
                  for name, role in names]
    try:
        _write_new_files(directory, output, token_file, principals, tokens)
    except Exception as error:
        parser.error(f'Credential provisioning failed ({type(error).__name__}); inspect the isolated directory before retrying.')
    if args.context_test:
        print('Local runtime-test context credentials provisioned; token value omitted.')
    else:
        print(f'Local credentials provisioned in {directory}; token values omitted.')


if __name__ == '__main__':
    main()
