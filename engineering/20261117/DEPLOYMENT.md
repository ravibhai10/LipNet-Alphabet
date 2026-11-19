# Local deployment and reproducibility

The supported demonstration is the preserved eight-word reference, CPU-only, one backend process on Windows localhost. Alphabet and LRW500 are experimental; twenty-word serving and legacy research training require explicit opt-in. Sentence recognition remains unavailable. No LAN binding is enabled by default.

## Credentials and startup

The supported production entry point is `start_server.bat`, run from the repository directory in the authorized Windows operator account. It selects the reference runtime when available and invokes `scripts/run_backend.py`, which takes an exclusive lock in the private store and starts Uvicorn with exactly one worker and reload disabled. App startup takes the same lock, so direct Uvicorn launches also fail if another backend owns it. The server binds only to `127.0.0.1:8000`.

From the repository directory, using a working Python environment:

```powershell
python scripts/configure_local_access.py
$env:LIPNET_ACCESS_FILE = "$PWD\.access\access.json"
$env:LIPNET_CPU_THREADS = '1'
$env:LIPNET_WORD_BACKEND = 'avhubert'
$env:LIPNET_AVHUBERT_WORD_HEAD = "$PWD\runs\word_accuracy\avhubert_miracl8_v1\head.pt"
.\start_server.bat
```

The configuration command refuses to overwrite existing credentials. `access.json` stores principal IDs, roles and digests; `tokens.json` contains the private tokens. Restrict the directory to the operator's Windows account. Keep both out of source control, backups intended for publication and Docker build context. Rotate token digests under the existing principal ID. Do not regenerate credentials at startup.

The private directory is ACL-restricted to the authorized operator account, SYSTEM and Administrators; verify these permissions after restoring or moving the store. The database and any SQLite WAL/SHM files, research jobs and backups inherit private permissions. Do not place backups outside the private directory unless their destination has equivalent access controls.

## Isolated context runtime-test identity

`runtime-test` is an optional local testing principal with role `context_tester`. It is not a verified human identity or webcam participant. It can call only `POST /api/context/interpret`, `POST /api/context/sessions`, the session consent grant/revoke routes, and session termination. The read-only `GET /api/context/health` remains public. It cannot submit context feedback or use collector, reviewer, adaptive, recording, participant-administration, or model-administration operations. Existing `local-collector` and `local-operator` behavior is unchanged.

| Route / route family | `context_tester` | Notes |
|---|---:|---|
| `GET /api/context/health` | Public | Read-only configuration status; no credential required. |
| `POST /api/context/interpret` | Allow | Requires owned active session and current purpose-specific consent. |
| `POST /api/context/sessions` | Allow | Creates a session owned by the configured principal; no participant ID is supplied by the test client. |
| `POST /api/context/sessions/consent` | Allow | Grant only the declared interpretation purpose and notice version. |
| `POST /api/context/sessions/consent/revoke` | Allow | Revoke consent for the caller-owned session. |
| `POST /api/context/sessions/terminate` | Allow | Terminate the caller-owned session. |
| `POST /api/context/feedback` | Deny | Human-reported feedback is outside the test identity's session/consent scope. |
| Adaptive, capture, recording, studio, participant, training, dataset, reviewer, model administration, and other protected routes | Deny | Middleware enforces the exact context-route allowlist before parsing protected request bodies. |

Existing collector and reviewer principals keep their current authorization behavior; the new role has no broad-role inheritance or numeric privilege ordering.

Provision this identity only into a new, separate directory. Do not point this procedure at an existing production credential directory. From an elevated or otherwise authorized PowerShell session, create the directory under the already protected private store, discover the operator account rather than assuming a username, and apply the restricted ACL before provisioning:

```powershell
$runtimeAccess = Join-Path $PWD 'data\adaptive_private\runtime_test_access'
New-Item -ItemType Directory -Path $runtimeAccess
$operatorAccount = (whoami).Trim()
icacls $runtimeAccess /inheritance:r
icacls $runtimeAccess /grant:r "$($operatorAccount):(OI)(CI)F" '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F'
icacls $runtimeAccess
python scripts/configure_local_access.py --context-test --directory $runtimeAccess
$env:LIPNET_ACCESS_FILE = Join-Path $runtimeAccess 'access.json'
```

The provisioning command fails closed unless it can verify that the directory and resulting credential files allow access only to the current Windows account, SYSTEM and Administrators. It also refuses to overwrite either credential file and checks that in-repository credential paths are Git-ignored. `access.json` contains the `runtime-test` principal and token SHA-256 digest; the raw random Bearer token is kept only in the private `tokens.json`. The command never prints the token. Keep the directory out of source control and published build contexts; `.access/` and `data/adaptive_private/` are ignored by the repository rules.

To verify the provisioned identity without displaying its token, run a local check that reads the token in memory and prints only the server-derived actor and role:

```powershell
python -c "import json,os; from app.security import access; p=os.environ['LIPNET_ACCESS_FILE']; t=json.load(open(os.path.join(os.path.dirname(p),'tokens.json'),encoding='utf-8'))['runtime-test']; print(access('Bearer '+t))"
```

For the supported runtime, keep the optional LLM disabled and launch with `start_server.bat`. The token must be sent as an Authorization Bearer header by the local test client; never put it in a URL, source file, command argument or log. To revoke the local test credential, first terminate any test session, stop the backend, unset `LIPNET_ACCESS_FILE`, and remove only `access.json` and `tokens.json` from the dedicated `runtime_test_access` directory. Do not remove the private database or other research data. The context test role is a permission boundary, not a substitute for distinct human accounts when credentials are shared.

Validate actual health before demonstration. A present checkpoint is insufficient: the publisher encoder source, trusted encoder hash, compatible runtime, ordered vocabulary and head-bound evaluation/rejection policy are all required. Alternate research launchers also acquire the process lock, but their ports and experimental model modes are not the supported production profile.

## Shared validated settings

| Environment variable | Default | Meaning |
|---|---|---|
| `LIPNET_RETENTION_DAYS` | 90 | Successful recording retention; 1–3650 days |
| `LIPNET_MAINTENANCE_SECONDS` | 60 | Cleanup/retry interval; 1–3600 seconds |
| `LIPNET_PRIVATE_QUOTA_BYTES` | 2147483648 | Adaptive media plus managed files/derivatives admission budget |
| `LIPNET_PRIVATE_ROOT` | `data/adaptive_private` | Dedicated private directory; explicit values must be inside repository data and outside public corpus roots |
| `LIPNET_INFERENCE_TIMEOUT_SECONDS` | 30 | Ordinary inference response deadline; native kernels retain their lock until completion |
| `LIPNET_CPU_THREADS` | 1 | Configured CPU threads, validated 1–32 and bounded by available host CPUs |
| `LIPNET_ENABLE_SERVER_CAMERA` | false | Server camera is opt-in; a compatible trained representation bundle is also required |
| `LIPNET_REGRESSION_PYTHON` | current interpreter | Interpreter for candidate regression gate |

The session fence, interpretation cache and maintenance lifecycle are process-local. The supported deployment is one backend process with `workers=1` and reload disabled; multiworker and multi-replica deployments are unsupported. Docker files express one backend container and use the same private-store lock, but Docker has not been executed or verified here. Do not scale the service or bypass the supported launcher and assume process-local state is shared.

Maintenance starts with the application, reconstructs durable pending cleanup and marks interrupted jobs honestly. Source access is denied at expiry even before scheduled deletion. Deletion errors remain visible as retries. The storage report distinguishes logical BLOB bytes from SQLite file bytes; deletion does not promise filesystem secure erasure or instantaneous database shrinking.

Managed legacy collection bundles expire through registered receipts. Existing legacy bundles, studio media and camera exports need a non-destructive ownership/consent migration before reuse; do not infer permission from a file's presence. Eligible backups must be explicitly registered through reviewer-only `POST /api/adaptive/retention/backups`, reside inside the private `backups` directory and reference an owned source capture. They inherit source expiry and withdrawal. Unregistered external backups and offline browser copies require separate operator/client deletion.

## Executed environment checks

The `.venv-word-recognition` environment passed `pip check` and executes the backend software suite. The existing `.venv-avhubert-cpu` environment executed real reference inference, but fails dependency consistency checks:

- sacrebleu lacks lxml and tabulate.
- The installed Fairseq revision declares older hydra-core and OmegaConf requirements than the installed versions.
- Installed jax, jaxlib and ml-dtypes require NumPy 2 while the encoder environment has NumPy 1.26.4.

No clean encoder install is claimed. Existing environment snapshots are preserved in `.qa/engineering_resume_20261117/*-environment.txt`. Resolve these conflicts in a separate fresh environment; preserve the existing working environment, pin the exact upstream/Fairseq revisions and compatibility patches, then rerun `pip check`, the full suite and actual selected-model smoke. A requirements listing or dry-run resolver is not clean-install verification.

The frontend passed an actual `npm ci --ignore-scripts` from its lockfile, TypeScript/production build and Node suites at the recorded source state. Frontend-owned files continue to change independently; rerun acceptance against the final coordinated frontend revision.

## Docker procedure: not executed

Docker is unavailable on this machine. Packaging was corrected to include `pipeline` and `src`, avoid copying excluded `runs`, exclude private stores/environments/credentials, bind published ports to loopback and mount credentials separately. The image does not contain protected checkpoint binaries; models and reference runs must be supplied as mounts.

On a Docker-enabled host, create the private writable directories with ownership compatible with container UID 10001, provide credentials and the official licensed source/runtime, then execute:

```powershell
docker compose build --no-cache
docker compose up -d
docker compose ps
curl.exe http://127.0.0.1:8000/api/health
```

Verify unauthorized writes fail, authorized collection/SQLite writes succeed in the private mount, protected models remain read-only, the encoder loads in the image, Nginx handles bounded uploads, and expiry/restart/withdrawal tests pass. The current generic Docker dependency installation does not establish a reproducible Fairseq reference runtime; container readiness remains blocked until this procedure succeeds. Do not claim container verification from YAML inspection.
