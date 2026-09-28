<#
Run the unchanged train selector using the pinned project GPU environment.
The runtime audit is a sibling of the exclusive evaluation directory.
This entry point does not change DLL search paths or repair system runtimes.
Example: .\scripts\run_scifact_sentence_selector.ps1 -OutputDirectory scifact_sentence_selector_train_oof_v4
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $OutputDirectory,
    [switch] $PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$PythonExecutable = Join-Path $WorkspaceRoot 'env\scifact-verisci-py39-gpu\Scripts\python.exe'
$PreflightScript = Join-Path $PSScriptRoot 'check_scifact_gpu_runtime.py'
$SelectorScript = Join-Path $PSScriptRoot 'calibrate_scifact_sentence_selector.py'
$ExpectedSelectorHash = '9D11DCC3575AF86C33055B941CFAEFB29B2D5CA287E09A15662A725A6FFE848A'

if ($OutputDirectory -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$' -or
    $OutputDirectory -notmatch '(^|[_.-])train([_.-]|$)' -or
    $OutputDirectory -match '(^|[_.-])(dev|test)([_.-]|$)' -or
    $OutputDirectory.EndsWith('.')) {
    throw 'OutputDirectory must be a train-only basename, not a path.'
}
$OutputPath = Join-Path (Join-Path $ProjectRoot 'eval_runs') $OutputDirectory
$AuditPath = $OutputPath + '.runtime.json'
$NativeLogPath = $OutputPath + '.runtime.native.log'
foreach ($PathToReserve in @($OutputPath, $AuditPath, $NativeLogPath)) {
    if (Test-Path -LiteralPath $PathToReserve) {
        throw "Exclusive output already exists: $PathToReserve"
    }
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Pinned GPU interpreter is missing: $PythonExecutable"
}

function Save-ProcessExit([string] $Phase, [int] $ExitCode) {
    if (Test-Path -LiteralPath $AuditPath -PathType Leaf) {
        $Audit = Get-Content -LiteralPath $AuditPath -Raw | ConvertFrom-Json
        $Audit | Add-Member -NotePropertyName ($Phase + '_process_exit_code') -NotePropertyValue $ExitCode -Force
        if ($Phase -eq 'preflight' -and $ExitCode -ne 0 -and $Audit.status -eq 'running') {
            $Audit.status = 'native_process_failure'
        }
        $Audit | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $AuditPath -Encoding UTF8
    }
}

Write-Host '[selector] Checking the fixed GPU runtime before opening an evaluation directory.'
& $PythonExecutable -u $PreflightScript --output-directory $OutputDirectory
$PreflightExitCode = $LASTEXITCODE
Save-ProcessExit 'preflight' $PreflightExitCode
if ($PreflightExitCode -ne 0) {
    throw "GPU runtime preflight failed (exit $PreflightExitCode). See $AuditPath"
}
$RuntimeAudit = Get-Content -LiteralPath $AuditPath -Raw | ConvertFrom-Json
if ($RuntimeAudit.status -ne 'passed' -or $RuntimeAudit.output_directory -ne $OutputPath) {
    throw 'The runtime audit did not pass for this output directory.'
}
if ($PreflightOnly) {
    Write-Host "[selector] Preflight passed. PreflightOnly: selector was not started. Audit: $AuditPath"
    exit 0
}
if ((Get-FileHash -LiteralPath $SelectorScript -Algorithm SHA256).Hash -ne $ExpectedSelectorHash) {
    throw 'The audited selector source changed; refusing to run.'
}
$SelectorArguments = @(
    '--corpus', (Join-Path $WorkspaceRoot 'env\datasets\scifact\raw\data\corpus.jsonl'),
    '--claims-train', (Join-Path $WorkspaceRoot 'env\datasets\scifact\raw\data\claims_train.jsonl'),
    '--retrieval-train', (Join-Path $ProjectRoot 'eval_runs\scifact_official_tfidf_train\abstract_retrieval.jsonl'),
    '--retrieval-manifest', (Join-Path $ProjectRoot 'eval_runs\scifact_official_tfidf_train\run_manifest.json'),
    '--model', (Join-Path $WorkspaceRoot 'env\models\MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli\6f5cf0a2b59cabb106aca4c287eed12e357e90eb'),
    '--out', $OutputPath,
    '--device', 'cuda'
)
$RuntimeAudit | Add-Member -NotePropertyName 'selector_arguments' -NotePropertyValue $SelectorArguments -Force
$RuntimeAudit | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $AuditPath -Encoding UTF8
Write-Host '[selector] Preflight passed; starting the unchanged official-train calibration.'
# A fresh process keeps the synthetic probe separate. SetErrorMode only suppresses
# Windows fault dialogs in this process; it does not alter DLL loading.
$Bootstrap = "import ctypes, runpy, sys; ctypes.windll.kernel32.SetErrorMode(0x8003); runpy.run_path(sys.argv.pop(1), run_name='__main__')"
& $PythonExecutable -u -X faulthandler -c $Bootstrap $SelectorScript @SelectorArguments
$SelectorExitCode = $LASTEXITCODE
Save-ProcessExit 'selector' $SelectorExitCode
Write-Host "[selector] Process exited with code $SelectorExitCode. Runtime audit: $AuditPath"
exit $SelectorExitCode
