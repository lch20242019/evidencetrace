<#
Calibrate the two-threshold selector using the audited train-only NLI cache.
The fixed Python environment supplies NumPy/scikit-learn; no GPU inference is run.
Example: .\scripts\run_scifact_sentence_selector_v2.ps1 -OutputDirectory scifact_sentence_selector_train_dual_threshold_v1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$PythonExecutable = Join-Path $WorkspaceRoot 'env\scifact-verisci-py39-gpu\Scripts\python.exe'
$SelectorScript = Join-Path $PSScriptRoot 'calibrate_scifact_sentence_selector_v2.py'

if ($OutputDirectory -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$' -or
    $OutputDirectory -notmatch '(^|[_.-])train([_.-]|$)' -or
    $OutputDirectory -match '(^|[_.-])(dev|test)([_.-]|$)' -or
    $OutputDirectory.EndsWith('.')) {
    throw 'OutputDirectory must be a new train-only basename, not a path.'
}
$OutputPath = Join-Path (Join-Path $ProjectRoot 'eval_runs') $OutputDirectory
if (Test-Path -LiteralPath $OutputPath) {
    throw "Exclusive output already exists: $OutputPath"
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Pinned calibration interpreter is missing: $PythonExecutable"
}
if (-not (Test-Path -LiteralPath $SelectorScript -PathType Leaf)) {
    throw "Two-threshold selector script is missing: $SelectorScript"
}

Write-Host '[selector-v2] Verifying frozen train inputs and cached NLI features before calibration.'
& $PythonExecutable -u -X faulthandler $SelectorScript --out $OutputPath
$SelectorExitCode = $LASTEXITCODE
Write-Host "[selector-v2] Exit code: $SelectorExitCode. Output: $OutputPath"
exit $SelectorExitCode
