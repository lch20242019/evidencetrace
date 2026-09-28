<# Run the train-only role ablation in the pinned local GPU environment. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $OutputDirectory,
    [string] $FrozenDirectory = 'scifact_selector_train_oof_frozen_v1'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$EvalRoot = Join-Path $ProjectRoot 'eval_runs'
$PythonExecutable = Join-Path $WorkspaceRoot 'env\scifact-verisci-py39-gpu\Scripts\python.exe'
foreach ($RunName in @($OutputDirectory, $FrozenDirectory)) {
    if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$' -or
        $RunName -notmatch '(^|[_.-])train([_.-]|$)' -or
        $RunName -match '(^|[_.-])(dev|test)([_.-]|$)' -or
        $RunName.EndsWith('.')) {
        throw 'Run names must be train-only basenames, not paths.'
    }
}
$OutputPath = Join-Path $EvalRoot $OutputDirectory
$FrozenPath = Join-Path $EvalRoot $FrozenDirectory
if (Test-Path -LiteralPath $OutputPath) { throw "Exclusive output already exists: $OutputPath" }
if (-not (Test-Path -LiteralPath (Join-Path $FrozenPath 'freeze.json') -PathType Leaf)) {
    throw 'Verified upstream freeze is missing.'
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw 'Pinned Python interpreter is missing.'
}
& $PythonExecutable -B -u -X faulthandler (Join-Path $PSScriptRoot 'run_scifact_frozen_evidence_ab.py') --frozen $FrozenPath --out $OutputPath
$RunExitCode = $LASTEXITCODE
Write-Host "[role-ablation] Exit code: $RunExitCode. Output: $OutputPath"
exit $RunExitCode
