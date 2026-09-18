$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
. "$PSScriptRoot\activate.ps1"
$env:PYTHONUNBUFFERED = "1"

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "outputs\logs"
$StatePath = Join-Path $LogDir "production_state.json"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path "configs\generated" | Out-Null
New-Item -ItemType Directory -Force -Path "reports\generated" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\evaluation" | Out-Null

function Write-State([string]$Stage, [string]$Status, [string]$Detail = "") {
    [ordered]@{
        timestamp = (Get-Date).ToString("o")
        stage = $Stage
        status = $Status
        detail = $Detail
        process_id = $PID
    } | ConvertTo-Json | Set-Content -Encoding UTF8 $StatePath
}

function Test-Complete([string]$OutputDir) {
    $Metadata = Join-Path $OutputDir "run_metadata.json"
    if (-not (Test-Path -LiteralPath $Metadata)) { return $false }
    return ((Get-Content -LiteralPath $Metadata -Raw | ConvertFrom-Json).status -eq "complete")
}

function Invoke-Training([string]$Stage, [string]$Config, [string]$OutputDir) {
    if (Test-Complete $OutputDir) {
        Write-State $Stage "skipped" "A completed run already exists."
        return
    }
    Write-State $Stage "running" $Config
    $LogPath = Join-Path $LogDir "$Stage.log"
    $TrainStage = $Stage.Split("-")[0]
    # Windows PowerShell 5 wraps any native stderr line as an ErrorRecord when
    # output is merged into the log. Transformers legitimately writes progress
    # and warnings to stderr, so judge native success by its exit code instead.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -m classical_llm.cli train $TrainStage --config $Config *>&1 |
        Tee-Object -FilePath $LogPath
    $TrainingExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($TrainingExitCode -ne 0) {
        Write-State $Stage "failed" "Exit code $TrainingExitCode; see $LogPath"
        throw "Training stage $Stage failed with exit code $TrainingExitCode"
    }
    Write-State $Stage "complete" $OutputDir
}

try {
    Invoke-Training "cpt-05" "configs/experiments/cpt_05_local.yaml" "outputs/cpt/qwen2.5-0.5b-classical05"
    Invoke-Training "cpt-20" "configs/experiments/cpt_local.yaml" "outputs/cpt/qwen2.5-0.5b-classical20"
    Invoke-Training "cpt-50" "configs/experiments/cpt_50_local.yaml" "outputs/cpt/qwen2.5-0.5b-classical50"

    $Candidates = @(
        [ordered]@{ name = "classical_05"; output = "outputs/cpt/qwen2.5-0.5b-classical05" },
        [ordered]@{ name = "classical_20"; output = "outputs/cpt/qwen2.5-0.5b-classical20" },
        [ordered]@{ name = "classical_50"; output = "outputs/cpt/qwen2.5-0.5b-classical50" }
    )
    $Scores = foreach ($Candidate in $Candidates) {
        $State = Get-Content -LiteralPath (Join-Path $Candidate.output "trainer_state.json") -Raw |
            ConvertFrom-Json
        [pscustomobject]@{
            name = $Candidate.name
            output = $Candidate.output
            best_eval_loss = [double]$State.best_metric
            best_global_step = [int]$State.best_global_step
        }
    }
    $Selected = $Scores | Sort-Object best_eval_loss | Select-Object -First 1
    [ordered]@{
        criterion = "minimum fixed-validation eval_loss"
        selected = $Selected
        candidates = @($Scores)
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 "reports/generated/cpt_selection.json"

    $SftTemplate = Get-Content -LiteralPath "configs/experiments/sft_local.yaml" -Raw
    $SelectedModel = ($Selected.output -replace "\\", "/") + "/best"
    $SftConfig = [regex]::Replace(
        $SftTemplate,
        '(?m)^model_name_or_path:.*$',
        "model_name_or_path: $SelectedModel"
    )
    $SftConfig | Set-Content -Encoding UTF8 "configs/generated/sft_selected.yaml"

    Invoke-Training "sft" "configs/generated/sft_selected.yaml" "outputs/sft/qwen2.5-0.5b-classical"
    Invoke-Training "reward" "configs/experiments/reward_local.yaml" "outputs/reward/qwen2.5-0.5b-classical-rm"
    Invoke-Training "dpo" "configs/experiments/dpo_local.yaml" "outputs/dpo/qwen2.5-0.5b-classical"
    Invoke-Training "grpo" "configs/experiments/grpo_local.yaml" "outputs/grpo/qwen2.5-0.5b-classical"

    Write-State "evaluation" "running" "Generating 5 x 400 domain responses and perplexities."
    $Models = [ordered]@{
        base = ".cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
        cpt = "$SelectedModel"
        sft = "outputs/sft/qwen2.5-0.5b-classical/best"
        dpo = "outputs/dpo/qwen2.5-0.5b-classical/best"
        grpo = "outputs/grpo/qwen2.5-0.5b-classical/best"
    }
    $GenerationMap = [ordered]@{}
    foreach ($Entry in $Models.GetEnumerator()) {
        $Generation = "outputs/evaluation/$($Entry.Key).jsonl"
        $Scored = "outputs/evaluation/$($Entry.Key).scored.jsonl"
        if (-not (Test-Path -LiteralPath $Generation)) {
            & $Python -m classical_llm.cli generate --model $Entry.Value `
                --dataset data/final/evaluation/domain_400.jsonl `
                --output $Generation --max-new-tokens 384
        }
        & $Python -m classical_llm.cli score --input $Generation --output $Scored
        $GenerationMap[$Entry.Key] = $Generation
        foreach ($Category in @("classical", "general_zh", "general_en")) {
            $PplOutput = "outputs/evaluation/$($Entry.Key).$Category.ppl.json"
            & $Python -m classical_llm.cli perplexity --model $Entry.Value `
                --dataset "data/final/evaluation/perplexity/$Category.jsonl" `
                --output $PplOutput --max-length 1024 --max-documents 200
        }
    }
    $GenerationJson = $GenerationMap | ConvertTo-Json -Compress
    & $Python -m classical_llm.cli blind-review --generations $GenerationJson `
        --output outputs/evaluation/blinded_review.jsonl
    & $Python -m classical_llm.cli report --output reports/generated/results.json
    Write-State "pipeline" "complete" "Training and automatic evaluation complete; human blind scores remain."
}
catch {
    Write-State "pipeline" "failed" $_.Exception.Message
    throw
}
