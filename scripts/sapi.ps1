param([Parameter(Mandatory=$true)][string]$TextFile,
      [Parameter(Mandatory=$true)][string]$OutputFile)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $chinese = $voice.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'zh-CN' } | Select-Object -First 1
    if (-not $chinese) { throw 'No zh-CN Windows speech voice is installed. Import narration audio or configure a TTS API.' }
    $voice.SelectVoice($chinese.VoiceInfo.Name)
    $voice.SetOutputToWaveFile($OutputFile)
    $voice.Speak([System.IO.File]::ReadAllText($TextFile, [System.Text.Encoding]::UTF8))
} finally { $voice.Dispose() }
