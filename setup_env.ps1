<#
.SYNOPSIS
    Build the SCULPTOR environment: fetch the exact CPython version the
    package was produced with, create .venv from it and install every
    dependency.

.DESCRIPTION
    Reproducibility starts with the interpreter.  The package was built and
    validated with CPython 3.11.9, so this script does not accept "whatever
    Python happens to be on PATH": it looks for 3.11.9, and if it is not on the
    machine it downloads the official 3.11.9 Windows build (the nuget.org
    "python" package, unpacked into runtime\python311 inside the package - no
    installer, no administrator rights, no registry changes, no PATH edits) and
    builds .venv from that interpreter.

    Every step checks its exit code, so the script never reports success after a
    failed download or install.

.PARAMETER Python
    Use an interpreter that is already installed, for example
    -Python "C:\Program Files\Python311\python.exe".  The version is still
    checked and reported.

.PARAMETER PythonVersion
    Version to provision when no suitable interpreter exists (default 3.11.9).

.PARAMETER NoDownload
    Never download an interpreter; stop instead.

.PARAMETER Check
    Report what would be used or downloaded, then stop without changing
    anything.

.PARAMETER Recreate
    Delete the existing .venv in this folder and build it again.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Cuda
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Check
#>
param(
    [string]$Python = "",
    [string]$PythonVersion = "3.11.9",
    [switch]$Cuda,
    [switch]$Lock,
    [switch]$NoDownload,
    [switch]$Check,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$Runtime = Join-Path $Root "runtime"
$RuntimePython = Join-Path $Runtime "python311"
$RuntimeExe = Join-Path $RuntimePython "tools\python.exe"
$RuntimeNote = Join-Path $Runtime "PYTHON_VERSION.txt"
$Cache = Join-Path $Runtime "_cache"
$Requirements = if ($Lock) {
    Join-Path $Root "requirements.lock.txt"
} else {
    Join-Path $Root "requirements.txt"
}
$majorMinor = ($PythonVersion -split '\.')[0..1] -join '.'

# Every runtime dependency, checked by importing it.  A machine that lost the
# network half way through the install ends up with a venv that exists but is
# incomplete; this list is what turns that state into something the script can
# detect and repair instead of a dead end.
#
# "core" is what the pipeline itself imports: without it nothing runs.
# "gui" is only the native control-panel window; the browser mode uses the
# standard library, so a machine that cannot install these still runs
# everything - the script warns instead of stopping.
$CoreModules = @(
    @{ Module = "numpy";           Package = "numpy" }
    @{ Module = "pandas";          Package = "pandas" }
    @{ Module = "scipy";           Package = "scipy" }
    @{ Module = "sklearn";         Package = "scikit-learn" }
    @{ Module = "joblib";          Package = "joblib" }
    @{ Module = "rdkit";           Package = "rdkit" }
    @{ Module = "torch";           Package = "torch" }
    @{ Module = "torch_geometric"; Package = "torch-geometric" }
    @{ Module = "e3nn";            Package = "e3nn" }
    @{ Module = "xgboost";         Package = "xgboost" }
    @{ Module = "h5py";            Package = "h5py" }
    @{ Module = "openpyxl";        Package = "openpyxl" }
    @{ Module = "psutil";          Package = "psutil" }
)

$GuiModules = @(
    @{ Module = "webview";         Package = "pywebview" }
    @{ Module = "proxy_tools";     Package = "proxy_tools" }
    @{ Module = "bottle";          Package = "bottle" }
)

$RequiredModules = $CoreModules + $GuiModules

function Stop-WithMessage {
    param([string]$Message)
    Write-Host ""
    Write-Host "SETUP STOPPED" -ForegroundColor Red
    Write-Host $Message
    Write-Host ""
    exit 1
}

function Get-InterpreterInfo {
    param([string]$Exe, [string[]]$Prefix = @())
    try {
        $text = & $Exe @Prefix -c "import sys,platform;print('%d.%d.%d'%sys.version_info[:3]);print(platform.architecture()[0]);print(sys.executable)" 2>$null
    } catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0 -or -not $text) { return $null }
    $lines = @($text | Where-Object { $_ -ne "" })
    if ($lines.Count -lt 3) { return $null }
    return @{
        Version = $lines[0]
        Bitness = $lines[1]
        Path    = $lines[2]
        Home    = Split-Path -Parent $lines[2]
    }
}

function Test-Environment {
    param([string]$Exe, [string]$Group = "all")
    # Returns the list of modules that fail to import; empty means usable.
    if (-not (Test-Path -LiteralPath $Exe)) { return @("__interpreter__") }
    $table = switch ($Group) {
        "core" { $CoreModules }
        "gui"  { $GuiModules }
        default { $RequiredModules }
    }
    $names = ($table | ForEach-Object { $_.Module }) -join ","
    $script = "import importlib,sys`nmissing=[]`n" +
              "for name in '$names'.split(','):`n" +
              "    try:`n" +
              "        importlib.import_module(name)`n" +
              "    except Exception:`n" +
              "        missing.append(name)`n" +
              "print('MISSING:' + ','.join(missing))`n"
    # The imports write warnings to stderr (torch does).  With
    # $ErrorActionPreference = "Stop" PowerShell would turn that into a
    # terminating error and the check would look like a broken interpreter, so
    # stderr is ignored for the duration of the call.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Exe -c $script 2>$null
        $code = $LASTEXITCODE
    } catch {
        return @("__interpreter__")
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0 -or -not $out) { return @("__interpreter__") }
    $line = @($out | Where-Object { $_ -like "MISSING:*" }) | Select-Object -Last 1
    if (-not $line) { return @("__interpreter__") }
    $missing = ($line -replace '^MISSING:', '').Trim()
    if (-not $missing) { return @() }
    return @($missing -split ',')
}

function Get-ModuleDiagnostics {
    param([string]$Exe, [string[]]$Modules)
    # Says whether each module is missing entirely or fails to import, and why.
    if (-not $Modules -or $Modules.Count -eq 0) { return }
    $list = $Modules -join ","
    $script = "import importlib,importlib.metadata as md,traceback`n" +
              "for name in '$list'.split(','):`n" +
              "    try:`n" +
              "        dist = md.version(name)`n" +
              "    except Exception:`n" +
              "        dist = None`n" +
              "    if dist is None:`n" +
              "        for guess in ('pywebview' if name=='webview' else name,):`n" +
              "            try:`n" +
              "                dist = md.version(guess); break`n" +
              "            except Exception:`n" +
              "                dist = None`n" +
              "    if dist is None:`n" +
              "        print('  ' + name + ': NOT INSTALLED')`n" +
              "        continue`n" +
              "    try:`n" +
              "        importlib.import_module(name)`n" +
              "        print('  ' + name + ': ok in a fresh process')`n" +
              "    except Exception as exc:`n" +
              "        print('  ' + name + ' ' + dist + ': ' + type(exc).__name__ + ': ' + str(exc))`n"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe -c $script 2>$null | ForEach-Object { Write-Host $_ }
    } catch {
        Write-Host "  (diagnostics could not run)"
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Install-PythonPackages {
    param([string]$Exe, [string[]]$Packages, [string]$LogPath)
    # Tries the configured index, then adds PyPI, then PyPI alone.
    $attempts = @(
        @{ Name = "the configured index"; Args = @() }
        @{ Name = "the configured index + PyPI";
           Args = @("--extra-index-url", "https://pypi.org/simple") }
        @{ Name = "PyPI only"; Args = @("--index-url", "https://pypi.org/simple") }
    )
    foreach ($attempt in $attempts) {
        Write-Host "      installing from $($attempt.Name) ..."
        $arguments = @("-m", "pip", "install", "--retries", "5", "--timeout", "60") +
                     $Packages + $attempt.Args
        $output = & $Exe @arguments 2>&1
        $code = $LASTEXITCODE
        $output | ForEach-Object { Write-Host "      $_" }
        if ($LogPath) {
            "=== $($attempt.Name) : $($Packages -join ' ') (exit $code)" |
                Add-Content -LiteralPath $LogPath -Encoding UTF8
            $output | Add-Content -LiteralPath $LogPath -Encoding UTF8
        }
        if ($code -eq 0) { return $true }
    }
    return $false
}

# Packages that are shipped inside the release because a mirror cannot be
# trusted to deliver them.  proxy_tools is the important one: some mirrors
# serve a wheel that contains only the dist-info, so pip reports success while
# "import webview" fails and re-running pip does nothing (the metadata says the
# requirement is already satisfied).
$Vendored = @(
    @{ Module = "proxy_tools"; Source = "tools\control_panel\vendor\proxy_tools" }
)

function Repair-VendoredPackages {
    param([string]$Exe, [string]$SitePackages)
    $repaired = @()
    foreach ($item in $Vendored) {
        if (Test-PythonImport -Exe $Exe -Code "import $($item.Module)") { continue }
        $source = Join-Path $Root $item.Source
        if (-not (Test-Path -LiteralPath $source)) { continue }
        $target = Join-Path $SitePackages $item.Module
        Write-Host "      installing the bundled $($item.Module) (the mirror's copy is incomplete)" -ForegroundColor Yellow
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        Copy-Item -Path (Join-Path $source "*") -Destination $target -Recurse -Force
        if (Test-PythonImport -Exe $Exe -Code "import $($item.Module)") {
            $repaired += $item.Module
        }
    }
    return $repaired
}

# A python -c that may print a traceback on stderr must never abort the script:
# Windows PowerShell 5.1 turns native stderr into a terminating error when
# $ErrorActionPreference is Stop.
function Test-PythonImport {
    param([string]$Exe, [string]$Code)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe -c $Code 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Test-PackageArchive {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -lt 5MB) { return $false }
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null
        $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
        $count = $archive.Entries.Count
        $archive.Dispose()
        return ($count -gt 0)
    } catch {
        return $false
    }
}

function Invoke-PythonProvision {
    param([string]$Version)

    Write-Host "`n[1/5] Python $Version was not found - downloading the official build ..."
    Write-Host "      source : https://www.nuget.org/api/v2/package/python/$Version"
    New-Item -ItemType Directory -Force -Path $Cache | Out-Null
    $package = Join-Path $Cache "python-$Version.nupkg"

    if (Test-PackageArchive -Path $package) {
        Write-Host "      using the cached download"
    } else {
        if (Test-Path -LiteralPath $package) {
            Write-Host "      the cached download is incomplete or corrupt - fetching it again" -ForegroundColor Yellow
            Remove-Item -LiteralPath $package -Force
        }
        # Download to a temporary name and rename only after the archive has
        # been verified, so an interrupted transfer can never be mistaken for a
        # usable cache on the next run.
        $partial = "$package.part"
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        $ok = $false
        for ($attempt = 1; $attempt -le 3 -and -not $ok; $attempt++) {
            try {
                Write-Host "      download attempt $attempt of 3 ..."
                Invoke-WebRequest -Uri "https://www.nuget.org/api/v2/package/python/$Version" -OutFile $partial -UseBasicParsing -TimeoutSec 300
                if (Test-PackageArchive -Path $partial) {
                    Move-Item -LiteralPath $partial -Destination $package -Force
                    $ok = $true
                } else {
                    Write-Host "      the file arrived incomplete" -ForegroundColor Yellow
                    if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
                }
            } catch {
                Write-Host "      $($_.Exception.Message)" -ForegroundColor Yellow
                if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
            }
        }
        if (-not $ok) {
            Stop-WithMessage @"
Could not download Python $Version (three attempts).

Run the script again later - a partial download is cleaned up automatically, so
it is safe to retry.  Or fetch the package by hand and save it as:
    $package
    https://www.nuget.org/packages/python/$Version

Or install Python $majorMinor (64-bit) yourself and rerun with:
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Python "C:\Program Files\Python311\python.exe" -Recreate
"@
        }
    }

    $size = (Get-Item -LiteralPath $package).Length
    $hash = (Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash
    Write-Host "      $([math]::Round($size/1MB,1)) MB, SHA256 $hash"

    if (Test-Path -LiteralPath $RuntimePython) {
        Write-Host "      removing the incomplete runtime\python311 ..."
        Remove-Item -LiteralPath $RuntimePython -Recurse -Force
    }
    # Expand-Archive only accepts .zip, and the distribution is a .nupkg, so
    # the archive is unpacked through .NET directly.
    Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null
    try {
        [System.IO.Compression.ZipFile]::ExtractToDirectory($package, $RuntimePython)
    } catch {
        Stop-WithMessage "Could not unpack $package`n    $($_.Exception.Message)"
    }

    if (-not (Test-Path -LiteralPath $RuntimeExe)) {
        Stop-WithMessage "The package was unpacked but $RuntimeExe is missing."
    }
    $info = Get-InterpreterInfo -Exe $RuntimeExe
    if (-not $info) {
        Stop-WithMessage "The unpacked interpreter at $RuntimeExe does not start."
    }
    if ($info.Version -ne $Version) {
        Stop-WithMessage "Expected Python $Version but the unpacked interpreter reports $($info.Version)."
    }
    Write-Host "      provisioned Python $($info.Version) at $($info.Path)"
    return @{ Exe = $RuntimeExe; Prefix = @(); Label = "provisioned runtime\python311"
              Version = $info.Version; Path = $info.Path; Home = $info.Home
              Source = "downloaded python-$Version.nupkg"; Hash = $hash }
}

# ------------------------------------------------------------------- header
Write-Host "SCULPTOR environment setup"
Write-Host "  package root : $Root"
Write-Host "  environment  : $Venv"
Write-Host "  target python: CPython $PythonVersion (64-bit)"
Write-Host "  OS           : $([System.Environment]::OSVersion.VersionString) ($env:PROCESSOR_ARCHITECTURE)"

# ------------------------------------------------------- find the interpreter
$chosen = $null
$notes = @()

if ($Python) {
    $info = Get-InterpreterInfo -Exe $Python
    if (-not $info) { Stop-WithMessage "The interpreter given with -Python does not run:`n    $Python" }
    $chosen = @{ Exe = $Python; Prefix = @(); Label = "explicit -Python"
                 Version = $info.Version; Path = $info.Path; Home = $info.Home
                 Source = "supplied on the command line"; Hash = "" }
} else {
    if (Test-Path -LiteralPath $RuntimeExe) {
        $info = Get-InterpreterInfo -Exe $RuntimeExe
        if ($info -and $info.Version -eq $PythonVersion) {
            $chosen = @{ Exe = $RuntimeExe; Prefix = @(); Label = "runtime\python311"
                         Version = $info.Version; Path = $info.Path; Home = $info.Home
                         Source = "already provisioned in the package"; Hash = "" }
        } elseif ($info) {
            $notes += "runtime\python311 holds Python $($info.Version), expected $PythonVersion - re-provisioning"
        }
    }
}

if (-not $chosen) {
    foreach ($candidate in @(
        @{ Exe = "py";      Prefix = @("-$majorMinor"); Label = "py -$majorMinor" },
        @{ Exe = "python";  Prefix = @();               Label = "python (PATH)" }
    )) {
        $info = Get-InterpreterInfo -Exe $candidate.Exe -Prefix $candidate.Prefix
        if (-not $info) { continue }
        if ($info.Version -ne $PythonVersion) {
            $notes += "$($candidate.Label) is Python $($info.Version), expected $PythonVersion"
            continue
        }
        $chosen = @{ Exe = $candidate.Exe; Prefix = $candidate.Prefix
                     Label = $candidate.Label; Version = $info.Version
                     Path = $info.Path; Home = $info.Home
                     Source = "found on this machine"; Hash = "" }
        break
    }
}

Write-Host ""
foreach ($note in $notes) { Write-Host "  note         : $note" -ForegroundColor DarkGray }

if (-not $chosen) {
    if ($NoDownload) {
        Stop-WithMessage @"
Python $PythonVersion was not found and -NoDownload was given.

Install CPython $PythonVersion (64-bit):
    https://www.python.org/downloads/release/python-3119/
or let the script fetch it:
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
"@
    }
    if ($Check) {
        Write-Host "  interpreter  : would be downloaded (nuget.org python/$PythonVersion)"
        Write-Host ""
        Write-Host "-Check: nothing was created or installed."
        exit 0
    }
    $chosen = Invoke-PythonProvision -Version $PythonVersion
}

if ((Split-Path -Leaf $chosen.Home) -eq "tools" -and
    (Test-Path -LiteralPath (Join-Path $chosen.Home "python.exe"))) {
    $chosen.Home = $chosen.Home
}

Write-Host "  interpreter  : Python $($chosen.Version) 64-bit  [$($chosen.Label)]"
Write-Host "  python path  : $($chosen.Path)"
Write-Host "  source       : $($chosen.Source)"
Write-Host "  requirements : $(Split-Path $Requirements -Leaf)  (CUDA build: $Cuda)"

if ($Check) {
    Write-Host ""
    Write-Host "-Check: nothing was created or installed."
    exit 0
}

# --------------------------------------------------------- record the choice
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
@(
    "python_version = $($chosen.Version)"
    "python_exe     = $($chosen.Path)"
    "python_source  = $($chosen.Source)"
    ("package_sha256 = " + $chosen.Hash)
    "recorded_at    = $(Get-Date -Format s)"
    "package_root   = $Root"
) | Set-Content -LiteralPath $RuntimeNote -Encoding UTF8

# ------------------------------------------------------------------ the venv
if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    Write-Host "`n[1/5] removing the existing .venv (-Recreate) ..."
    Remove-Item -LiteralPath $Venv -Recurse -Force
}

$rebuild = $false
if (Test-Path -LiteralPath $VenvPython) {
    $config = Join-Path $Venv "pyvenv.cfg"
    # $home is not usable here: $HOME is a read-only automatic variable and
    # PowerShell variable names are case-insensitive.
    $venvHome = ""
    if (Test-Path -LiteralPath $config) {
        $line = Get-Content -LiteralPath $config | Where-Object { $_ -match '^\s*home\s*=' } | Select-Object -First 1
        if ($line) { $venvHome = ($line -split '=', 2)[1].Trim() }
    }
    if ($venvHome -and -not (Test-Path -LiteralPath $venvHome)) {
        Write-Host "`n[1/5] the existing .venv points at $venvHome, which no longer exists" -ForegroundColor Yellow
        Write-Host "      (the package was moved) - rebuilding it" -ForegroundColor Yellow
        $rebuild = $true
    } elseif ($venvHome -and ($venvHome -ne $chosen.Home)) {
        Write-Host "`n[1/5] the existing .venv was built from $venvHome, not $($chosen.Home)" -ForegroundColor Yellow
        Write-Host "      - rebuilding it" -ForegroundColor Yellow
        $rebuild = $true
    } else {
        $missing = Test-Environment -Exe $VenvPython
        if ($missing.Count -gt 0) {
            if ($missing[0] -eq "__interpreter__") {
                Write-Host "`n[1/5] the existing .venv cannot run - rebuilding it" -ForegroundColor Yellow
                $rebuild = $true
            } else {
                Write-Host "`n[1/5] the existing .venv is incomplete" -ForegroundColor Yellow
                Write-Host "      missing: $($missing -join ', ')" -ForegroundColor Yellow
                Write-Host "      finishing the installation (nothing is deleted)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "`n[1/5] .venv already exists and is complete, reusing it"
        }
    }
} else {
    $rebuild = $true
}

if ($rebuild) {
    if (Test-Path -LiteralPath $Venv) {
        Write-Host "      removing the unusable .venv ..."
        Remove-Item -LiteralPath $Venv -Recurse -Force
    }
    Write-Host "      creating .venv with Python $($chosen.Version) ..."
    & $chosen.Exe @($chosen.Prefix) -m venv $Venv
    if ($LASTEXITCODE -ne 0) {
        Stop-WithMessage "Could not create the virtual environment with $($chosen.Path)."
    }
}

Write-Host "`n[2/5] upgrading pip ..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "      warning: pip could not be upgraded, continuing with the bundled version"
}

Write-Host "`n[3/5] checking that every dependency has a wheel for Python $($chosen.Version) ..."
& $VenvPython -m pip install --dry-run --no-deps -r $Requirements 2>&1 | ForEach-Object { Write-Host "      $_" }
if ($LASTEXITCODE -ne 0) {
    Stop-WithMessage @"
At least one dependency has no build for this interpreter.
    interpreter : $($chosen.Path)  (Python $($chosen.Version))

The packages pip could not resolve are listed above.  Useful checks:
    .\.venv\Scripts\python.exe -VV
    .\.venv\Scripts\python.exe -m pip index versions rdkit
    .\.venv\Scripts\python.exe -m pip debug --verbose | Select-String 'cp3'
"@
}

if ($Cuda) {
    Write-Host "`n[4/5] installing the CUDA build of PyTorch ..."
    & $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "`n[4/5] installing the CPU build of PyTorch ..."
    & $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) {
    Stop-WithMessage "PyTorch could not be installed. Check the network connection to download.pytorch.org."
}

$pipLog = Join-Path $Runtime "pip_install.log"
Write-Host "`n[5/5] installing dependencies from $(Split-Path $Requirements -Leaf) ..."
Write-Host "      a full log is written to runtime\pip_install.log"
$installOutput = & $VenvPython -m pip install -r $Requirements --retries 5 --timeout 60 2>&1
$installCode = $LASTEXITCODE
$installOutput | ForEach-Object { Write-Host "      $_" }
"=== full requirements (exit $installCode)" | Add-Content -LiteralPath $pipLog -Encoding UTF8
$installOutput | Add-Content -LiteralPath $pipLog -Encoding UTF8
if ($installCode -ne 0) {
    Write-Host ""
    Write-Host "      the list did not install completely; retrying the missing packages one by one" -ForegroundColor Yellow
}

# Second pass: whatever is still missing is installed by name.  This isolates a
# package that a mirror cannot serve or that failed on a flaky connection, and
# it cannot drag the rest of the install down with it.
$missingCore = Test-Environment -Exe $VenvPython -Group "core"
if ($missingCore.Count -gt 0 -and $missingCore[0] -ne "__interpreter__") {
    $packages = @($CoreModules | Where-Object { $missingCore -contains $_.Module } |
                  ForEach-Object { $_.Package })
    Write-Host "`n      still missing: $($missingCore -join ', ')"
    [void](Install-PythonPackages -Exe $VenvPython -Packages $packages -LogPath $pipLog)
    $missingCore = Test-Environment -Exe $VenvPython -Group "core"
}

if ($missingCore.Count -gt 0 -and $missingCore[0] -eq "__interpreter__") {
    Stop-WithMessage "The interpreter in .venv cannot run.  Rerun the script - it will rebuild the environment."
}

if ($missingCore.Count -gt 0) {
    $config = & $VenvPython -m pip config list 2>$null
    Write-Host "`nCore dependency problem:" -ForegroundColor Yellow
    Get-ModuleDiagnostics -Exe $VenvPython -Modules $missingCore
    Stop-WithMessage @"
The core pipeline dependencies are still incomplete.
    missing     : $($missingCore -join ', ')
    interpreter : $($chosen.Path)  (Python $($chosen.Version))
    pip index   : $($config -join '; ')
    pip log     : $pipLog

The lines above say whether each package is absent or installed but failing to
import.  To install one of them by hand:
    .\.venv\Scripts\python.exe -m pip install <package> --index-url https://pypi.org/simple
"@
}

# The native panel window is optional: the browser mode needs no extra package.
# First give the bundled copies a chance: a mirror that ships an empty
# proxy_tools wheel cannot be repaired by pip.
$sitePackages = Join-Path $Venv "Lib\site-packages"
$repaired = Repair-VendoredPackages -Exe $VenvPython -SitePackages $sitePackages
if ($repaired.Count -gt 0) {
    Write-Host "      repaired from the package: $($repaired -join ', ')"
}

$missingGui = Test-Environment -Exe $VenvPython -Group "gui"
if ($missingGui.Count -gt 0) {
    $packages = @($GuiModules | Where-Object { $missingGui -contains $_.Module } |
                  ForEach-Object { $_.Package })
    Write-Host "`n      the native panel packages are missing: $($missingGui -join ', ')" -ForegroundColor Yellow
    [void](Install-PythonPackages -Exe $VenvPython -Packages $packages -LogPath $pipLog)
    $missingGui = Test-Environment -Exe $VenvPython -Group "gui"
}
if ($missingGui.Count -gt 0) {
    Write-Host ""
    Write-Host "      NOTE: the native window needs $($missingGui -join ', '), which this machine" -ForegroundColor Yellow
    Write-Host "            could not install.  Everything else works; start the panel in" -ForegroundColor Yellow
    Write-Host "            the browser instead:" -ForegroundColor Yellow
    Write-Host "              start_sculptor.bat web" -ForegroundColor Yellow
}

Write-Host "`nEnvironment summary"
& $VenvPython -c @"
import sys
print('  interpreter :', sys.executable)
print('  python      :', sys.version.split()[0])
"@
if ($LASTEXITCODE -ne 0) {
    Stop-WithMessage "The interpreter in .venv cannot run; rerun the script - it will rebuild the environment."
}

$missing = Test-Environment -Exe $VenvPython
if ($missing.Count -gt 0 -and $missing[0] -eq "__interpreter__") {
    Stop-WithMessage "The interpreter in .venv cannot import the dependencies; rerun the script - it will rebuild the environment."
}

& $VenvPython -c @"
import torch, rdkit, torch_geometric, numpy, pandas, e3nn, sklearn
print('  torch       :', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('  rdkit       :', rdkit.__version__)
print('  pyg         :', torch_geometric.__version__)
print('  numpy       :', numpy.__version__)
print('  pandas      :', pandas.__version__)
print('  e3nn        :', e3nn.__version__)
"@
if ($LASTEXITCODE -ne 0) {
    Stop-WithMessage "The environment was installed but the import check failed (see the traceback above)."
}

if ($missingGui.Count -eq 0) {
    Write-Host "  panel       : native window available"
} else {
    Write-Host "  panel       : browser mode only (start_sculptor.bat web)"
}

Write-Host "`nRecord written to runtime\PYTHON_VERSION.txt"
Write-Host "Done. Start the control panel with:"
Write-Host "  .\.venv\Scripts\python.exe tools\control_panel\server.py"
