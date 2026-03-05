# Build script for the C++ hybrid chess engine (Windows / MSYS2 ucrt64 g++)
# Usage: .\cpp\build.ps1

$ErrorActionPreference = "Stop"

$GXX = "C:\msys64\ucrt64\bin\g++.exe"
$SrcDir = "$PSScriptRoot\src"
$OutDir = "$PSScriptRoot\..\hybrid\cpp_engine"

# Get pybind11 includes and Python extension suffix
$PyIncludes = (python -m pybind11 --includes) -split ' '
$ExtSuffix = python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
$PythonDll = python -c "import sys, os; print(os.path.join(sys.prefix, 'python' + str(sys.version_info.major) + str(sys.version_info.minor) + '.dll'))"

$OutputFile = "$OutDir\hybrid_cpp_engine$ExtSuffix"

Write-Host "=== Building hybrid_cpp_engine ==="
Write-Host "  Compiler:   $GXX"
Write-Host "  Sources:    board.cpp, rules.cpp, ab_search.cpp, bindings.cpp"
Write-Host "  Output:     $OutputFile"
Write-Host "  Python DLL: $PythonDll"
Write-Host ""

# Ensure output directory exists
if (!(Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

# Compile: link directly against python3XX.dll (MinGW-compatible).
# Static link libgcc/libstdc++ to avoid MSYS2 runtime dependency.
$args = @(
    "-std=c++17",
    "-O2",
    "-Wall",
    "-shared",
    "-static-libgcc",
    "-static-libstdc++",
    "-static"
) + $PyIncludes + @(
    $PythonDll,
    "-o", $OutputFile,
    "$SrcDir\board.cpp",
    "$SrcDir\rules.cpp",
    "$SrcDir\ab_search.cpp",
    "$SrcDir\bindings.cpp"
)

Write-Host "Command: $GXX $($args -join ' ')"
Write-Host ""

& $GXX @args

if ($LASTEXITCODE -ne 0) {
    Write-Host "BUILD FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "BUILD SUCCEEDED: $OutputFile" -ForegroundColor Green

# Quick import test
Write-Host ""
Write-Host "Testing import..."
python -c "from hybrid.cpp_engine import hybrid_cpp_engine; print('  Import OK:', dir(hybrid_cpp_engine))"
