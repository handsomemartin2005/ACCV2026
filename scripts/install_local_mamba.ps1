$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$mambaRoot = Join-Path $repoRoot "ultralytics\nn\extra_modules\mamba"

$env:MAMBA_SKIP_CUDA_BUILD = "TRUE"
python -m pip install --no-build-isolation --no-deps -e $mambaRoot

$verify = @'
import torch
import mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_cuda

B, D, L, N = 1, 4, 8, 3
u = torch.randn(B, D, L)
delta = torch.randn(B, D, L)
A = -torch.rand(D, N)
Bv = torch.randn(B, N, L)
Cv = torch.randn(B, N, L)
Dskip = torch.ones(D)
y = selective_scan_fn(u, delta, A, Bv, Cv, Dskip, delta_softplus=True)

print("mamba_ssm", mamba_ssm.__version__)
print("mamba_path", mamba_ssm.__file__)
print("selective_scan_cuda", selective_scan_cuda)
print("selective_scan_output", tuple(y.shape), torch.isfinite(y).all().item())
'@

$verify | python -
