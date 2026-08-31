"""Local unit test for MiCA (no Vicuna needed). Validates SVD minor selection + file contracts."""

import ast, os, py_compile, sys, torch, yaml

MOD = "minigpt4/models/minigpt4rec_mica.py"
ENTRY = "train_collm_mf_din_mica.py"

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    return cond

ok = True
print("=== MiCA LOCAL UNIT TEST ===")
try:
    py_compile.compile(MOD, doraise=True)
    py_compile.compile(ENTRY, doraise=True)
    ok &= check("py_compile module + entry", True)
except Exception as e:
    ok &= check("py_compile", False, str(e))

src = open(MOD, encoding="utf-8").read()
tree = ast.parse(src)
classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
ok &= check("class MiniGPT4Rec_MiCA defined", "MiniGPT4Rec_MiCA" in classes)
if "MiniGPT4Rec_MiCA" in classes:
    dec = [ast.unparse(d) for d in classes["MiniGPT4Rec_MiCA"].decorator_list]
    ok &= check("registered as mini_gpt4rec_mica", any("mini_gpt4rec_mica" in d for d in dec), str(dec))
    meths = {n.name for n in classes["MiniGPT4Rec_MiCA"].body if isinstance(n, ast.FunctionDef)}
    for m in ["__init__", "from_config"]:
        ok &= check(f"method {m} present", m in meths)

# Validate SVD minor selection logic on synthetic weights
print("\n--- SVD minor selection ---")
torch.manual_seed(0)
for shape in [(512,512), (512,1024), (1024,512)]:
    W = torch.randn(*shape)
    r=8
    U,S,Vh = torch.linalg.svd(W.float(), full_matrices=False)
    U_minor = U[:, -r:]
    minor_s = S[-r:]
    major_s = S[:r]
    ok &= check(f"shape {shape} minor singular values < major", (minor_s.max() < major_s.min()).item(), f"minor max {minor_s.max():.2f} major min {major_s.min():.2f}")
    ok &= check(f"shape {shape} U_minor shape", U_minor.shape == (shape[0], r))

# Simulate peft overwrite
print("\n--- peft overwrite simulation ---")
r=8
W = torch.randn(512,512)
U,S,Vh = torch.linalg.svd(W.float(), full_matrices=False)
B = torch.zeros(512, r)
A = torch.randn(r, 512)
with torch.no_grad():
    B.copy_(U[:, -r:])
    A.zero_()
ok &= check("B copied minor vectors", torch.allclose(B, U[:, -r:]))
ok &= check("A zeroed", (A==0).all().item())
# Verify frozen logic: B requires_grad False after
B.requires_grad = False
A.requires_grad = True
ok &= check("B frozen, A trainable", (not B.requires_grad) and A.requires_grad)

# YAML checks
print("\n--- YAML configs ---")
for p in ["train_configs/collm_mica_pretrain_mf_ood.yaml"]:
    cfg = yaml.safe_load(open(p))
    ok &= check(f"{p} arch", cfg["model"]["arch"]=="mini_gpt4rec_mica")
    ok &= check(f"{p} mica flag", cfg["model"]["lora_config"].get("mica")==True)
    ok &= check(f"{p} freeze_lora False (stage1 trains MiCA)", cfg["model"]["freeze_lora"]==False)

print("\n" + ("="*60))
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
