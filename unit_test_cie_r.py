"""
Local unit test for CIE-R (no GPU, no transformers/peft needed).

Validates, against the SHIPPED SOURCE TEXT of
minigpt4/models/minigpt4rec_cie_r.py:
  1. file compiles (py_compile) and parses (ast)
  2. structural contract: registry name 'mini_gpt4rec_cie_r', class
     MiniGPT4Rec_CIE_R subclasses MiniGPT4Rec_v2, from_config override reads
     cie_reg_lambda, forward_v2 guards on self.training / lambda / encoder,
     graceful-skip branch present
  3. the pure loss function cie_relational_loss is extracted verbatim from the
     module via ast and validated numerically:
       a. identical spaces            -> loss ~ 0
       b. independent random spaces   -> loss > threshold
       c. row-shuffled projection     -> loss increases vs aligned projection
       d. gradient flows into a linear map fitted to preserve structure
          (loss -> small; grad norm > 0)
       e. determinism: repeated calls identical
  4. YAML proxy configs parse and contain the expected keys/values

Run:  python -u unit_test_cie_r.py
"""

import ast
import os
import py_compile
import sys

import torch
import yaml

MODULE = os.path.join("minigpt4", "models", "minigpt4rec_cie_r.py")
ENTRY = "train_collm_mf_din_cie_r.py"
CONFIGS = [
    "train_configs/collm_cie_r_l001_proxy.yaml",
    "train_configs/collm_cie_r_l01_proxy.yaml",
    "train_configs/collm_cie_r_l1_proxy.yaml",
]

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if cond:
        PASS += 1
    else:
        FAIL += 1


def extract_function(module_path, func_name):
    """Return the loss function compiled standalone from the shipped source."""
    src = open(module_path, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            mod = ast.Module(body=[node], type_ignores=[])
            code = compile(mod, module_path, "exec")
            ns = {"torch": torch, "F": torch.nn.functional}
            exec(code, ns)
            return ns[func_name]
    raise RuntimeError(f"function {func_name} not found in {module_path}")


def main():
    torch.manual_seed(42)

    print("=" * 72)
    print("CIE-R LOCAL UNIT TEST")
    print(f"python : {sys.version.split()[0]}")
    print(f"torch  : {torch.__version__}")
    print("=" * 72)

    print("\n--- section 1: compile ---")
    try:
        py_compile.compile(MODULE, doraise=True)
        py_compile.compile(ENTRY, doraise=True)
        check("py_compile module + entry script", True)
    except py_compile.PyCompileError as e:
        check("py_compile module + entry script", False, str(e))
        return finish()

    print("\n--- section 2: structural contract (ast) ---")
    src = open(MODULE, encoding="utf-8").read()
    tree = ast.parse(src)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    check("class MiniGPT4Rec_CIE_R defined", "MiniGPT4Rec_CIE_R" in classes)
    deco = [ast.unparse(d) for d in classes.get("MiniGPT4Rec_CIE_R", ast.Pass()).decorator_list] if "MiniGPT4Rec_CIE_R" in classes else []
    check("registered as mini_gpt4rec_cie_r", any("mini_gpt4rec_cie_r" in d and "registry" in d for d in deco), str(deco))
    cls = classes["MiniGPT4Rec_CIE_R"]
    bases = [ast.unparse(b) for b in cls.bases]
    check("subclasses MiniGPT4Rec_v2", bases == ["MiniGPT4Rec_v2"], str(bases))
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    for m in ["from_config", "_batch_relation_loss", "forward_v2", "__init__"]:
        check(f"method {m} present", m in methods)
    fc_src = ast.unparse(methods["from_config"])
    check("from_config reads cie_reg_lambda", "cie_reg_lambda" in fc_src)
    fv_src = ast.unparse(methods["forward_v2"])
    check("forward_v2 trains-only guard", "self.training" in fv_src and "self.cie_reg_lambda > 0" in fv_src)
    bl_src = ast.unparse(methods["_batch_relation_loss"])
    check("graceful skip when no history field", "hist_ids is None" in bl_src or 'samples.get("InteractedItemIDs_pad")' in bl_src)
    check("raw side detached (frozen encoder)", ".detach()" in bl_src)

    print("\n--- section 3: loss math (extracted from shipped source) ---")
    loss_fn = extract_function(MODULE, "cie_relational_loss")

    N, DR, DP = 12, 8, 16
    base = torch.nn.functional.normalize(torch.randn(N, DR), dim=-1)

    # (a) identical spaces
    proj_ident = base.clone().requires_grad_(False)
    l_a = loss_fn(base, proj_ident)
    print(f"    identical-space loss      : {l_a.item():.3e}")
    check("(a) identical spaces -> ~0", l_a.item() < 1e-6)

    # (b) independent random spaces
    indep = torch.randn(N, DP)
    l_b = loss_fn(base, indep)
    print(f"    independent-space loss    : {l_b.item():.6f}")
    check("(b) independent spaces -> large (>0.1)", l_b.item() > 0.1)

    # (c) alignment beats shuffling
    g = torch.Generator().manual_seed(7)
    W = torch.randn(DR, DP, generator=g) / DR ** 0.5
    aligned = base @ W
    perm = torch.randperm(N, generator=g)
    shuffled = aligned[perm]
    l_al = loss_fn(base, aligned).item()
    l_sh = loss_fn(base, shuffled).item()
    print(f"    aligned-projection loss   : {l_al:.6f}")
    print(f"    shuffled-projection loss  : {l_sh:.6f}")
    check("(c) shuffling increases loss", l_sh > l_al * 2)

    # (d) gradient flows into the projector and optimizes the loss down
    lin = torch.nn.Linear(DR, DP)
    with torch.no_grad():
        lin.weight.copy_(W.T)
        lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), lr=5e-2)
    first, last = None, None
    for step in range(60):
        opt.zero_grad()
        out = lin(base.detach().clone())
        l = loss_fn(base, out)
        if first is None:
            first = l.item()
        last = l.item()
        l.backward()
        gn = sum(p.grad.norm().item() for p in lin.parameters() if p.grad is not None)
        if step == 0:
            print(f"    step 0: loss={first:.6f} grad_norm={gn:.3e}")
            check("(d) gradients reach projector at step 0", gn > 0)
        opt.step()
    print(f"    after 60 steps: loss={last:.6f} (first={first:.6f})")
    check("(d) optimization reduces loss substantially", last < first * 0.25)
    check("(d) optimized loss near zero (<1e-3)", last < 1e-3, f"{last:.2e}")

    # (e) determinism
    l1 = loss_fn(base, aligned).item()
    l2 = loss_fn(base, aligned).item()
    check("(e) deterministic", abs(l1 - l2) == 0.0)

    # tiny-N edge
    l_tiny = loss_fn(torch.randn(1, DR), torch.randn(1, DP))
    check("edge: single token returns finite 0-ish scalar", torch.isfinite(l_tiny).item())

    print("\n--- section 4: proxy configs ---")
    expected_lambda = {"collm_cie_r_l001_proxy.yaml": 0.01,
                       "collm_cie_r_l01_proxy.yaml": 0.1,
                       "collm_cie_r_l1_proxy.yaml": 1.0}
    for cfg_path in CONFIGS:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        name = os.path.basename(cfg_path)
        ok_arch = cfg["model"]["arch"] == "mini_gpt4rec_cie_r"
        ok_lam = float(cfg["model"]["cie_reg_lambda"]) == expected_lambda[name]
        run = cfg["run"]
        ok_run = (run["max_epoch"] == 12 and run["eval_freq"] == 2
                  and run["distributed"] is False and run["world_size"] == 1
                  and cfg["model"]["freeze_rec"] is True
                  and cfg["model"]["freeze_lora"] is True
                  and cfg["model"]["freeze_proj"] is False
                  and cfg["model"]["proj_token_num"] == 1
                  and name.split("_l")[1].split("_")[0] in run["output_dir"])
        check(f"config {name}", ok_arch and ok_lam and ok_run,
              f"arch_ok={ok_arch} lambda={cfg['model']['cie_reg_lambda']} run_ok={ok_run}")

    return finish()


def finish():
    print("\n" + "=" * 72)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 72)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
