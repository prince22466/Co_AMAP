"""Upload fixed serving_logits_v2.py and execute Step 2 compliance + rank1 + rank2 serving logits."""
import subprocess
from pathlib import Path

from provision_and_step2 import CLI, HERE, SESSION, upload_verified
from run_step3_and_step4 import run_remote_stage


def main():
    upload_verified(HERE / "serving_logits_v2.py", "/content/wsp/serving_logits_v2.py")
    py = "/content/wsp/eval_env/bin/python"
    run_remote_stage(
        "step2_logits",
        f"{py} -u /content/wsp/serving_logits_v2.py compliance && "
        f"{py} -u /content/wsp/serving_logits_v2.py rank1 && "
        f"{py} -u /content/wsp/serving_logits_v2.py rank2",
        timeout_sec=900,
    )
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/compliance_preflight.json", str(HERE / "compliance_preflight.json")], check=True)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/logit_preflight/comparison.json", str(HERE / "logit_comparison.json")], check=True)
    print("=== COMPLIANCE PREFLIGHT ===")
    print((HERE / "compliance_preflight.json").read_text())
    print("=== STEP 2 LOGIT COMPARISON ===")
    print((HERE / "logit_comparison.json").read_text())


if __name__ == "__main__":
    main()
