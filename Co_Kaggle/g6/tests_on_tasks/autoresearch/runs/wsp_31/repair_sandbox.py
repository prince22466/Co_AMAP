"""Install Colab's Python 3.12 venv support, exercise it, and restart the interrupted cohort."""
import os
import subprocess
import tempfile
from pathlib import Path

root = Path("/content/wsp")
python = root/"eval_env/bin/python"
env = dict(os.environ, DEBIAN_FRONTEND="noninteractive", PATH=f"{python.parent}:"+os.environ["PATH"],
           VIRTUAL_ENV=str(python.parent.parent),PYTHONNOUSERSITE="1",
           MAX_JOBS="2",TORCHINDUCTOR_COMPILE_THREADS="1")
with (root/"sandbox_setup.log").open("w") as setup:
    for cmd in (["apt-get","update","-qq"],["apt-get","install","-y","-qq","python3.12-venv"]):
        subprocess.run(cmd,stdout=setup,stderr=subprocess.STDOUT,check=True,env=env)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([str(python),"-m","venv",tmp],check=True,env=env,stdout=setup,stderr=subprocess.STDOUT)
        subprocess.run([str(Path(tmp)/"bin/python"),"-m","pip","--version"],check=True,env=env,stdout=setup)
with (root/"discovery02.log").open("x") as log:
    proc=subprocess.Popen([str(python),"-u",str(root/"run_evaluator.py"),"--split","discovery","--name","discovery02"],
                          cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print("Verified sandbox with pip. Discovery process",proc.pid)
