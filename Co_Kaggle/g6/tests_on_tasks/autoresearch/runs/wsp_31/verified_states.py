"""Construct verified-submit calibration histories on disjoint, published development snapshots."""
import copy
import difflib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

ROOT=Path("/content/wsp")
DATA=ROOT/"input/competitions/gemma-4-developer-agent"
CHECKS={
    "rich_3506": "from rich.segment import Segment; s=Segment(chr(0x6c34)*2+'aaaaa'); a,b=s.split_cells(3); assert (a.cell_length,b.cell_length)==(3,s.cell_length-3)",
    "rich_3067": "from rich.highlighter import ReprHighlighter; s='https://example.org/~sky'; t=ReprHighlighter()(s); assert any(p.style=='repr.url' and p.start==0 and p.end==len(s) for p in t.spans)",
    "rich_3296": "from io import StringIO; from rich.console import Console; from rich.syntax import Syntax; out=StringIO(); c=Console(file=out,width=24,force_terminal=True,color_system='truecolor'); c.print(Syntax('x = 1','python',padding=(1,2),background_color='red')); assert chr(27)+'[41m' in out.getvalue().splitlines()[0]",
}


def invoke(work,argv):
    p=subprocess.run(argv,cwd=work,text=True,capture_output=True,timeout=30)
    return dict(status="ok" if p.returncode==0 else "error",stdout=p.stdout[-1200:],stderr=p.stderr[-1200:],exit_code=p.returncode)


def extend(messages,name,args,response):
    ident=f"calibration-{len(messages)}"
    messages.append({"role":"assistant","tool_calls":[{"id":ident,"type":"function",
                    "function":{"name":name,"arguments":args}}]})
    messages.append({"role":"tool","tool_call_id":ident,"content":json.dumps(response)})


def render(tokenizer,request,messages):
    normalized=copy.deepcopy(messages)
    for message in normalized:
        # The native server renders OpenAI text content as typed parts.
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        for c in message.get("tool_calls",[]):
            args=c["function"]["arguments"]
            if isinstance(args,str): c["function"]["arguments"]=json.loads(args)
    out=tokenizer.apply_chat_template(normalized,tools=request.get("tools"),tokenize=True,
                                      add_generation_prompt=True,return_dict=False,**request.get("chat_template_kwargs",{}))
    return list(out)


def main():
    out=ROOT/"preservation/exact"
    out.mkdir(exist_ok=True,parents=True)
    run=ROOT/"experiments/discovery02"
    requests=list(map(json.loads,(run/"requests.jsonl").read_text().splitlines()))
    exact={r["request_sha"]:r["tokens"] for r in map(json.loads,(run/"replay_tokens.jsonl").read_text().splitlines())}
    tasks={r["instance_id"]:r for r in map(json.loads,(DATA/"tasks.jsonl").read_text().splitlines())}
    tokenizer=AutoTokenizer.from_pretrained(json.loads((ROOT/"assets.json").read_text())["model"],local_files_only=True)
    for request in requests:
        digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        if render(tokenizer,request,request["messages"]) != exact[digest]:
            raise ValueError("Rendered calibration template differs from native captured request "+digest)
    cases,report=[],[]
    for issue,check in CHECKS.items():
        task=tasks[issue]
        request=next(r for r in requests if len(r["messages"])==2 and task["problem_statement"] in r["messages"][0]["content"])
        messages=copy.deepcopy(request["messages"])
        digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        template_matches=render(tokenizer,request,messages)==exact[digest]
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            with tarfile.open(DATA/"snapshots"/(issue+".tgz")) as archive:
                archive.extractall(work,filter="data")
            py=str(ROOT/"eval_env/bin/python")
            argv=[py,"-c",check]
            before=invoke(work,argv)
            files=[line[len("+++ b/"):] for line in task["patch"].splitlines() if line.startswith("+++ b/")]
            original={f:(work/f).read_text() for f in files}
            patch=task["patch"]
            subprocess.run(["git","apply","--whitespace=nowarn","-"],input=patch if patch.endswith("\n") else patch+"\n",
                           text=True,cwd=work,check=True)
            after=invoke(work,argv)
            audit=invoke(work,["git","diff","--check"])
            entry=dict(issue=issue,base_commit=task["base_commit"],baseline=before,correct=after,diff_check=audit,
                       template_matches=template_matches)
            report.append(entry)
            if before["exit_code"]!=1 or "AssertionError" not in before["stderr"] or after["exit_code"] or audit["exit_code"]:
                continue
            for file in files:
                a=original[file].splitlines(keepends=True)
                b=(work/file).read_text().splitlines(keepends=True)
                ops=[e for e in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if e[0]!="equal"]
                start=max(0,ops[0][1]-2); end=min(len(a),ops[-1][2]+2)
                old="".join(a[start:end])
                new="".join(b[ops[0][3]-(ops[0][1]-start):ops[-1][4]+end-ops[-1][2]])
                if original[file].count(old)!=1 or original[file].replace(old,new)!=(work/file).read_text():
                    raise ValueError("Ambiguous calibration edit")
                extend(messages,"read_file",dict(filepath=file,start_line=start+1,end_line=end),
                       dict(status="ok",filepath=file,content=old,is_truncated=False))
                delta="".join(difflib.unified_diff(a,b,fromfile="a/"+file,tofile="b/"+file))
                extend(messages,"edit_file",dict(filepath=file,old_string=old,new_string=new,allow_multiple=False),
                       dict(status="ok",filepath=file,occurrences=1,strategy="exact",diff=delta))
            extend(messages,"run_command",dict(command="python -c "+shlex.quote(check)),after)
            extend(messages,"run_command",dict(command="git diff --check"),audit)
            token_ids=render(tokenizer,request,messages)
            cases.append(dict(issue=issue,snapshot=task["base_commit"],role="verified",action="submit_patch",arguments={},
                              tokens=token_ids,prompt_sha=hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
                              source="executed development calibration history"))
            print(issue,"behavioral check failed before and passed after",len(token_ids),flush=True)
            (out/"verified_checks.json").write_text(json.dumps(report,indent=2))
            (out/"cases.json").write_text(json.dumps(cases))
    (out/"verified_checks.json").write_text(json.dumps(report,indent=2))
    (out/"cases.json").write_text(json.dumps(cases))
    if not cases: raise RuntimeError("No verified calibration histories")


if __name__=="__main__":
    main()
