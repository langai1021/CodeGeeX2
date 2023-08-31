import uuid

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
import argparse


#获取选项        
def add_code_generation_args(parser):
    group = parser.add_argument_group(title="CodeGeeX2 DEMO")
    group.add_argument(
        "--model-path",
        type=str,
        default="THUDM/codegeex2-6b",
    )
    group.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
    )
    group.add_argument(
        "--port",
        type=int,
        default=7860,
    )
    group.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    group.add_argument(                      
        "--half",
        action="store_true",
    )
    group.add_argument(
        "--quantize",
        type=int,
        default=None,
    )
    group.add_argument(
        "--max-length",
        type=int,
        default=2048
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=0.95
    )
    group.add_argument(
        "--n-sample",
        type=int,
        default=20
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=0.2
    )
    return parser

LANGUAGE_TAG = {
    "Abap"         : "* language: Abap",
    "ActionScript" : "// language: ActionScript",
    "Ada"          : "-- language: Ada",
    "Agda"         : "-- language: Agda",
    "ANTLR"        : "// language: ANTLR",
    "AppleScript"  : "-- language: AppleScript",
    "Assembly"     : "; language: Assembly",
    "Augeas"       : "// language: Augeas",
    "AWK"          : "// language: AWK",
    "Basic"        : "' language: Basic",
    "C"            : "// language: C",
    "C#"           : "// language: C#",
    "C++"          : "// language: C++",
    "CMake"        : "# language: CMake",
    "Cobol"        : "// language: Cobol",
    "CSS"          : "/* language: CSS */",
    "CUDA"         : "// language: Cuda",
    "Dart"         : "// language: Dart",
    "Delphi"       : "{language: Delphi}",
    "Dockerfile"   : "# language: Dockerfile",
    "Elixir"       : "# language: Elixir",
    "Erlang"       : f"% language: Erlang",
    "Excel"        : "' language: Excel",
    "F#"           : "// language: F#",
    "Fortran"      : "!language: Fortran",
    "GDScript"     : "# language: GDScript",
    "GLSL"         : "// language: GLSL",
    "Go"           : "// language: Go",
    "Groovy"       : "// language: Groovy",
    "Haskell"      : "-- language: Haskell",
    "HTML"         : "<!--language: HTML-->",
    "Isabelle"     : "(*language: Isabelle*)",
    "Java"         : "// language: Java",
    "JavaScript"   : "// language: JavaScript",
    "Julia"        : "# language: Julia",
    "Kotlin"       : "// language: Kotlin",
    "Lean"         : "-- language: Lean",
    "Lisp"         : "; language: Lisp",
    "Lua"          : "// language: Lua",
    "Markdown"     : "<!--language: Markdown-->",
    "Matlab"       : f"% language: Matlab",
    "Objective-C"  : "// language: Objective-C",
    "Objective-C++": "// language: Objective-C++",
    "Pascal"       : "// language: Pascal",
    "Perl"         : "# language: Perl",
    "PHP"          : "// language: PHP",
    "PowerShell"   : "# language: PowerShell",
    "Prolog"       : f"% language: Prolog",
    "Python"       : "# language: Python",
    "R"            : "# language: R",
    "Racket"       : "; language: Racket",
    "RMarkdown"    : "# language: RMarkdown",
    "Ruby"         : "# language: Ruby",
    "Rust"         : "// language: Rust",
    "Scala"        : "// language: Scala",
    "Scheme"       : "; language: Scheme",
    "Shell"        : "# language: Shell",
    "Solidity"     : "// language: Solidity",
    "SPARQL"       : "# language: SPARQL",
    "SQL"          : "-- language: SQL",
    "Swift"        : "// language: swift",
    "TeX"          : f"% language: TeX",
    "Thrift"       : "/* language: Thrift */",
    "TypeScript"   : "// language: TypeScript",
    "Vue"          : "<!--language: Vue-->",
    "Verilog"      : "// language: Verilog",
    "Visual Basic" : "' language: Visual Basic",
}

app = FastAPI()
def device():
    if not args.half:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, device='cuda')
    else:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, device='cuda').half().to("cuda")
    if args.quantize in [4, 8]:
        print(f"Model is quantized to INT{args.quantize} format.")
        model = model.half().quantize(args.quantize)

    return model.eval()

@app.post("/v1/chat/completions")
async def completions(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    lang = json_post_list.get('lang')
    if lang is None:
        lang = "Java"
    prompt = None
    for message in json_post_list['messages']:
        if message['role'] == 'user':
            prompt = message['content']
            break
    max_length = args.max_length
    top_p = args.top_p
    temperature = args.temperature
    n_sample = args.n_sample
    if lang != "None":
        prompt = LANGUAGE_TAG[lang] + "\n// " + prompt + "\n"

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs,
                             max_length=max_length,
                             top_p=top_p,
                             temperature=temperature,
                             do_sample=True,
                             n_sample=n_sample)
    response = tokenizer.decode(outputs[0])
    # response = model.chat(tokenizer,
    #                       prompt,
    #                       max_length=max_length,
    #                       top_p=top_p,
    #                       top_k=top_k,
    #                       temperature=temperature)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "id": "LLM-" + str(uuid.uuid1()),
        "object": "chat.completion",
        "created": int(now.timestamp() * 1000),
        "model": args.model_path,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response,
            },
            "finish_reason": "stop"
        }]
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)

    return answer



if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = device()
    uvicorn.run(app, host=args.listen, port=args.port, workers=args.workers)