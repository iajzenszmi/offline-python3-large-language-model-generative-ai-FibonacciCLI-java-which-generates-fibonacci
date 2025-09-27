#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# --------- Helpers ---------
def run_cmd_capture(cmd, cwd=None):
    """Run a command and capture stdout+stderr; raise on spawn error."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    out, _ = proc.communicate()
    return proc.returncode, out

def looks_like_java(text: str) -> bool:
    return ("class " in text) or ("public static void main" in text)

def extract_java_code(text: str) -> str | None:
    # 1) ```java fenced
    m = re.search(r"```(?:\s*java)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if looks_like_java(candidate):
            return candidate

    # 2) Any triple backtick with Java-y content
    for m in re.finditer(r"```(.*?)```", text, re.DOTALL):
        candidate = m.group(1).strip()
        if looks_like_java(candidate):
            return candidate

    # 3) Fallback: from first "class " onward
    idx = text.find("class ")
    if idx >= 0:
        return text[idx:].strip()

    # 4) Last-ditch attempt: whole text if it smells like Java
    if looks_like_java(text):
        return text.strip()

    return None

def infer_class_name(code: str) -> str | None:
    m = re.search(r"\bpublic\s+class\s+(\w+)", code)
    if m:
        return m.group(1)
    m = re.search(r"\bclass\s+(\w+)", code)
    if m:
        return m.group(1)
    m = re.search(r"\bpublic\s+record\s+(\w+)", code)
    if m:
        return m.group(1)
    return None

def build_system_prompt(user_prompt: str) -> str:
    return (
        "You are a code generator that outputs a single complete Java source file only.\n"
        "Rules:\n"
        "- Return exactly one file, with a single public class.\n"
        "- Include a main method when a CLI is implied.\n"
        "- Use only the Java standard library.\n"
        "- If multiple helpers are needed, use static nested classes.\n"
        "- If input is needed, use args[] or standard input.\n"
        "- Prefer readability and correct error handling.\n\n"
        f"Task: {user_prompt}\n"
        "Output: One compilable Java source file.\n"
    )

def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"Wrote: {path}")

def strip_dot_java(filename: str) -> str:
    base = os.path.basename(filename)
    return base[:-5] if base.lower().endswith(".java") else base

# --------- Engines ---------
def call_ollama(model: str, prompt: str) -> str:
    # Minimal, offline-safe:
    #   ollama run <model> "prompt"
    cmd = ["ollama", "run", model, prompt]
    rc, out = run_cmd_capture(cmd)
    if rc != 0 and not out.strip():
        raise RuntimeError(f"ollama failed (exit {rc})")
    return out

def call_llamacpp(model_path: str, prompt: str) -> str:
    # Minimal, offline-safe:
    #   ./main -m <model.gguf> -p "prompt"
    cmd = ["./main", "-m", model_path, "-p", prompt]
    rc, out = run_cmd_capture(cmd)
    if rc != 0 and not out.strip():
        raise RuntimeError(f"llama.cpp failed (exit {rc})")
    return out

# --------- Main flow ---------
def main():
    ap = argparse.ArgumentParser(
        description="Offline Java code generator via local LLM (Ollama or llama.cpp)."
    )
    ap.add_argument("--engine", choices=["ollama", "llamacpp"], default="ollama",
                    help="Local engine to use (default: ollama)")
    ap.add_argument("-m", "--model", required=False, default="qwen2.5-coder:7b",
                    help="Ollama: model tag; llama.cpp: path to .gguf")
    ap.add_argument("-o", "--out", help="Output .java filename (inferred if omitted)")
    ap.add_argument("--compile", action="store_true", help="Compile with javac")
    ap.add_argument("--run", action="store_true", help="Run with java after compile")
    ap.add_argument("--raw", action="store_true", help="Send prompt verbatim (no wrapper)")
    ap.add_argument("--show-output", action="store_true",
                    help="Also print raw model output (debug)")
    ap.add_argument("prompt", nargs="+", help="English prompt for the generator")

    args = ap.parse_args()
    user_prompt = " ".join(args.prompt)

    full_prompt = user_prompt if args.raw else build_system_prompt(user_prompt)

    # Call engine
    if args.engine == "ollama":
        model_text = call_ollama(args.model, full_prompt)
    else:
        model_text = call_llamacpp(args.model, full_prompt)

    if args.show_output:
        print("----- BEGIN RAW MODEL OUTPUT -----")
        print(model_text)
        print("----- END RAW MODEL OUTPUT -----")

    code = extract_java_code(model_text)
    if not code:
        print("No Java code could be extracted. Raw model output below:\n", file=sys.stderr)
        print(model_text, file=sys.stderr)
        sys.exit(2)

    class_name = infer_class_name(code) or "GeneratedProgram"
    out_file = args.out if args.out else f"{class_name}.java"
    if not out_file.lower().endswith(".java"):
        print(f"Output must be a .java file: {out_file}", file=sys.stderr)
        sys.exit(2)

    out_path = Path(out_file).resolve()
    write_text(out_path, code)

    # Compile
    if args.compile:
        rc, _ = run_cmd_capture(["javac", str(out_path)])
        if rc != 0:
            print("Compilation failed. See errors above.", file=sys.stderr)
            sys.exit(rc)
        print("Compilation succeeded.")

    # Run
    if args.run:
        main_class = strip_dot_java(out_file)
        rc, _ = run_cmd_capture(["java", main_class])
        sys.exit(rc)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Command not found: {e}", file=sys.stderr)
        sys.exit(127)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
