# offline-python3-large-language-model-generative-ai-FibonacciCLI-java-which-generates-fibonacci
offline python3 large language model generative ai program generates FibonacciCLI.java program which generates fibonacci series numbers.
Skip to content
Chat history
You said:
An LLm gen ai capable of producing java programs from english language prompts coded in java.
ChatGPT said:

Got it — here’s a single-file Java CLI app that talks to a local LLM (e.g., Ollama) to generate Java source code from natural-language prompts, optionally saves it to *.java, compiles it with javac, and runs it.
How it works

    Uses Java 11’s java.net.http.HttpClient to call the Ollama REST API (http://localhost:11434/api/generate).

    Default model: qwen2.5-coder:7b (change with -m).

    Prompts the model with a strict system-style instruction so it returns a compilable Java file.

    Extracts java code fences if present; otherwise falls back to best-effort extraction.

    Infers class name from class <Name>; otherwise uses GeneratedProgram.

    Optional --compile and --run.

File: CodeGenAI.java

import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * CodeGenAI
 *
 * A single-file Java CLI that asks a local LLM (e.g., Ollama) to generate Java programs from English prompts.
 *
 * Requirements:
 *  - Java 11+
 *  - A running Ollama server (default: http://localhost:11434) with a code-capable model pulled, e.g.:
 *      ollama pull qwen2.5-coder:7b
 *      ollama pull codellama:7b
 *
 * Usage examples:
 *   javac CodeGenAI.java
 *   java CodeGenAI "Write a Java program that prints primes up to 100."
 *
 *   # Choose a model and write to file:
 *   java CodeGenAI -m qwen2.5-coder:7b -o Primes.java "Program that lists primes to N using Sieve, N from args"
 *
 *   # Generate, compile, and run immediately:
 *   java CodeGenAI --compile --run "CLI Fibonacci that prints first 20 numbers"
 *
 * Options:
 *   -m <model>        Set Ollama model (default: qwen2.5-coder:7b)
 *   -o <file>         Save output to this filename (must end with .java)
 *   -t <temp>         Temperature (e.g., 0.2)
 *   --host <url>      Base URL for Ollama (default: http://localhost:11434)
 *   --compile         Run 'javac' on the generated file
 *   --run             Run 'java <MainClass>' after successful compile
 *   --raw             Do not wrap the prompt with system instructions
 *   --help            Show help
 *
 * Notes:
 *  - If the LLM returns multiple classes, the primary public class should match the filename you specify with -o.
 *  - If no -o is given, the tool infers class name from the code and writes <ClassName>.java.
 */
public class CodeGenAI {

    // Simple arg parsing struct
    static class Config {
        String model = "qwen2.5-coder:7b";
        String host = "http://localhost:11434";
        String outFile = null;   // if null, infer from class name
        boolean compile = false;
        boolean run = false;
        boolean rawPrompt = false;
        double temperature = 0.2;
        String prompt = null;
    }

    public static void main(String[] args) {
        try {
            Config cfg = parseArgs(args);
            if (cfg == null) return; // help shown

            String fullPrompt = cfg.rawPrompt ? cfg.prompt : buildSystemPrompt(cfg.prompt);

            String response = callOllamaGenerate(cfg.host, cfg.model, fullPrompt, cfg.temperature);

            String code = extractJavaCode(response).orElseGet(() -> fallbackExtract(response));
            if (code == null || code.isBlank()) {
                System.err.println("No Java code found in model response. Raw response below:\n");
                System.out.println(response);
                System.exit(2);
            }

            // Infer class name if needed
            String className = inferPrimaryClassName(code).orElse("GeneratedProgram");
            String fileName = cfg.outFile != null ? cfg.outFile : className + ".java";

            if (!fileName.toLowerCase(Locale.ROOT).endsWith(".java")) {
                System.err.println("Output file must end with .java: " + fileName);
                System.exit(2);
            }

            Path out = Path.of(fileName).toAbsolutePath();
            Files.writeString(out, code, StandardCharsets.UTF_8);
            System.out.println("Wrote: " + out);

            if (cfg.compile) {
                int ec = runProcess(List.of("javac", out.toString()));
                if (ec != 0) {
                    System.err.println("Compilation failed (exit code " + ec + "). See errors above.");
                    System.exit(ec);
                } else {
                    System.out.println("Compilation succeeded.");
                }
            }

            if (cfg.run) {
                String mainClass = stripDotJavaToClass(fileName);
                int ec = runProcess(List.of("java", mainClass));
                System.exit(ec);
            }

        } catch (UncheckedIOException e) {
            System.err.println("I/O error: " + e.getMessage());
            System.exit(2);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(2);
        }
    }

    // Build an instruction that nudges the model to emit a single, compilable Java file.
    private static String buildSystemPrompt(String userPrompt) {
        StringBuilder sb = new StringBuilder();
        sb.append("You are a code generator that ONLY outputs a single complete Java source file.\n");
        sb.append("Requirements:\n");
        sb.append("- Return code in one file, not explanations.\n");
        sb.append("- Use a single public class whose name matches the filename if specified.\n");
        sb.append("- Include a main method if a CLI is implied.\n");
        sb.append("- Avoid external dependencies; only use the Java standard library.\n");
        sb.append("- If input is required, read from args[] or standard input.\n");
        sb.append("- If the task suggests multiple classes, nest static classes in the same file.\n");
        sb.append("- Do not include Markdown fencing unless necessary; if you do, use ```java fences.\n\n");
        sb.append("Task: ").append(userPrompt).append("\n");
        sb.append("Output: A single compilable Java source file.\n");
        return sb.toString();
    }

    // Call Ollama /api/generate (non-streaming) with a plain prompt
    private static String callOllamaGenerate(String host, String model, String prompt, double temperature) throws IOException, InterruptedException {
        String url = host.endsWith("/") ? host + "api/generate" : host + "/api/generate";
        String json = "{\"model\":\"" + escapeJson(model) + "\","
                + "\"prompt\":\"" + escapeJson(prompt) + "\","
                + "\"stream\":false,"
                + "\"options\":{\"temperature\":" + temperature + "}"
                + "}";

        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();

        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(120))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();

        HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 300) {
            throw new IOException("Ollama returned HTTP " + resp.statusCode() + ": " + resp.body());
        }

        // Expected response JSON has a "response" field with model text
        String body = resp.body();
        String modelText = extractJsonField(body, "response").orElse(body);
        return modelText;
    }

    // Extract Java fenced code if present
    private static Optional<String> extractJavaCode(String text) {
        Pattern p = Pattern.compile("```java\\s*(.*?)```", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
        Matcher m = p.matcher(text);
        if (m.find()) {
            String code = m.group(1).trim();
            return Optional.of(code);
        }
        // Sometimes models return triple backticks without language
        Pattern p2 = Pattern.compile("```\\s*(.*?)```", Pattern.DOTALL);
        Matcher m2 = p2.matcher(text);
        if (m2.find()) {
            String code = m2.group(1).trim();
            // Heuristic: looks like Java?
            if (code.contains("class ") || code.contains("public static void main")) {
                return Optional.of(code);
            }
        }
        return Optional.empty();
    }

    // If no fences, try to grab likely Java content by heuristics
    private static String fallbackExtract(String text) {
        // Very simple heuristic: find first 'class ' and capture until end
        int idx = text.indexOf("class ");
        if (idx >= 0) return text.substring(idx);
        // else return entire text; user can edit
        return text.trim();
    }

    // Infer the public or primary class name
    private static Optional<String> inferPrimaryClassName(String code) {
        // Try public class first
        Matcher pub = Pattern.compile("\\bpublic\\s+class\\s+(\\w+)").matcher(code);
        if (pub.find()) return Optional.of(pub.group(1));

        // Fallback to any class
        Matcher any = Pattern.compile("\\bclass\\s+(\\w+)").matcher(code);
        if (any.find()) return Optional.of(any.group(1));

        // Try record (Java 16+)
        Matcher rec = Pattern.compile("\\bpublic\\s+record\\s+(\\w+)").matcher(code);
        if (rec.find()) return Optional.of(rec.group(1));

        return Optional.empty();
    }

    private static int runProcess(List<String> cmd) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();
        Process p = pb.start();
        return p.waitFor();
    }

    private static String stripDotJavaToClass(String fileName) {
        String base = fileName;
        int i = base.lastIndexOf('/');
        if (i >= 0) base = base.substring(i + 1);
        int j = base.lastIndexOf('\\');
        if (j >= 0) base = base.substring(j + 1);
        if (base.toLowerCase(Locale.ROOT).endsWith(".java")) {
            base = base.substring(0, base.length() - 5);
        }
        return base;
    }

    // Minimal JSON parsing helpers (no external libs)
    private static Optional<String> extractJsonField(String json, String field) {
        // Very naive string search for "field":"..."
        String needle = "\"" + field + "\":";
        int i = json.indexOf(needle);
        if (i < 0) return Optional.empty();
        int start = json.indexOf('"', i + needle.length());
        if (start < 0) return Optional.empty();
        int end = findJsonStringEnd(json, start + 1);
        if (end < 0) return Optional.empty();
        String raw = json.substring(start + 1, end);
        return Optional.of(unescapeJson(raw));
    }

    private static int findJsonStringEnd(String s, int from) {
        boolean esc = false;
        for (int i = from; i < s.length(); i++) {
            char c = s.charAt(i);
            if (esc) {
                esc = false;
            } else if (c == '\\') {
                esc = true;
            } else if (c == '"') {
                return i;
            }
        }
        return -1;
    }

    private static String escapeJson(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 16);
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"'  -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default   -> {
                    if (c < 32) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }

    private static String unescapeJson(String s) {
        // Handles common escapes. Unicode escapes are preserved if malformed.
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length();) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char n = s.charAt(i + 1);
                switch (n) {
                    case '"'  -> { sb.append('"');  i += 2; }
                    case '\\' -> { sb.append('\\'); i += 2; }
                    case '/'  -> { sb.append('/');  i += 2; }
                    case 'b'  -> { sb.append('\b'); i += 2; }
                    case 'f'  -> { sb.append('\f'); i += 2; }
                    case 'n'  -> { sb.append('\n'); i += 2; }
                    case 'r'  -> { sb.append('\r'); i += 2; }
                    case 't'  -> { sb.append('\t'); i += 2; }
                    case 'u'  -> {
                        if (i + 6 <= s.length()) {
                            String hex = s.substring(i + 2, i + 6);
                            try {
                                int cp = Integer.parseInt(hex, 16);
                                sb.append((char) cp);
                                i += 6;
                            } catch (NumberFormatException e) {
                                sb.append("\\u").append(hex);
                                i += 6;
                            }
                        } else {
                            sb.append("\\u");
                            i += 2;
                        }
                    }
                    default -> { sb.append(n); i += 2; }
                }
            } else {
                sb.append(c);
                i++;
            }
        }
        return sb.toString();
    }

    private static Config parseArgs(String[] args) {
        if (args.length == 0) {
            printHelp();
            return null;
        }
        Config cfg = new Config();
        List<String> tail = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--help":
                case "-h":
                    printHelp();
                    return null;
                case "-m":
                    ensureHas(args, i + 1, "-m requires a model");
                    cfg.model = args[++i];
                    break;
                case "-o":
                    ensureHas(args, i + 1, "-o requires a filename");
                    cfg.outFile = args[++i];
                    break;
                case "-t":
                    ensureHas(args, i + 1, "-t requires a number");
                    cfg.temperature = Double.parseDouble(args[++i]);
                    break;
                case "--host":
                    ensureHas(args, i + 1, "--host requires a URL");
                    cfg.host = args[++i];
                    break;
                case "--compile":
                    cfg.compile = true;
                    break;
                case "--run":
                    cfg.run = true;
                    break;
                case "--raw":
                    cfg.rawPrompt = true;
                    break;
                default:
                    tail.add(a);
            }
        }
        if (tail.isEmpty()) {
            System.err.println("Missing prompt. Example: java CodeGenAI \"Write a Java program that prints primes up to 100.\"");
            return null;
        }
        cfg.prompt = String.join(" ", tail);
        return cfg;
    }

    private static void ensureHas(String[] args, int idx, String err) {
        if (idx >= args.length) {
            throw new IllegalArgumentException(err);
        }
    }

    private static void printHelp() {
        System.out.println("CodeGenAI - Java LLM code generator");
        System.out.println("Usage: java CodeGenAI [options] \"your prompt here\"");
        System.out.println("Options:");
        System.out.println("  -m <model>     Set model (default qwen2.5-coder:7b)");
        System.out.println("  -o <file>      Save as file (must end with .java)");
        System.out.println("  -t <temp>      Temperature (default 0.2)");
        System.out.println("  --host <url>   Ollama host (default http://localhost:11434)");
        System.out.println("  --compile      Compile with javac");
        System.out.println("  --run          Run with java after compile");
        System.out.println("  --raw          Use your prompt verbatim (no system wrapper)");
        System.out.println("  --help         Show help");
    }
}

Quick start

# 1) Make sure Ollama is running and you have a coder model:
ollama pull qwen2.5-coder:7b
ollama serve   # or have the daemon running

# 2) Compile the CLI
javac CodeGenAI.java

# 3) Generate code
java CodeGenAI "Write a Java program that prints the first 25 Fibonacci numbers."

# 4) Generate, compile, and run immediately
java CodeGenAI --compile --run "CLI that reads an integer N and prints all primes up to N using the Sieve of Eratosthenes."

# 5) Choose model and explicit filename
java CodeGenAI -m codellama:7b -o Primes.java --compile --run "Print primes up to 200"

If you prefer Ollama chat format or want multi-turn memory, I can share a variant that uses /api/chat with a system + user message array. Want that too?
You said:
all offline
ChatGPT said:

Perfect — here’s a single-file, fully offline Java CLI that generates Java programs from English prompts without any HTTP calls.

It can drive either:

    Ollama CLI (local, offline, uses your already-downloaded models), or

    llama.cpp’s main binary (point it at a local .gguf model).

You choose the engine with --engine ollama (default) or --engine llamacpp.
File: CodeGenAI_Offline.java

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * CodeGenAI_Offline
 *
 * Generate a single-file Java program from an English prompt using a LOCAL LLM engine
 * with ZERO network calls. Supports:
 *   --engine ollama    (default)  -> uses:  ollama run <model> -p "<prompt>"
 *   --engine llamacpp             -> uses:  ./main -m <model_path.gguf> -p "<prompt>"
 *
 * You must have the chosen engine installed locally and a model present on disk.
 *
 * Usage:
 *   javac CodeGenAI_Offline.java
 *   java CodeGenAI_Offline "Print Fibonacci numbers up to 100"
 *
 *   # Choose engine/model, compile, and run:
 *   java CodeGenAI_Offline --engine ollama -m qwen2.5-coder:7b --compile --run \
 *     "CLI that reads N and prints primes to N using Sieve"
 *
 *   # llama.cpp example:
 *   java CodeGenAI_Offline --engine llamacpp -m /models/Qwen2.5-Coder-7B.Q4_K_M.gguf \
 *     --compile --run "Program that sorts lines from stdin"
 *
 * Options:
 *   --engine <ollama|llamacpp>  Select local engine (default: ollama)
 *   -m <model>                  Model name (ollama) or .gguf path (llama.cpp)
 *   -o <file>                   Save as this .java file (otherwise inferred)
 *   -t <temp>                   Temperature (default 0.2)
 *   --max-tokens <n>            Max tokens to sample (engine-specific default if omitted)
 *   --compile                   Run 'javac' after generation
 *   --run                       Run 'java <MainClass>' after compile
 *   --raw                       Do not wrap your prompt (send verbatim)
 *   --help                      Show help
 *
 * Notes:
 * - Entirely offline at runtime. For Ollama you should have already pulled the model once.
 * - For multi-class outputs, the tool expects a single public class; nest helpers if needed.
 */
public class CodeGenAI_Offline {

    static class Config {
        String engine = "ollama";           // or "llamacpp"
        String model = "qwen2.5-coder:7b";  // for llama.cpp, set to a .gguf path
        String outFile = null;
        boolean compile = false;
        boolean run = false;
        boolean rawPrompt = false;
        double temperature = 0.2;
        Integer maxTokens = null;
        String prompt = null;
    }

    public static void main(String[] args) {
        try {
            Config cfg = parseArgs(args);
            if (cfg == null) return;

            String fullPrompt = cfg.rawPrompt ? cfg.prompt : buildSystemPrompt(cfg.prompt);

            String modelText = switch (cfg.engine.toLowerCase(Locale.ROOT)) {
                case "ollama"   -> runOllama(fullPrompt, cfg);
                case "llamacpp" -> runLlamaCpp(fullPrompt, cfg);
                default         -> throw new IllegalArgumentException("Unknown engine: " + cfg.engine);
            };

            String code = extractJavaCode(modelText).orElseGet(() -> fallbackExtract(modelText));
            if (code == null || code.isBlank()) {
                System.err.println("No Java code found. Raw model output follows:\n");
                System.out.println(modelText);
                System.exit(2);
            }

            String className = inferPrimaryClassName(code).orElse("GeneratedProgram");
            String fileName = cfg.outFile != null ? cfg.outFile : className + ".java";
            if (!fileName.toLowerCase(Locale.ROOT).endsWith(".java")) {
                System.err.println("Output file must end with .java: " + fileName);
                System.exit(2);
            }

            Path out = Path.of(fileName).toAbsolutePath();
            Files.writeString(out, code, StandardCharsets.UTF_8);
            System.out.println("Wrote: " + out);

            if (cfg.compile) {
                int ec = runProcessInheritIO(List.of("javac", out.toString()));
                if (ec != 0) {
                    System.err.println("Compilation failed (exit " + ec + ").");
                    System.exit(ec);
                } else {
                    System.out.println("Compilation succeeded.");
                }
            }

            if (cfg.run) {
                String mainClass = stripDotJavaToClass(fileName);
                int ec = runProcessInheritIO(List.of("java", mainClass));
                System.exit(ec);
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(2);
        }
    }

    // Wrap the user task so the model emits ONE compilable Java file.
    private static String buildSystemPrompt(String user) {
        StringBuilder sb = new StringBuilder();
        sb.append("You are a code generator that outputs a single complete Java source file only.\n");
        sb.append("Rules:\n");
        sb.append("- Return exactly one file, with a single public class.\n");
        sb.append("- Include a main method when a CLI is implied.\n");
        sb.append("- Use only Java standard library.\n");
        sb.append("- If multiple helpers are needed, use static nested classes.\n");
        sb.append("- If input is needed, use args[] or standard input.\n");
        sb.append("- Prefer readability and correct error handling.\n\n");
        sb.append("Task: ").append(user).append("\n");
        sb.append("Output: One compilable Java source file.\n");
        return sb.toString();
    }

    /* ===================== Engines (Local, Offline) ===================== */

    private static String runOllama(String prompt, Config cfg) throws IOException, InterruptedException {
        // Uses the ollama CLI. Entirely offline if model is present locally.
        // Command form: ollama run <model> -p "<prompt>"
        List<String> cmd = new ArrayList<>();
        cmd.add("ollama");
        cmd.add("run");
        cmd.add(cfg.model);
        cmd.add("-p");
        cmd.add(prompt);

        // Engine options (if supported by your version)
        if (cfg.maxTokens != null) {
            cmd.add("--num-predict");
            cmd.add(String.valueOf(cfg.maxTokens));
        }
        cmd.add("--temperature");
        cmd.add(String.valueOf(cfg.temperature));

        return runProcessCaptureStdout(cmd);
    }

    private static String runLlamaCpp(String prompt, Config cfg) throws IOException, InterruptedException {
        // Uses llama.cpp ./main binary. Entirely offline with a local .gguf model.
        // Example: ./main -m <model.gguf> -p "<prompt>" -n 1024 --temp 0.2
        List<String> cmd = new ArrayList<>();
        // If your binary is named differently or in PATH, adjust "main" accordingly.
        cmd.add("./main");
        cmd.add("-m"); cmd.add(cfg.model);
        cmd.add("-p"); cmd.add(prompt);
        if (cfg.maxTokens != null) {
            cmd.add("-n"); cmd.add(String.valueOf(cfg.maxTokens));
        }
        cmd.add("--temp"); cmd.add(String.valueOf(cfg.temperature));

        return runProcessCaptureStdout(cmd);
    }

    /* ===================== Code Extraction & Utilities ===================== */

    private static Optional<String> extractJavaCode(String text) {
        // Prefer ```java fenced blocks
        Pattern p = Pattern.compile("```\\s*java\\s*(.*?)```", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
        Matcher m = p.matcher(text);
        if (m.find()) return Optional.of(m.group(1).trim());

        // Then any triple-backtick with likely Java
        Pattern p2 = Pattern.compile("```\\s*(.*?)```", Pattern.DOTALL);
        Matcher m2 = p2.matcher(text);
        if (m2.find()) {
            String candidate = m2.group(1).trim();
            if (looksLikeJava(candidate)) return Optional.of(candidate);
        }
        return Optional.empty();
    }

    private static boolean looksLikeJava(String s) {
        return s.contains("class ") || s.contains("public static void main");
    }

    private static String fallbackExtract(String text) {
        int idx = text.indexOf("class ");
        if (idx >= 0) return text.substring(idx).trim();
        return text.trim();
    }

    private static Optional<String> inferPrimaryClassName(String code) {
        Matcher pub = Pattern.compile("\\bpublic\\s+class\\s+(\\w+)").matcher(code);
        if (pub.find()) return Optional.of(pub.group(1));
        Matcher any = Pattern.compile("\\bclass\\s+(\\w+)").matcher(code);
        if (any.find()) return Optional.of(any.group(1));
        Matcher rec = Pattern.compile("\\bpublic\\s+record\\s+(\\w+)").matcher(code);
        if (rec.find()) return Optional.of(rec.group(1));
        return Optional.empty();
    }

    private static int runProcessInheritIO(List<String> cmd) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();
        Process p = pb.start();
        return p.waitFor();
    }

    private static String runProcessCaptureStdout(List<String> cmd) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        try (InputStream in = p.getInputStream()) {
            String out = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            int ec = p.waitFor();
            if (ec != 0 && out.isBlank()) {
                throw new IOException("Process failed (exit " + ec + "): " + String.join(" ", cmd));
            }
            return out;
        }
    }

    private static String stripDotJavaToClass(String fileName) {
        String base = fileName;
        int i = Math.max(base.lastIndexOf('/'), base.lastIndexOf('\\'));
        if (i >= 0) base = base.substring(i + 1);
        if (base.toLowerCase(Locale.ROOT).endsWith(".java")) {
            base = base.substring(0, base.length() - 5);
        }
        return base;
    }

    private static Config parseArgs(String[] args) {
        if (args.length == 0) { printHelp(); return null; }
        Config cfg = new Config();
        List<String> tail = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--help": case "-h": printHelp(); return null;
                case "--engine":
                    ensureHas(args, i + 1, "--engine requires a value");
                    cfg.engine = args[++i];
                    break;
                case "-m":
                    ensureHas(args, i + 1, "-m requires a model or path");
                    cfg.model = args[++i];
                    break;
                case "-o":
                    ensureHas(args, i + 1, "-o requires a filename ending with .java");
                    cfg.outFile = args[++i];
                    break;
                case "-t":
                    ensureHas(args, i + 1, "-t requires a number");
                    cfg.temperature = Double.parseDouble(args[++i]);
                    break;
                case "--max-tokens":
                    ensureHas(args, i + 1, "--max-tokens requires a number");
                    cfg.maxTokens = Integer.parseInt(args[++i]);
                    break;
                case "--compile":
                    cfg.compile = true; break;
                case "--run":
                    cfg.run = true; break;
                case "--raw":
                    cfg.rawPrompt = true; break;
                default:
                    tail.add(a);
            }
        }
        if (tail.isEmpty()) {
            System.err.println("Missing prompt. Example:");
            System.err.println("  java CodeGenAI_Offline \"Write a Java program that prints primes up to 100.\"");
            return null;
        }
        cfg.prompt = String.join(" ", tail);
        return cfg;
    }

    private static void ensureHas(String[] args, int idx, String err) {
        if (idx >= args.length) throw new IllegalArgumentException(err);
    }

    private static void printHelp() {
        System.out.println("CodeGenAI_Offline - local, offline Java code generator");
        System.out.println("Usage: java CodeGenAI_Offline [options] \"your prompt\"");
        System.out.println("Options:");
        System.out.println("  --engine <ollama|llamacpp>   (default ollama)");
        System.out.println("  -m <model_or_path>           ollama: model tag ; llamacpp: .gguf file path");
        System.out.println("  -o <file.java>               output filename; inferred if omitted");
        System.out.println("  -t <temp>                    temperature (default 0.2)");
        System.out.println("  --max-tokens <n>             max tokens to generate");
        System.out.println("  --compile                    compile with javac");
        System.out.println("  --run                        run with java after compile");
        System.out.println("  --raw                        send prompt verbatim (no wrapper)");
        System.out.println("  --help                       show this help");
    }
}

Offline setup tips

Ollama (no network at runtime)

    Make sure the model is already on disk (one-time download was required earlier).

    Test locally:

ollama list
ollama run qwen2.5-coder:7b -p "Hello"

Then:

    javac CodeGenAI_Offline.java
    java CodeGenAI_Offline --engine ollama -m qwen2.5-coder:7b --compile --run \
      "CLI that prints the first 25 Fibonacci numbers"

llama.cpp (pure local binaries & .gguf)

    Build llama.cpp and place your model file locally, e.g. /models/coder.gguf.

    Test:

./main -m /models/coder.gguf -p "Hello"

Then:

    javac CodeGenAI_Offline.java
    java CodeGenAI_Offline --engine llamacpp -m /models/coder.gguf --compile --run \
      "Console app to read lines and sort them"

If you want, I can also add a simple local prompt template file loader, or a sandbox mode that writes preview code without compiling.
You said:
ian@debian:~$ javac CodeGenAI_Offline.java
java CodeGenAI_Offline --engine ollama -m qwen2.5-coder:7b --compile --run \
  "CLI that prints the first 25 Fibonacci numbers"
Wrote: /home/ian/GeneratedProgram.java
/home/ian/GeneratedProgram.java:1: error: class, interface, annotation type, enum, record, method or field expected
Error: unknown shorthand flag: 'p' in -p
^
1 error
Compilation failed (exit 1).
ian@debian:~$ 
ChatGPT said:

I see what happened, Ian — the error wasn’t from the generated Java code at all, but from how my program tried to call ollama run.

On your system, ollama run doesn’t support the shorthand -p "<prompt>". That’s why you got:

Error: unknown shorthand flag: 'p' in -p

Fix

The correct syntax is simply:

ollama run qwen2.5-coder:7b "your prompt here"

No -p flag.
Patch for CodeGenAI_Offline.java

In the runOllama method, replace:

cmd.add("-p");
cmd.add(prompt);

with:

cmd.add(prompt);

So the method becomes:

private static String runOllama(String prompt, Config cfg) throws IOException, InterruptedException {
    List<String> cmd = new ArrayList<>();
    cmd.add("ollama");
    cmd.add("run");
    cmd.add(cfg.model);
    cmd.add(prompt);

    // Engine options (temperature, num tokens, etc.) may or may not be supported in your Ollama build
    // If not supported, remove these lines
    if (cfg.maxTokens != null) {
        cmd.add("--num-predict");
        cmd.add(String.valueOf(cfg.maxTokens));
    }
    cmd.add("--temperature");
    cmd.add(String.valueOf(cfg.temperature));

    return runProcessCaptureStdout(cmd);
}

Recompile & Retry

javac CodeGenAI_Offline.java
java CodeGenAI_Offline --engine ollama -m qwen2.5-coder:7b --compile --run \
  "CLI that prints the first 25 Fibonacci numbers"

👉 Question for you: do you want me to remove all optional flags (--temperature, --num-predict) as well, so it sticks to the absolute minimum (ollama run model "prompt") for maximum compatibility with your local Ollama?
You said:
yes
ChatGPT said:

Great 👍 — here’s the minimal Ollama-compatible version of runOllama.
This uses only:

ollama run <model> "prompt"

(no -p, no --temperature, no --num-predict).
Patch for CodeGenAI_Offline.java

Replace the whole runOllama method with this:

private static String runOllama(String prompt, Config cfg) throws IOException, InterruptedException {
    // Minimal offline-safe command: ollama run <model> "prompt"
    List<String> cmd = new ArrayList<>();
    cmd.add("ollama");
    cmd.add("run");
    cmd.add(cfg.model);
    cmd.add(prompt);

    return runProcessCaptureStdout(cmd);
}

Rebuild & Run

javac CodeGenAI_Offline.java
java CodeGenAI_Offline --engine ollama -m qwen2.5-coder:7b --compile --run \
  "CLI that prints the first 25 Fibonacci numbers"

Would you like me to also show you the full cleaned-up file with that patch applied (so you don’t have to edit manually), or do you prefer just this snippet to drop in?
You said:
full cleaned up file
ChatGPT said:

Got it 👍 — here’s the full cleaned-up CodeGenAI_Offline.java file, with the Ollama runner simplified so it only uses:

ollama run <model> "prompt"

(no extra flags at all).
File: CodeGenAI_Offline.java

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * CodeGenAI_Offline
 *
 * Generate a single-file Java program from an English prompt using a LOCAL LLM engine
 * with ZERO network calls. Supports:
 *   --engine ollama    (default)  -> uses:  ollama run <model> "prompt"
 *   --engine llamacpp             -> uses:  ./main -m <model_path.gguf> -p "prompt"
 *
 * You must have the chosen engine installed locally and a model present on disk.
 *
 * Usage:
 *   javac CodeGenAI_Offline.java
 *   java CodeGenAI_Offline "Print Fibonacci numbers up to 100"
 *
 *   # Choose engine/model, compile, and run:
 *   java CodeGenAI_Offline --engine ollama -m qwen2.5-coder:7b --compile --run \
 *     "CLI that reads N and prints primes to N using Sieve"
 *
 *   # llama.cpp example:
 *   java CodeGenAI_Offline --engine llamacpp -m /models/Qwen2.5-Coder-7B.Q4_K_M.gguf \
 *     --compile --run "Program that sorts lines from stdin"
 */
public class CodeGenAI_Offline {

    static class Config {
        String engine = "ollama";           // or "llamacpp"
        String model = "qwen2.5-coder:7b";  // for llama.cpp, set to a .gguf path
        String outFile = null;
        boolean compile = false;
        boolean run = false;
        boolean rawPrompt = false;
        String prompt = null;
    }

    public static void main(String[] args) {
        try {
            Config cfg = parseArgs(args);
            if (cfg == null) return;

            String fullPrompt = cfg.rawPrompt ? cfg.prompt : buildSystemPrompt(cfg.prompt);

            String modelText = switch (cfg.engine.toLowerCase(Locale.ROOT)) {
                case "ollama"   -> runOllama(fullPrompt, cfg);
                case "llamacpp" -> runLlamaCpp(fullPrompt, cfg);
                default         -> throw new IllegalArgumentException("Unknown engine: " + cfg.engine);
            };

            String code = extractJavaCode(modelText).orElseGet(() -> fallbackExtract(modelText));
            if (code == null || code.isBlank()) {
                System.err.println("No Java code found. Raw model output follows:\n");
                System.out.println(modelText);
                System.exit(2);
            }

            String className = inferPrimaryClassName(code).orElse("GeneratedProgram");
            String fileName = cfg.outFile != null ? cfg.outFile : className + ".java";
            if (!fileName.toLowerCase(Locale.ROOT).endsWith(".java")) {
                System.err.println("Output file must end with .java: " + fileName);
                System.exit(2);
            }

            Path out = Path.of(fileName).toAbsolutePath();
            Files.writeString(out, code, StandardCharsets.UTF_8);
            System.out.println("Wrote: " + out);

            if (cfg.compile) {
                int ec = runProcessInheritIO(List.of("javac", out.toString()));
                if (ec != 0) {
                    System.err.println("Compilation failed (exit " + ec + ").");
                    System.exit(ec);
                } else {
                    System.out.println("Compilation succeeded.");
                }
            }

            if (cfg.run) {
                String mainClass = stripDotJavaToClass(fileName);
                int ec = runProcessInheritIO(List.of("java", mainClass));
                System.exit(ec);
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(2);
        }
    }

    // Wrap the user task so the model emits ONE compilable Java file.
    private static String buildSystemPrompt(String user) {
        StringBuilder sb = new StringBuilder();
        sb.append("You are a code generator that outputs a single complete Java source file only.\n");
        sb.append("Rules:\n");
        sb.append("- Return exactly one file, with a single public class.\n");
        sb.append("- Include a main method when a CLI is implied.\n");
        sb.append("- Use only Java standard library.\n");
        sb.append("- If multiple helpers are needed, use static nested classes.\n");
        sb.append("- If input is needed, use args[] or standard input.\n");
        sb.append("- Prefer readability and correct error handling.\n\n");
        sb.append("Task: ").append(user).append("\n");
        sb.append("Output: One compilable Java source file.\n");
        return sb.toString();
    }

    /* ===================== Engines (Local, Offline) ===================== */

    private static String runOllama(String prompt, Config cfg) throws IOException, InterruptedException {
        // Minimal offline-safe command: ollama run <model> "prompt"
        List<String> cmd = new ArrayList<>();
        cmd.add("ollama");
        cmd.add("run");
        cmd.add(cfg.model);
        cmd.add(prompt);

        return runProcessCaptureStdout(cmd);
    }

    private static String runLlamaCpp(String prompt, Config cfg) throws IOException, InterruptedException {
        // Uses llama.cpp ./main binary. Entirely offline with a local .gguf model.
        // Example: ./main -m <model.gguf> -p "prompt"
        List<String> cmd = new ArrayList<>();
        cmd.add("./main");
        cmd.add("-m"); cmd.add(cfg.model);
        cmd.add("-p"); cmd.add(prompt);

        return runProcessCaptureStdout(cmd);
    }

    /* ===================== Code Extraction & Utilities ===================== */

    private static Optional<String> extractJavaCode(String text) {
        // Prefer ```java fenced blocks
        Pattern p = Pattern.compile("```\\s*java\\s*(.*?)```", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
        Matcher m = p.matcher(text);
        if (m.find()) return Optional.of(m.group(1).trim());

        // Then any triple-backtick with likely Java
        Pattern p2 = Pattern.compile("```\\s*(.*?)```", Pattern.DOTALL);
        Matcher m2 = p2.matcher(text);
        if (m2.find()) {
            String candidate = m2.group(1).trim();
            if (looksLikeJava(candidate)) return Optional.of(candidate);
        }
        return Optional.empty();
    }

    private static boolean looksLikeJava(String s) {
        return s.contains("class ") || s.contains("public static void main");
    }

    private static String fallbackExtract(String text) {
        int idx = text.indexOf("class ");
        if (idx >= 0) return text.substring(idx).trim();
        return text.trim();
    }

    private static Optional<String> inferPrimaryClassName(String code) {
        Matcher pub = Pattern.compile("\\bpublic\\s+class\\s+(\\w+)").matcher(code);
        if (pub.find()) return Optional.of(pub.group(1));
        Matcher any = Pattern.compile("\\bclass\\s+(\\w+)").matcher(code);
        if (any.find()) return Optional.of(any.group(1));
        Matcher rec = Pattern.compile("\\bpublic\\s+record\\s+(\\w+)").matcher(code);
        if (rec.find()) return Optional.of(rec.group(1));
        return Optional.empty();
    }

    private static int runProcessInheritIO(List<String> cmd) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();
        Process p = pb.start();
        return p.waitFor();
    }

    private static String runProcessCaptureStdout(List<String> cmd) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        try (InputStream in = p.getInputStream()) {
            String out = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            int ec = p.waitFor();
            if (ec != 0 && out.isBlank()) {
                throw new IOException("Process failed (exit " + ec + "): " + String.join(" ", cmd));
            }
            return out;
        }
    }

    private static String stripDotJavaToClass(String fileName) {
        String base = fileName;
        int i = Math.max(base.lastIndexOf('/'), base.lastIndexOf('\\'));
        if (i >= 0) base = base.substring(i + 1);
        if (base.toLowerCase(Locale.ROOT).endsWith(".java")) {
            base = base.substring(0, base.length() - 5);
        }
        return base;
    }

    private static Config parseArgs(String[] args) {
        if (args.length == 0) { printHelp(); return null; }
        Config cfg = new Config();
        List<String> tail = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--help": case "-h": printHelp(); return null;
                case "--engine":
                    ensureHas(args, i + 1, "--engine requires a value");
                    cfg.engine = args[++i];
                    break;
                case "-m":
                    ensureHas(args, i + 1, "-m requires a model or path");
                    cfg.model = args[++i];
                    break;
                case "-o":
                    ensureHas(args, i + 1, "-o requires a filename ending with .java");
                    cfg.outFile = args[++i];
                    break;
                case "--compile":
                    cfg.compile = true; break;
                case "--run":
                    cfg.run = true; break;
                case "--raw":
                    cfg.rawPrompt = true; break;
                default:
                    tail.add(a);
            }
        }
        if (tail.isEmpty()) {
            System.err.println("Missing prompt. Example:");
            System.err.println("  java CodeGenAI_Offline \"Write a Java program that prints primes up to 100.\"");
            return null;
        }
        cfg.prompt = String.join(" ", tail);
        return cfg;
    }

    private static void ensureHas(String[] args, int idx, String err) {
        if (idx >= args.length) throw new IllegalArgumentException(err);
    }

    private static void printHelp() {
        System.out.println("CodeGenAI_Offline - local, offline Java code generator");
        System.out.println("Usage: java CodeGenAI_Offline [options] \"your prompt\"");
        System.out.println("Options:");
        System.out.println("  --engine <ollama|llamacpp>   (default ollama)");
        System.out.println("  -m <model_or_path>           ollama: model tag ; llamacpp: .gguf file path");
        System.out.println("  -o <file.java>               output filename; inferred if omitted");
        System.out.println("  --compile                    compile with javac");
        System.out.println("  --run                        run with java after compile");
        System.out.println("  --raw                        send prompt verbatim (no wrapper)");
        System.out.println("  --help                       show this help");
    }
}

✅ Now it will call Ollama exactly as:

ollama run qwen2.5-coder:7b "prompt"

and nothing else.

Do you want me to also make the llama.cpp runner minimal (strip out -n, --temp, etc.) for full symmetry?
You said:
yes
ChatGPT said:

Awesome — the llama.cpp path was already minimal, but here’s the final fully-minimal version again so you can just paste/replace. Both engines now use only the bare commands:

    Ollama: ollama run <model> "prompt"

    llama.cpp: ./main -m <model.gguf> -p "prompt"

File: CodeGenAI_Offline.java

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * CodeGenAI_Offline
 *
 * Generate a single-file Java program from an English prompt using a LOCAL LLM engine
 * with ZERO network calls. Supports:
 *   --engine ollama    (default)  -> uses:  ollama run <model> "prompt"
 *   --engine llamacpp             -> uses:  ./main -m <model_path.gguf> -p "prompt"
 *
 * You must have the chosen engine installed locally and a model present on disk.
 *
 * Usage:
 *   javac CodeGenAI_Offline.java
 *   java CodeGenAI_Offline "Print Fibonacci numbers up to 100"
 *
 *   # Choose engine/model, compile, and run:
 *   java CodeGenAI_Offline --engine ollama -m qwen2.5-coder:7b --compile --run \
 *     "CLI that reads N and prints primes to N using Sieve"
 *
 *   # llama.cpp example:
 *   java CodeGenAI_Offline --engine llamacpp -m /models/Qwen2.5-Coder-7B.Q4_K_M.gguf \
 *     --compile --run "Program that sorts lines from stdin"
 */
public class CodeGenAI_Offline {

    static class Config {
        String engine = "ollama";           // or "llamacpp"
        String model = "qwen2.5-coder:7b";  // for llama.cpp, set to a .gguf path
        String outFile = null;
        boolean compile = false;
        boolean run = false;
        boolean rawPrompt = false;
        String prompt = null;
    }

    public static void main(String[] args) {
        try {
            Config cfg = parseArgs(args);
            if (cfg == null) return;

            String fullPrompt = cfg.rawPrompt ? cfg.prompt : buildSystemPrompt(cfg.prompt);

            String modelText = switch (cfg.engine.toLowerCase(Locale.ROOT)) {
                case "ollama"   -> runOllama(fullPrompt, cfg);
                case "llamacpp" -> runLlamaCpp(fullPrompt, cfg);
                default         -> throw new IllegalArgumentException("Unknown engine: " + cfg.engine);
            };

            String code = extractJavaCode(modelText).orElseGet(() -> fallbackExtract(modelText));
            if (code == null || code.isBlank()) {
                System.err.println("No Java code found. Raw model output follows:\n");
                System.out.println(modelText);
                System.exit(2);
            }

            String className = inferPrimaryClassName(code).orElse("GeneratedProgram");
            String fileName = cfg.outFile != null ? cfg.outFile : className + ".java";
            if (!fileName.toLowerCase(Locale.ROOT).endsWith(".java")) {
                System.err.println("Output file must end with .java: " + fileName);
                System.exit(2);
            }

            Path out = Path.of(fileName).toAbsolutePath();
            Files.writeString(out, code, StandardCharsets.UTF_8);
            System.out.println("Wrote: " + out);

            if (cfg.compile) {
                int ec = runProcessInheritIO(List.of("javac", out.toString()));
                if (ec != 0) {
                    System.err.println("Compilation failed (exit " + ec + ").");
                    System.exit(ec);
                } else {
                    System.out.println("Compilation succeeded.");
                }
            }

            if (cfg.run) {
                String mainClass = stripDotJavaToClass(fileName);
                int ec = runProcessInheritIO(List.of("java", mainClass));
                System.exit(ec);
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(2);
        }
    }

    // Wrap the user task so the model emits ONE compilable Java file.
    private static String buildSystemPrompt(String user) {
        StringBuilder sb = new StringBuilder();
        sb.append("You are a code generator that outputs a single complete Java source file only.\n");
        sb.append("Rules:\n");
        sb.append("- Return exactly one file, with a single public class.\n");
        sb.append("- Include a main method when a CLI is implied.\n");
        sb.append("- Use only Java standard library.\n");
        sb.append("- If multiple helpers are needed, use static nested classes.\n");
        sb.append("- If input is needed, use args[] or standard input.\n");
        sb.append("- Prefer readability and correct error handling.\n\n");
        sb.append("Task: ").append(user).append("\n");
        sb.append("Output: One compilable Java source file.\n");
        return sb.toString();
    }

    /* ===================== Engines (Local, Offline) ===================== */

    private static String runOllama(String prompt, Config cfg) throws IOException, InterruptedException {
        // Minimal offline-safe command: ollama run <model> "prompt"
        List<String> cmd = new ArrayList<>();
        cmd.add("ollama");
        cmd.add("run");
        cmd.add(cfg.model);
        cmd.add(prompt);

        return runProcessCaptureStdout(cmd);
    }

    private static String runLlamaCpp(String prompt, Config cfg) throws IOException, InterruptedException {
        // Minimal offline-safe command: ./main -m <model.gguf> -p "prompt"
        List<String> cmd = new ArrayList<>();
        cmd.add("./main");
        cmd.add("-m"); cmd.add(cfg.model);
        cmd.add("-p"); cmd.add(prompt);

        return runProcessCaptureStdout(cmd);
    }

    /* ===================== Code Extraction & Utilities ===================== */

    private static Optional<String> extractJavaCode(String text) {
        // Prefer ```java fenced blocks
        Pattern p = Pattern.compile("```\\s*java\\s*(.*?)```", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
        Matcher m = p.matcher(text);
        if (m.find()) return Optional.of(m.group(1).trim());

        // Then any triple-backtick with likely Java
        Pattern p2 = Pattern.compile("```\\s*(.*?)```", Pattern.DOTALL);
        Matcher m2 = p2.matcher(text);
        if (m2.find()) {
            String candidate = m2.group(1).trim();
            if (looksLikeJava(candidate)) return Optional.of(candidate);
        }
        return Optional.empty();
    }

    private static boolean looksLikeJava(String s) {
        return s.contains("class ") || s.contains("public static void main");
    }

    private static String fallbackExtract(String text) {
        int idx = text.indexOf("class ");
        if (idx >= 0) return text.substring(idx).trim();
        return text.trim();
    }

    private static Optional<String> inferPrimaryClassName(String code) {
        Matcher pub = Pattern.compile("\\bpublic\\s+class\\s+(\\w+)").matcher(code);
        if (pub.find()) return Optional.of(pub.group(1));
        Matcher any = Pattern.compile("\\bclass\\s+(\\w+)").matcher(code);
        if (any.find()) return Optional.of(any.group(1));
        Matcher rec = Pattern.compile("\\bpublic\\s+record\\s+(\\w+)").matcher(code);
        if (rec.find()) return Optional.of(rec.group(1));
        return Optional.empty();
    }

    private static int runProcessInheritIO(List<String> cmd) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();
        Process p = pb.start();
        return p.waitFor();
    }

    private static String runProcessCaptureStdout(List<String> cmd) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        try (InputStream in = p.getInputStream()) {
            String out = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            int ec = p.waitFor();
            if (ec != 0 && out.isBlank()) {
                throw new IOException("Process failed (exit " + ec + "): " + String.join(" ", cmd));
            }
            return out;
        }
    }

    private static String stripDotJavaToClass(String fileName) {
        String base = fileName;
        int i = Math.max(base.lastIndexOf('/'), base.lastIndexOf('\\'));
        if (i >= 0) base = base.substring(i + 1);
        if (base.toLowerCase(Locale.ROOT).endsWith(".java")) {
            base = base.substring(0, base.length() - 5);
        }
        return base;
    }

    private static Config parseArgs(String[] args) {
        if (args.length == 0) { printHelp(); return null; }
        Config cfg = new Config();
        List<String> tail = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--help": case "-h": printHelp(); return null;
                case "--engine":
                    ensureHas(args, i + 1, "--engine requires a value");
                    cfg.engine = args[++i];
                    break;
                case "-m":
                    ensureHas(args, i + 1, "-m requires a model or path");
                    cfg.model = args[++i];
                    break;
                case "-o":
                    ensureHas(args, i + 1, "-o requires a filename ending with .java");
                    cfg.outFile = args[++i];
                    break;
                case "--compile":
                    cfg.compile = true; break;
                case "--run":
                    cfg.run = true; break;
                case "--raw":
                    cfg.rawPrompt = true; break;
                default:
                    tail.add(a);
            }
        }
        if (tail.isEmpty()) {
            System.err.println("Missing prompt. Example:");
            System.err.println("  java CodeGenAI_Offline \"Write a Java program that prints primes up to 100.\"");
            return null;
        }
        cfg.prompt = String.join(" ", tail);
        return cfg;
    }

    private static void ensureHas(String[] args, int idx, String err) {
        if (idx >= args.length) throw new IllegalArgumentException(err);
    }

    private static void printHelp() {
        System.out.println("CodeGenAI_Offline - local, offline Java code generator");
        System.out.println("Usage: java CodeGenAI_Offline [options] \"your prompt\"");
        System.out.println("Options:");
        System.out.println("  --engine <ollama|llamacpp>   (default ollama)");
        System.out.println("  -m <model_or_path>           ollama: model tag ; llamacpp: .gguf file path");
        System.out.println("  -o <file.java>               output filename; inferred if omitted");
        System.out.println("  --compile                    compile with javac");
        System.out.println("  --run                        run with java after compile");
        System.out.println("  --raw                        send prompt verbatim (no wrapper)");
        System.out.println("  --help                       show this help");
    }
}

Quick test

javac CodeGenAI_Offline.java

# Ollama (model already on disk)
java CodeGenAI_Offline --engine ollama -m qwen2.5-coder:7b --compile --run \
  "CLI that prints the first 25 Fibonacci numbers"

# llama.cpp (adjust path to your gguf + binary)
java CodeGenAI_Offline --engine llamacpp -m /models/coder.gguf --compile --run \
  "Program that reads N and prints primes up to N"

If you hit any engine-specific quirks (e.g., different binary name or path), tell me what you see and I’ll tweak the launcher.
You said:
code in python3
ChatGPT said:

Sweet — here’s a fully offline Python 3 version that does the same job: it prompts a local LLM (Ollama or llama.cpp), extracts a single Java source file, writes it to disk, and can compile and run it.
File: codegen_ai_offline.py

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

Usage

# Make it executable (optional)
chmod +x codegen_ai_offline.py

# Ollama (model already on disk)
./codegen_ai_offline.py --engine ollama -m qwen2.5-coder:7b --compile --run \
  "CLI that prints the first 25 Fibonacci numbers"

# llama.cpp (adjust paths to your binary and .gguf)
./codegen_ai_offline.py --engine llamacpp -m /models/Qwen2.5-Coder-7B.Q4_K_M.gguf --compile --run \
  "Program that reads N and prints primes up to N"

If your llama.cpp binary isn’t ./main or it lives elsewhere, run it from that directory or edit the call_llamacpp() command to point to the right binary.


ChatGPT can make mistakes. Check important info.

