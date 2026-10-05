// Mac parity check for converted .litertlm models — generate text on the Mac
// (no device push/launch), so we can iterate on the "degenerate output" bug fast.
//
//   swift run litert-mac-verify <model.litertlm> ["prompt"] [--max-tokens N]
//
// --max-tokens (default 256) lets reasoning models (<think>) finish their chain
// of thought and emit the actual answer instead of being truncated mid-thought.

import Foundation
import LiteRTFoundation
import LiteRTLM

// Parse: <model> [prompt] with an optional `--max-tokens N` / `-n N` flag anywhere.
var positional: [String] = []
var maxTokens = 256
var engineBackend: String? = nil  // --backend cpu|gpu → run via low-level Engine API
var engineTemp: Float = 0.0  // --temp T → sampler temperature for the Engine path (models whose greedy decode collapses, e.g. RL tunes, need their official sampling)
var engineTopK: Int = 40  // --topk K → sampler top-k for the Engine path
var greedy = false  // --greedy → temperature 0 (match HF do_sample=False) instead of LiteRTChat's default temp 0.8
var thinking = false  // --thinking → ThinkingConfig(enableThinking: true, budget -1); v0.15.0+ runtime only
var runs = 1  // --runs N → repeat generation N times in-process (run 1 = cold, 2+ = warm engine)
// Fidelity-suite batch mode (apple-silicon-llm-bench evaldata/fidelity_suite_v1.jsonl):
//   --suite <jsonl> --out <jsonl> --backend cpu|gpu [--reps N] [--tier smoke|core|full]
//   [--suite-max-num-tokens M] [--resume] [--timeout S]
// One long-lived Engine, greedy sampler, one conversation per (case × rep); every
// generation appended to --out as a JSON line. maxNumTokens is TOTAL CONTEXT for this
// runtime — undersizing corrupts output instead of truncating — so it is sized from the
// largest included case (est × 1.35 + budget + 64) unless overridden.
var suitePath: String? = nil
var suiteOutPath: String? = nil
var suiteReps = 3
var suiteTier = "smoke"
var suiteMaxNumTokens: Int? = nil
var suiteResume = false
var suiteTimeout: Double = 300
var suiteRuntimeVersion: String? = nil  // --runtime-version V → recorded per generation
var suiteActivation: String? = nil  // --activation f32 → FLOAT32 activations; recorded backend becomes "<backend>-f32"
var i = 1
let argv = CommandLine.arguments
while i < argv.count {
  let a = argv[i]
  if a == "--max-tokens" || a == "-n", i + 1 < argv.count {
    maxTokens = Int(argv[i + 1]) ?? maxTokens
    i += 2
  } else if a == "--runs", i + 1 < argv.count {
    runs = Int(argv[i + 1]) ?? runs
    i += 2
  } else if a == "--backend", i + 1 < argv.count {
    engineBackend = argv[i + 1]
    i += 2
  } else if a == "--temp", i + 1 < argv.count {
    engineTemp = Float(argv[i + 1]) ?? engineTemp
    i += 2
  } else if a == "--topk", i + 1 < argv.count {
    engineTopK = Int(argv[i + 1]) ?? engineTopK
    i += 2
  } else if a == "--suite", i + 1 < argv.count {
    suitePath = argv[i + 1]
    i += 2
  } else if a == "--out", i + 1 < argv.count {
    suiteOutPath = argv[i + 1]
    i += 2
  } else if a == "--reps", i + 1 < argv.count {
    suiteReps = Int(argv[i + 1]) ?? suiteReps
    i += 2
  } else if a == "--tier", i + 1 < argv.count {
    suiteTier = argv[i + 1]
    i += 2
  } else if a == "--suite-max-num-tokens", i + 1 < argv.count {
    suiteMaxNumTokens = Int(argv[i + 1])
    i += 2
  } else if a == "--runtime-version", i + 1 < argv.count {
    suiteRuntimeVersion = argv[i + 1]
    i += 2
  } else if a == "--activation", i + 1 < argv.count {
    suiteActivation = argv[i + 1]
    i += 2
  } else if a == "--resume" {
    suiteResume = true
    i += 1
  } else if a == "--timeout", i + 1 < argv.count {
    suiteTimeout = Double(argv[i + 1]) ?? suiteTimeout
    i += 2
  } else if a == "--greedy" {
    greedy = true
    i += 1
  } else if a == "--thinking" {
    thinking = true
    i += 1
  } else {
    positional.append(a)
    i += 1
  }
}
guard positional.count >= 1 else {
  FileHandle.standardError.write(
    Data("usage: litert-mac-verify <model.litertlm> [prompt] [--max-tokens N]\n".utf8))
  exit(2)
}
let modelPath = positional[0]
let prompt = positional.count >= 2 ? positional[1] : "Explain on-device AI in one short sentence."

// ---------------------------------------------------------------------------
// Fidelity-suite batch mode. Requires --backend. Exit 3 = a generation timed out
// (treat the engine as poisoned — the #2814 class hangs permanently; rerun with
// --resume to continue from the recorded position).
if let sp = suitePath {
  guard let bk = engineBackend else {
    FileHandle.standardError.write(Data("--suite requires --backend cpu|gpu\n".utf8))
    exit(2)
  }
  guard let outPath = suiteOutPath else {
    FileHandle.standardError.write(Data("--suite requires --out <jsonl>\n".utf8))
    exit(2)
  }
  struct SuiteCase {
    let id: String
    let prompt: String
    let genBudget: Int
    let estTokens: Int
  }
  // tier filter: smoke ⊂ core ⊂ full
  let tierRank = ["smoke": 0, "core": 1, "full": 2]
  guard let wantRank = tierRank[suiteTier] else {
    FileHandle.standardError.write(Data("--tier must be smoke|core|full\n".utf8))
    exit(2)
  }
  var cases: [SuiteCase] = []
  guard let suiteData = try? String(contentsOfFile: sp, encoding: .utf8) else {
    FileHandle.standardError.write(Data("cannot read suite \(sp)\n".utf8))
    exit(2)
  }
  for line in suiteData.split(separator: "\n") {
    guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
      let cid = obj["id"] as? String,
      let prompt = obj["prompt"] as? String
    else { continue }
    let tier = obj["tier"] as? String ?? "full"
    guard let r = tierRank[tier], r <= wantRank else { continue }
    cases.append(
      SuiteCase(
        id: cid, prompt: prompt,
        genBudget: obj["max_tokens"] as? Int ?? 64,
        estTokens: obj["est_prompt_tokens"] as? Int ?? 1024))
  }
  guard !cases.isEmpty else {
    FileHandle.standardError.write(Data("suite has no cases at tier \(suiteTier)\n".utf8))
    exit(2)
  }
  // maxNumTokens is TOTAL CONTEXT here; undersizing corrupts (bench CLAUDE.md rule 3).
  let neededTokens =
    suiteMaxNumTokens
    ?? cases.map { Int(Double($0.estTokens) * 1.35) + $0.genBudget + 64 }.max()!
  let engineMaxTokens = ((neededTokens + 127) / 128) * 128
  // resume: skip (case_id, rep) pairs already recorded
  var done = Set<String>()
  if suiteResume, let existing = try? String(contentsOfFile: outPath, encoding: .utf8) {
    for line in existing.split(separator: "\n") {
      if let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
        let cid = obj["case_id"] as? String, let rep = obj["rep"] as? Int
      {
        done.insert("\(cid)#\(rep)")
      }
    }
  }
  if !FileManager.default.fileExists(atPath: outPath) {
    FileManager.default.createFile(atPath: outPath, contents: nil)
  }
  guard let outHandle = FileHandle(forWritingAtPath: outPath) else {
    FileHandle.standardError.write(Data("cannot open out \(outPath)\n".utf8))
    exit(2)
  }
  outHandle.seekToEndOfFile()
  let modelName = (modelPath as NSString).lastPathComponent
  if let act = suiteActivation {
    guard act == "f32" else {
      FileHandle.standardError.write(Data("--activation must be f32\n".utf8))
      exit(2)
    }
    liteRTLMActivationDataTypeOverride = 0
  }
  // Scorer arms are keyed on the backend label, so a non-default activation type gets its own.
  var backendLabel = suiteActivation.map { "\(bk)-\($0)" } ?? bk
  // An explicit context size is a separate arm too (context-sizing control).
  if let m = suiteMaxNumTokens { backendLabel += "-ctx\(m)" }

  struct SuiteTimeout: Error {}
  func generate(_ conv: Conversation, _ prompt: String, _ deadline: Double) async throws -> (
    String, Int
  ) {
    try await withThrowingTaskGroup(of: (String, Int)?.self) { group in
      group.addTask {
        var acc = ""
        var yields = 0
        for try await chunk in conv.sendMessageStream(Message(prompt)) {
          acc += chunk.toString
          yields += 1
        }
        return (acc, yields)
      }
      group.addTask {
        try await Task.sleep(nanoseconds: UInt64(deadline * 1e9))
        return nil
      }
      guard let first = try await group.next()!, case let (acc, yields) = first else {
        group.cancelAll()
        throw SuiteTimeout()
      }
      group.cancelAll()
      return (acc, yields)
    }
  }

  let semS = DispatchSemaphore(value: 0)
  var exitCode: Int32 = 0
  Task {
    do {
      ExperimentalFlags.optIntoExperimentalAPIs()
      ExperimentalFlags.enableBenchmark = true
      let backend: Backend = (bk == "cpu") ? .cpu() : .gpu
      let t0 = Date()
      let config = try EngineConfig(
        modelPath: modelPath, backend: backend, maxNumTokens: engineMaxTokens)
      let engine = Engine(engineConfig: config)
      try await engine.initialize()
      print(
        String(
          format: "suite[\(bk)] engine init %.1fs · maxNumTokens \(engineMaxTokens) · %d cases × %d reps",
          Date().timeIntervalSince(t0), cases.count, suiteReps))
      let sampler = try SamplerConfig(topK: 1, topP: 1.0, temperature: 0.0)  // greedy: fidelity protocol
      var engineSeq = 0
      for c in cases {
        for rep in 1...suiteReps {
          if done.contains("\(c.id)#\(rep)") { continue }
          engineSeq += 1
          var record: [String: Any] = [
            "suite": "fidelity-v1", "case_id": c.id, "runtime": "litert-lm",
            "model": modelName, "backend": backendLabel, "device": "mac",
            "rep": rep, "engine_seq": engineSeq,
            "ts": ISO8601DateFormatter().string(from: Date()),
          ]
          if let v = suiteRuntimeVersion { record["runtime_version"] = v }
          record["max_num_tokens"] = engineMaxTokens
          do {
            let conv = try await engine.createConversation(
              with: ConversationConfig(samplerConfig: sampler))
            let tg = Date()
            let (acc, yields) = try await generate(conv, c.prompt, suiteTimeout)
            record["output"] = acc
            record["gen_s"] = Date().timeIntervalSince(tg)
            record["stream_yields"] = yields
            if let b = try? conv.getBenchmarkInfo() {
              record["decode_tokens"] = b.lastDecodeTokenCount
              record["prefill_tokens"] = b.lastPrefillTokenCount
              record["prefill_tps"] = b.lastPrefillTokensPerSecond
              record["decode_tps"] = b.lastDecodeTokensPerSecond
            }
          } catch is SuiteTimeout {
            record["output"] = ""
            record["error"] = "TIMEOUT after \(suiteTimeout)s — engine treated as poisoned"
            let data = try! JSONSerialization.data(withJSONObject: record)
            outHandle.write(data)
            outHandle.write(Data("\n".utf8))
            print("TIMEOUT at \(c.id) rep \(rep) — exiting 3; rerun with --resume")
            exitCode = 3
            semS.signal()
            return
          } catch {
            record["output"] = ""
            record["error"] = "\(error)"
          }
          let data = try! JSONSerialization.data(withJSONObject: record)
          outHandle.write(data)
          outHandle.write(Data("\n".utf8))
          if engineSeq % 10 == 0 { print("  …\(engineSeq) generations (at \(c.id))") }
        }
      }
      print("suite[\(bk)] done → \(outPath)")
    } catch {
      print("SUITE FAILED[\(bk)]: \(error)")
      exitCode = 1
    }
    semS.signal()
  }
  semS.wait()
  exit(exitCode)
}

// Low-level Engine path with an explicit backend (--backend cpu|gpu) — used to
// localize where a recipe runs vs hangs (e.g. weight-only FLOAT compute).
if let bk = engineBackend {
  let backend: Backend = (bk == "cpu") ? .cpu() : .gpu
  let semE = DispatchSemaphore(value: 0)
  Task {
    do {
      ExperimentalFlags.optIntoExperimentalAPIs()
      ExperimentalFlags.enableBenchmark = true
      let t0 = Date()
      let config = try EngineConfig(modelPath: modelPath, backend: backend, maxNumTokens: maxTokens)
      let engine = Engine(engineConfig: config)
      try await engine.initialize()
      print(String(format: "engine[\(bk)] init %.1fs", Date().timeIntervalSince(t0)))
      let sampler = try SamplerConfig(topK: engineTopK, topP: 0.95, temperature: engineTemp)
      for r in 1...runs {
        let conv = try await engine.createConversation(with: ConversationConfig(samplerConfig: sampler))
        var yields = 0
        var acc = ""
        for try await chunk in conv.sendMessageStream(Message(prompt)) {
          yields += 1
          acc += chunk.toString
        }
        let oneLine = acc.replacingOccurrences(of: "\n", with: "⏎")
        if r == 1 { print("OUTPUT: [\(oneLine)]") }
        let b = try conv.getBenchmarkInfo()
        print("OK[\(bk)] run=\(r) cold=\(r == 1 ? 1 : 0) decode_tokens=\(b.lastDecodeTokenCount) stream_yields=\(yields)")
        print(String(format: "run %d: decode %.1f tok/s · prefill %.1f tok/s", r,
          b.lastDecodeTokensPerSecond, b.lastPrefillTokensPerSecond))
      }
    } catch {
      print("FAILED[\(bk)]: \(error)")
    }
    semE.signal()
  }
  semE.wait()
  exit(0)
}

let sem = DispatchSemaphore(value: 0)
Task {
  do {
    let t0 = Date()
    let chat = try await LiteRTChat(
      modelFileURL: URL(fileURLWithPath: modelPath),
      modalities: [] as Modality,
      maxTokens: maxTokens,
      enableBenchmark: true,
      sampler: greedy ? (try SamplerConfig(topK: 1, topP: 1.0, temperature: 0.0)) : nil,
      prewarm: false,
      thinking: thinking ? ThinkingConfig(enableThinking: true, thinkingTokenBudget: -1) : nil)
    print(String(format: "loaded in %.1fs", Date().timeIntervalSince(t0)))
    let resp = try await chat.respond(prompt)
    // brackets + single-line so an empty/whitespace response is visible and greppable
    let oneLine = resp.replacingOccurrences(of: "\n", with: "⏎")
    print("OUTPUT: [\(oneLine)]")
    if let b = try? chat.lastBenchmark() {
      print(String(format: "decode %.1f tok/s · prefill %.1f tok/s",
        b.lastDecodeTokensPerSecond, b.lastPrefillTokensPerSecond))
    }
  } catch {
    print("FAILED: \(error)")
  }
  sem.signal()
}
sem.wait()
