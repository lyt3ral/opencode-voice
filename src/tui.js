// V2 TUI entry for opencode-voice: voice input only (no TTS).
//
// V1 used `{ id, tui(api, options) }` with `api.command.register`, `api.kv`,
// `api.ui.dialog.replace` and a server-side `client.tui.appendPrompt()`.
// V2 requires `Plugin.define` from `@opencode/plugin/tui`; prompt injection is
// now an RPC event rather than a client method.

import { Plugin } from "@opencode/plugin/tui";
import fs from "node:fs";
import { spawn, execFileSync } from "node:child_process";

const MODELS_DIRS = [
  [process.env.HOME, ".local", "share", "whisper-cpp"].join("/"),
  "/usr/local/share/whisper-cpp/models",
];

const MODELS = {
  "large-v3-turbo-q5_0": {
    label: "Large v3 Turbo Q5 (most accurate)",
    file: "ggml-large-v3-turbo-q5_0.bin",
  },
  "medium-q5_0": { label: "Medium Q5 (multilingual)", file: "ggml-medium-q5_0.bin" },
  "small-q5_1": { label: "Small Q5_1 (multilingual, balanced)", file: "ggml-small-q5_1.bin" },
  small: { label: "Small multilingual", file: "ggml-small.bin" },
  "small.en": { label: "Small English", file: "ggml-small.en.bin" },
  "base.en": { label: "Base English (fastest)", file: "ggml-base.en.bin" },
  base: { label: "Base multilingual (fast)", file: "ggml-base.bin" },
  "tiny.en": { label: "Tiny English", file: "ggml-tiny.en.bin" },
  tiny: { label: "Tiny multilingual", file: "ggml-tiny.bin" },
};
const DEFAULT_MODEL = "base.en";

const LANGUAGES = {
  auto: "Auto-detect",
  en: "English",
  zh: "Chinese",
  ja: "Japanese",
  ko: "Korean",
  de: "German",
  fr: "French",
  es: "Spanish",
  pt: "Portuguese",
  ru: "Russian",
  it: "Italian",
};

const RECORD_RATE = 16000;
const TRANSCRIBE_TIMEOUT_MS = 60000;

// Capture devices, minus the .monitor sources (playback taps, not mics).
function listInputDevices() {
  try {
    const out = execFileSync("pactl", ["list", "sources", "short"], {
      encoding: "utf-8",
      timeout: 5000,
    });
    return out
      .split("\n")
      .map((l) => l.trim().split(/\s+/))
      .filter((p) => p[1] && !p[1].endsWith(".monitor"))
      .map((p) => ({ name: p[1], label: p[1] }));
  } catch {
    return [];
  }
}

export default Plugin.define({
  id: "opencode-voice",

  setup(ctx) {
    const opts = ctx.options ?? {};
    const log = (scope, msg, level = "debug") => {
      try {
        ctx.client.log?.({ level, scope, message: `[opencode-voice] ${msg}` });
      } catch {}
    };

    const [settings, update] = ctx.storage.store("opencode-voice", {
      initial: {
        model: opts.model ?? DEFAULT_MODEL,
        language: opts.language ?? "auto",
        mic: opts.mic ?? "",
        trimSilence: opts.trimSilence ?? true,
        lastTranscript: "",
      },
    });

    const toast = (message, variant = "info") =>
      ctx.ui.toast.show({ message, variant, duration: 3000 });

    const tmpDir = opts.tmpDir || "/tmp";
    const wav = `${tmpDir}/opencode-voice-stt.wav`;

    // ---- model / language / mic resolution ----

    const modelsDir = () => {
      for (const dir of MODELS_DIRS) {
        try {
          if (fs.existsSync(dir)) return dir;
        } catch {}
      }
      return MODELS_DIRS[0];
    };

    const modelEntry = () => MODELS[settings.model] ?? MODELS[DEFAULT_MODEL];
    const modelPath = () => `${modelsDir()}/${modelEntry().file}`;

    // ---- recording state ----

    let rec = null;
    let recording = false;
    let processing = false;

    function killRecorder() {
      if (rec) {
        try {
          rec.kill("SIGKILL");
        } catch {}
        rec = null;
      }
    }

    function startRecording() {
      if (rec) return;
      try {
        fs.unlinkSync(wav);
      } catch {}
      // PipeWire-native capture; falls back to sox when pw-record is absent.
      // pw-record defaults are wrong for STT: Playback category, 48 kHz,
      // stereo. Whisper needs 16 kHz mono capture from a Capture-category node.
      const args = [
        "--media-category",
        "Capture",
        "--media-role",
        "Media",
        "--rate",
        String(RECORD_RATE),
        "--channels",
        "1",
        "--format",
        "s16",
        "--target",
        settings.mic || "auto",
        wav,
      ];
      const cmd = opts.recorder ?? "pw-record";
      log("STT", `record start cmd=${cmd} mic=${settings.mic || "auto"}`);
      rec = spawn(cmd, args, {
        stdio: ["ignore", "ignore", "pipe"],
      });
      let stderr = "";
      rec.stderr?.on("data", (c) => {
        stderr += c.toString();
      });
      rec.on("error", (err) => {
        rec = null;
        recording = false;
        toast(`Recording failed: ${err.message}`, "error");
      });
      rec.on("exit", (code) => {
        rec = null;
        if (recording && code !== 0 && code !== null && !processing) {
          recording = false;
          toast(`Recording error (${code})`, "error");
          log("STT", `recorder exit ${code}: ${stderr.trim().slice(0, 200)}`, "warn");
        }
      });
      recording = true;
    }

    function stopRecording() {
      // SIGINT makes pw-record finalise the WAV header.
      if (rec) rec.kill("SIGINT");
    }

    // ---- transcription ----

    function transcribe() {
      const mp = modelPath();
      if (!fs.existsSync(mp)) {
        return Promise.resolve({
          error: `Model not found: ${settings.model} (${mp}). Download it from huggingface.co/ggerganov/whisper.cpp`,
        });
      }
      if (!fs.existsSync(wav)) {
        return Promise.resolve({ error: "No recording file - recorder may have failed" });
      }
      if (fs.statSync(wav).size <= 44) {
        return Promise.resolve({ error: "Recording is empty - no audio captured" });
      }

      return new Promise((resolve) => {
        let stdout = "";
        let stderr = "";
        const proc = spawn(
          "whisper-cli",
          ["-m", mp, "-f", wav, "-l", settings.language || "auto", "-np", "-nt"],
          { stdio: ["ignore", "pipe", "pipe"] },
        );
        proc.stdout.on("data", (c) => {
          stdout += c.toString();
        });
        proc.stderr.on("data", (c) => {
          stderr += c.toString();
        });
        const timer = setTimeout(() => {
          proc.kill("SIGKILL");
          resolve({ error: "Transcription timed out (60s)" });
        }, TRANSCRIBE_TIMEOUT_MS);
        proc.on("error", (err) => {
          clearTimeout(timer);
          resolve({ error: `whisper-cli failed: ${err.message}` });
        });
        proc.on("exit", (code) => {
          clearTimeout(timer);
          const langErr = stderr.match(/error: unknown language '([^']+)'/);
          if (langErr) return resolve({ error: `Unknown language: ${langErr[1]}` });
          if (code !== 0) {
            return resolve({
              error: stderr.trim().split("\n").pop() || `whisper-cli exited (${code})`,
            });
          }
          resolve({
            text: stdout
              .replace(/\[.*?\]/g, "")
              .replace(/\(.*?\)/g, "")
              .replace(/\s+/g, " ")
              .trim(),
          });
        });
      });
    }

    // ---- delivery ----
    // V2 gives a plugin no way to write into the composer. The
    // `tui.prompt.append` event type has no matching RPC method - the server
    // answers "RPC is unavailable: tui.prompt.append" - and the TUI Context
    // exposes slots for rendering but nothing for injection. So a transcript is
    // sent as a prompt directly rather than staged in the composer for review.
    // Delivery is therefore explicit: "send" submits, "clipboard" leaves the
    // text for the user to paste.

    function currentSessionID() {
      const route = ctx.ui.router.current();
      return route?.type === "session" ? route.sessionID : null;
    }

    // The wire shape is a bare string. SessionPromptInput renders these fields
    // via indexed-access types ({...}["text"]) in the .d.ts, which reads like an
    // object but resolves to a plain string; the server rejects { text } with
    // "Expected string at [\"text\"]". openapi.json is the authority here.
    async function send(text) {
      const sessionID = currentSessionID();
      if (!sessionID) throw new Error("No active session - open one first");
      const res = await ctx.client.session.prompt({ sessionID, text });
      if (res?.error) throw new Error(res.error.data?.message ?? res.error.name);
    }

    function copyToClipboard(text) {
      return new Promise((resolve, reject) => {
        const p = spawn("wl-copy", [], { stdio: ["pipe", "ignore", "ignore"] });
        p.on("error", reject);
        p.on("close", (code) =>
          code === 0 ? resolve() : reject(new Error(`wl-copy exited (${code})`)),
        );
        p.stdin.write(text);
        p.stdin.end();
      });
    }

    // ---- pipeline ----

    async function run(deliver) {
      if (processing) return toast("Busy, please wait...");
      processing = true;
      try {
        stopRecording();
        // give the recorder a moment to finalise the WAV header
        for (let i = 0; i < 20 && rec; i++) {
          await new Promise((r) => setTimeout(r, 100));
        }
        recording = false;

        toast("Transcribing...");
        const result = await transcribe();
        if (result.error) return toast(result.error, "error");
        if (!result.text) return toast("No speech detected", "warning");

        await update((draft) => {
          draft.lastTranscript = result.text;
        });
        if (deliver === "clipboard") {
          await copyToClipboard(result.text);
          toast("Copied to clipboard - paste where you want it", "success");
        } else {
          await send(result.text);
          toast("Sent", "success");
        }
      } catch (err) {
        toast(`STT error: ${err.message}`, "error");
        log("STT", `pipeline error: ${err.message}`, "error");
      } finally {
        processing = false;
        recording = false;
      }
    }

    // ---- commands ----

    const commands = [
      {
        id: "opencode-voice.record",
        title: "Voice: record and send",
        group: "Voice",
        description: "Toggle recording; press again to transcribe and send as a prompt",
        bind: "ctrl+y",
        palette: true,
        slash: { name: "voice" },
        run: () => {
          if (processing) return toast("Busy, please wait...");
          if (recording) {
            toast("Stopping...");
            return void run("send");
          }
          startRecording();
          if (recording) toast("Recording... press again to send");
        },
      },
      {
        id: "opencode-voice.copy",
        title: "Voice: record to clipboard",
        group: "Voice",
        description: "Stop recording, transcribe, and copy to the clipboard instead of sending",
        bind: "<leader>r",
        palette: true,
        run: () => {
          if (!recording) return toast("Not recording", "warning");
          void run("clipboard");
        },
      },
      {
        id: "opencode-voice.cancel",
        title: "Voice: cancel recording",
        group: "Voice",
        description: "Discard the current recording",
        palette: true,
        slash: { name: "voice-stop" },
        run: () => {
          if (!recording) return;
          recording = false;
          killRecorder();
          toast("Cancelled");
        },
      },
      {
        id: "opencode-voice.model",
        title: "Voice: select model",
        group: "Voice",
        palette: true,
        slash: { name: "voice-model" },
        run: async () => {
          const picked = await ctx.ui.dialog.select({
            title: "Whisper model",
            current: settings.model,
            options: Object.entries(MODELS).map(([value, v]) => ({
              title: v.label,
              value,
            })),
          });
          if (picked) {
            // update() takes a mutation callback that receives the store draft, not a
            // patch object - passing an object silently did nothing, so the model,
            // language and mic selections never persisted.
            await update((draft) => {
              draft.model = picked;
            });
            toast(`Model: ${MODELS[picked].label}`);
          }
        },
      },
      {
        id: "opencode-voice.language",
        title: "Voice: select language",
        group: "Voice",
        palette: true,
        slash: { name: "voice-language" },
        run: async () => {
          const picked = await ctx.ui.dialog.select({
            title: "Transcription language",
            current: settings.language,
            options: Object.entries(LANGUAGES).map(([value, label]) => ({
              title: label,
              value,
            })),
          });
          if (picked) {
            await update((draft) => {
              draft.language = picked;
            });
            toast(`Language: ${LANGUAGES[picked]}`);
          }
        },
      },
      {
        id: "opencode-voice.mic",
        title: "Voice: select microphone",
        group: "Voice",
        palette: true,
        slash: { name: "voice-mic" },
        run: async () => {
          const devices = listInputDevices();
          if (devices.length === 0) return toast("No input devices found", "warning");
          const picked = await ctx.ui.dialog.select({
            title: "Microphone",
            current: settings.mic,
            options: [
              { title: "System default", value: "" },
              ...devices.map((d) => ({ title: d.label, value: d.name })),
            ],
          });
          if (picked !== undefined) {
            await update((draft) => {
              draft.mic = picked;
            });
            toast(`Mic: ${picked || "system default"}`);
          }
        },
      },
      {
        id: "opencode-voice.last",
        title: "Voice: copy last transcript",
        group: "Voice",
        palette: true,
        description: "Put the most recent transcript back on the clipboard",
        run: async () => {
          if (!settings.lastTranscript) return toast("No transcript yet", "warning");
          await copyToClipboard(settings.lastTranscript);
          toast("Copied to clipboard");
        },
      },
    ];

    // ctx.keymap.layer needs a Solid component context, so it is registered from
    // inside an app-slot render function; calling it directly from setup()
    // registers the bindings but silently drops palette and slash entries.
    // mode "global" opts out of the default "base" input mode, otherwise the
    // layer is not reachable from the prompt's slash completion.
    const stopSlot = ctx.ui.slot({
      append: "app",
      render: () => {
        ctx.keymap.layer(() => ({
          mode: "global",
          priority: 100,
          commands,
          bindings: ["opencode-voice.record", "opencode-voice.copy"],
        }));
        return null;
      },
    });

    ctx.data.on("session.idle", () => {
      killRecorder();
    });

    log("STT", `registered ${commands.length} commands, model=${settings.model}`);
    return () => {
      stopSlot();
      killRecorder();
    };
  },
});
