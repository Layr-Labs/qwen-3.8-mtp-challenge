// Hilbert Pi/OMP trace transport — hilbert-pi-family-trace-extension-v1.
//
// Pi and OMP sessions are native append-only JSONL files. This extension forwards low-frequency
// lifecycle metadata; Hilbert owns transcript reading, redaction, checkpoints, retry, and upload.
import { spawn } from "node:child_process";

const WRAPPER = ".hilbert/hooks/hilbert-trace.sh";

type PiContext = {
  cwd: string;
  sessionManager: {
    getHeader(): { id: string } | null;
    getSessionFile(): string | undefined;
  };
};

type PiExtension = {
  pi?: {
    settings: {
      getShellConfig(): { env: Record<string, string> };
    };
  };
  on(
    event: "session_start" | "before_agent_start" | "agent_end" | "session_shutdown",
    handler: (event: unknown, context: PiContext) => Promise<void>,
  ): void;
};

export default function HilbertTrace(pi: PiExtension): void {
  const agent = pi.pi === undefined ? "pi" : "omp";
  const rootSessionId = agent === "pi" ? "YUKON_PI_SESSION_ID" : "YUKON_OMP_SESSION_ID";
  let parentSessionId: string | undefined;

  pi.on("session_start", async (_event, context) => {
    const session = currentSession(context);
    if (session === null) return;
    parentSessionId = process.env[rootSessionId];
    if (parentSessionId === undefined) {
      setRootSessionId(pi, rootSessionId, session.sessionId);
      await forward(context, agent, "session", "sessionStart");
    }
    await forward(context, agent, "capture", "sessionStart", parentSessionId);
  });
  pi.on("before_agent_start", async (_event, context) => {
    await forward(context, agent, "capture", "beforeAgentStart", parentSessionId);
  });
  pi.on("agent_end", async (_event, context) => {
    await forward(context, agent, "capture", "agentEnd", parentSessionId);
  });
  pi.on("session_shutdown", async (_event, context) => {
    await forward(context, agent, "capture", "sessionShutdown", parentSessionId);
    const session = currentSession(context);
    if (session === null) return;
    if (parentSessionId === undefined) clearRootSessionId(pi, rootSessionId, session.sessionId);
  });
}

function currentSession(context: PiContext): { sessionId: string; transcriptPath: string } | null {
  const header = context.sessionManager.getHeader();
  const transcriptPath = context.sessionManager.getSessionFile();
  return header === null || transcriptPath === undefined
    ? null
    : { sessionId: header.id, transcriptPath };
}

function setRootSessionId(pi: PiExtension, key: string, sessionId: string): void {
  process.env[key] = sessionId;
  const shellEnv = pi.pi?.settings.getShellConfig().env;
  if (shellEnv !== undefined) shellEnv[key] = sessionId;
}

function clearRootSessionId(pi: PiExtension, key: string, sessionId: string): void {
  if (process.env[key] === sessionId) delete process.env[key];
  const shellEnv = pi.pi?.settings.getShellConfig().env;
  if (shellEnv?.[key] === sessionId) delete shellEnv[key];
}

async function forward(
  context: PiContext,
  agent: "omp" | "pi",
  kind: "session" | "capture",
  hookEventName: string,
  parentSessionId?: string,
): Promise<void> {
  try {
    const session = currentSession(context);
    if (session === null) return;
    const payload = JSON.stringify({
      hook_event_name: hookEventName,
      session_id: session.sessionId,
      transcript_path: session.transcriptPath,
      parent_session_id: parentSessionId,
    });
    await runWrapper(context.cwd, agent, kind, payload);
  } catch {
    // Trace transport must never alter Pi's event processing.
  }
}

function runWrapper(
  cwd: string,
  agent: "omp" | "pi",
  kind: "session" | "capture",
  payload: string,
): Promise<void> {
  return new Promise((resolve) => {
    const child = spawn("sh", [WRAPPER, agent, kind], {
      cwd,
      stdio: ["pipe", "ignore", "ignore"],
    });
    child.on("error", () => resolve());
    child.on("close", () => resolve());
    child.stdin.on("error", () => {});
    child.stdin.end(payload);
  });
}
