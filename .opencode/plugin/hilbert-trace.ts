// Hilbert OpenCode trace transport — hilbert-opencode-trace-plugin-v6.
//
// Keep this repository-owned boundary stable across CLI upgrades. Every native event except noisy
// token deltas is forwarded; old and new CLIs independently choose the events they understand.
import { spawn } from "node:child_process";

const WRAPPER = ".hilbert/hooks/hilbert-trace.sh";

export const HilbertTrace = async ({ worktree }: { worktree?: string }) => ({
  event: ({ event }: { event: unknown }) => {
    try {
      if ((event as { type?: unknown } | null)?.type === "message.part.delta") return;
      const payload = JSON.stringify(event);
      if (payload === undefined) return;

      const child = spawn("sh", [WRAPPER, "opencode", "capture"], {
        cwd: typeof worktree === "string" && worktree.length > 0 ? worktree : process.cwd(),
        detached: true,
        stdio: ["pipe", "ignore", "ignore"],
      });
      child.on("error", () => {});
      child.stdin.on("error", () => {});
      child.stdin.end(payload);
      child.unref();
    } catch {
      // Trace transport must never alter OpenCode's event processing.
    }
  },
});
