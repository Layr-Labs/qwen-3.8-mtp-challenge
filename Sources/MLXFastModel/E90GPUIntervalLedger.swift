import Darwin
import Foundation
import Metal
import ObjectiveC

// E90 RESEARCH INSTRUMENT -- record the GPU execution interval of every command
// buffer this process submits, so an offline reader can place GPU busy and GPU
// idle time inside the host anchors of one drafting round. Off unless
// `MLX_E90_GPU_INTERVALS=1` is set, and the shipped default is a no-op.
//
// WHY NOT THE E58/E80 CENSUS. That instrument answers "which kernel spent the
// time" and pays for it with a lock on every dispatch, every pipeline bind and
// every barrier, which its own header calls unfit for timing. E90 asks only
// "was the device running at host time T", so it hooks ONE selector -- command
// buffer commit -- and does two clock reads plus one array append per buffer.
// A drafting round submits tens of buffers, not thousands of dispatches, so the
// added host cost stays about three orders of magnitude below the 4.7 ms window
// under investigation.
//
// The `MLX_` prefix is load-bearing: `sanitizedRuntimeWorkerEnvironment` admits
// `MLX_` and drops `MLXFAST_`, so an `MLXFAST_`-spelled gate would never reach
// the worker process that owns the decode path.
public enum E90GPUIntervals {
    private static let environment = ProcessInfo.processInfo.environment

    public static let enabled = environment["MLX_E90_GPU_INTERVALS"] == "1"

    /// Installs the commit hook. Safe to call more than once.
    public static func installIfRequested() {
        guard enabled else { return }
        E90IntervalLedger.shared.install()
    }
}

/// `@unchecked Sendable` because every mutable field is reached only under
/// `lock`. Completion handlers run on whichever thread the Metal driver picks,
/// while `install` runs on the startup thread.
private final class E90IntervalLedger: @unchecked Sendable {
    static let shared = E90IntervalLedger()

    private let lock = NSLock()
    private var records: [(commit: UInt64, start: UInt64, end: UInt64, done: UInt64)] = []
    private var installed = false
    private var flushIndex = 0
    private var committedBuffers = 0
    private var completedBuffers = 0
    /// Buffers whose driver timestamps are zero or inverted. They are counted
    /// and dropped rather than folded into the union, because a zero start
    /// would swallow the whole timeline.
    private var invalidBuffers = 0

    private static let flushEvery = 2048

    /// Held for the lifetime of the process: a cancelled source stops firing.
    private var terminationSource: DispatchSourceSignal?

    private let sink: FileHandle = {
        guard let path = ProcessInfo.processInfo
            .environment["MLX_E90_GPU_INTERVALS_PATH"], !path.isEmpty
        else { return FileHandle.standardError }
        let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { return FileHandle.standardError }
        return FileHandle(fileDescriptor: fd, closeOnDealloc: false)
    }()

    func install() {
        lock.lock()
        let already = installed
        installed = true
        lock.unlock()
        guard !already else { return }
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let probe = queue.makeCommandBuffer()
        else {
            write(["event": "e90_install_failed"])
            return
        }
        let bufferClass: AnyClass = type(of: probe as AnyObject)
        let ok = swizzleCommit(bufferClass)
        atexit_b { E90IntervalLedger.shared.flush(reason: "exit") }
        installTerminationFlush()
        write([
            "event": "e90_installed",
            "buffer_class": String(describing: bufferClass),
            "commit_hook": ok ? 1 : 0,
        ])
    }

    /// The harness stops the worker with SIGTERM, whose default action skips
    /// `atexit`, so without this the last partial batch never reaches the sink.
    /// A dispatch source runs the flush on an ordinary queue thread instead of
    /// in signal context, then re-raises SIGTERM so the parent still observes
    /// the same termination reason.
    private func installTerminationFlush() {
        signal(SIGTERM, SIG_IGN)
        let source = DispatchSource.makeSignalSource(
            signal: SIGTERM, queue: .global(qos: .userInitiated))
        source.setEventHandler {
            E90IntervalLedger.shared.flush(reason: "sigterm")
            signal(SIGTERM, SIG_DFL)
            raise(SIGTERM)
        }
        source.resume()
        terminationSource = source
    }

    func record(commit: UInt64, start: UInt64, end: UInt64, done: UInt64) {
        lock.lock()
        completedBuffers += 1
        if start == 0 || end == 0 || end < start {
            invalidBuffers += 1
        } else {
            records.append((commit: commit, start: start, end: end, done: done))
        }
        let full = records.count >= Self.flushEvery
        lock.unlock()
        if full { flush(reason: "rolling") }
    }

    func noteCommit() {
        lock.lock()
        committedBuffers += 1
        lock.unlock()
    }

    func flush(reason: String) {
        lock.lock()
        let batch = records
        records.removeAll(keepingCapacity: true)
        let index = flushIndex
        flushIndex += 1
        let committed = committedBuffers
        let completed = completedBuffers
        let invalid = invalidBuffers
        lock.unlock()
        guard !batch.isEmpty || reason == "exit" else { return }
        // One flat array per field keeps the payload about a third the size of
        // an array of objects, and the reader zips them by index.
        write([
            "event": "e90_intervals",
            "reason": reason,
            "flush": index,
            "buffers": batch.count,
            "committed_total": committed,
            "completed_total": completed,
            "invalid_total": invalid,
            "commit_ns": batch.map { $0.commit },
            "gpu_start_ns": batch.map { $0.start },
            "gpu_end_ns": batch.map { $0.end },
            "completed_ns": batch.map { $0.done },
        ])
    }

    private func write(_ payload: [String: Any]) {
        var line = payload
        line["pid"] = ProcessInfo.processInfo.processIdentifier
        line["t_ns"] = DispatchTime.now().uptimeNanoseconds
        guard let data = try? JSONSerialization.data(withJSONObject: line) else { return }
        lock.lock()
        sink.write(data)
        sink.write(Data("\n".utf8))
        lock.unlock()
    }
}

private typealias E90CommitIMP = @convention(c) (AnyObject, Selector) -> Void

private func swizzleCommit(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("commit")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(
        method_getImplementation(method), to: E90CommitIMP.self)
    let replacement: @convention(block) (AnyObject) -> Void = { buffer in
        let commitNs = DispatchTime.now().uptimeNanoseconds
        if let commandBuffer = buffer as? MTLCommandBuffer {
            E90IntervalLedger.shared.noteCommit()
            commandBuffer.addCompletedHandler { done in
                // `gpuStartTime` and `gpuEndTime` are CFTimeInterval seconds on
                // the same mach uptime clock as `DispatchTime`, so the host
                // anchors of a round and the device intervals share one axis.
                let start = done.gpuStartTime
                let end = done.gpuEndTime
                E90IntervalLedger.shared.record(
                    commit: commitNs,
                    start: start > 0 ? UInt64((start * 1e9).rounded()) : 0,
                    end: end > 0 ? UInt64((end * 1e9).rounded()) : 0,
                    done: DispatchTime.now().uptimeNanoseconds)
            }
        }
        original(buffer, selector)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}
