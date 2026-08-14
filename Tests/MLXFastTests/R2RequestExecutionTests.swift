import Foundation
import Testing

/// Executes `.github/scripts/{download,upload}-r2-object.sh` end to end.
///
/// Every other test of these scripts reads them as text or splices a few of
/// their lines into a fresh shell. That is how two signing bugs shipped: the
/// signed path is only ever *taken* when `aws` is missing from PATH, so before
/// M5-C (runner PATH `/usr/bin:/bin:/usr/sbin:/sbin`) it had literally never
/// run anywhere, and its first execution was a production hidden-golden fetch.
///
/// These tests run the real scripts, unmodified, with `R2_FORCE_SIGNED=1` and
/// with stub `curl`/`date`/`aws` binaries first on PATH. Nothing about the
/// script is faked: argument validation, the SigV4 signer, the retry loop, the
/// temp-file handling and the exit codes are the shipped ones. What is
/// observed is the argv and headers curl was actually handed, attempt by
/// attempt.
///
/// The stub PATH is deliberately `/usr/bin:/bin:/usr/sbin:/sbin` plus the stub
/// directory — the same minimal PATH M5-C's runner has, so `shasum`, `openssl`
/// (LibreSSL), `xxd` and friends resolve exactly as they do there.
@Suite("R2 request execution")
struct R2RequestExecutionTests {
    // MARK: - Harness

    /// One network attempt as the stub curl saw it.
    struct Attempt {
        var index: Int
        /// Bytes already in the `--output` file when the attempt began. The
        /// script is required to truncate between attempts, so this must be 0.
        var preexistingOutputBytes: Int
        var argv: [String]

        func headerValue(_ name: String) -> String? {
            let prefix = "\(name): "
            for (i, arg) in argv.enumerated() where arg == "-H" || arg == "--header" {
                guard i + 1 < argv.count else { continue }
                if argv[i + 1].hasPrefix(prefix) {
                    return String(argv[i + 1].dropFirst(prefix.count))
                }
            }
            return nil
        }

        var authorizationSignature: String? {
            guard let auth = headerValue("Authorization"),
                let range = auth.range(of: "Signature=")
            else { return nil }
            return String(auth[range.upperBound...])
        }

        var url: String? { argv.last }

        func flagValue(_ flag: String) -> String? {
            guard let i = argv.firstIndex(of: flag), i + 1 < argv.count else { return nil }
            return argv[i + 1]
        }
    }

    struct Run {
        var status: Int32
        var stdout: String
        var stderr: String
        var attempts: [Attempt]
        var awsInvocations: [String]
        var outputContents: String?
    }

    /// A stub `curl`.
    ///
    /// It models the two behaviours of real curl this concern turns on:
    ///
    ///   * `--retry N` is honoured *inside one process*, replaying the argv it
    ///     was given. That is precisely the defect — a curl-level retry cannot
    ///     re-sign, because the Authorization and x-amz-date headers are fixed
    ///     in argv before curl starts. So the pre-fix script logs N+1 attempts
    ///     with identical headers, exactly as measured against real R2.
    ///   * `--max-time` is what converts a stalled-but-open connection into a
    ///     failure. Without it, the `stall` script below returns HTTP 200 and
    ///     exit 0 having written nothing — the hang/empty-body outcome the fix
    ///     exists to prevent.
    ///
    /// It deliberately does NOT truncate the `--output` file (it appends), so
    /// that whether the file is empty at the start of an attempt is a property
    /// of the *script*, which is the thing under test. Real curl truncates only
    /// once it opens the file, which it never does on a connect-time failure.
    private static let curlStub = #"""
        #!/bin/bash
        set -u
        log="${R2_STUB_CURL_LOG}"
        counter="${R2_STUB_CURL_COUNTER}"

        out=""
        retries=0
        args=("$@")
        for ((i = 0; i < ${#args[@]}; i++)); do
          case "${args[i]}" in
            --output) out="${args[i+1]}" ;;
            --retry) retries="${args[i+1]}" ;;
          esac
        done

        have_max_time=0
        for a in "$@"; do
          if [[ "${a}" == "--max-time" ]]; then have_max_time=1; fi
        done

        IFS=' ' read -r -a plan <<< "${R2_STUB_PLAN:-200:0}"

        rc=0
        for ((k = 0; k <= retries; k++)); do
          n=$(( $(cat "${counter}" 2>/dev/null || echo 0) + 1 ))
          printf '%s' "${n}" > "${counter}"

          pre=0
          if [[ -n "${out}" && -e "${out}" ]]; then
            pre="$(wc -c < "${out}" | tr -d '[:space:]')"
          fi

          {
            printf 'ATTEMPT %s\n' "${n}"
            printf 'PREEXISTING %s\n' "${pre}"
            for a in "$@"; do printf 'ARG %s\n' "${a}"; done
            printf 'END\n'
          } >> "${log}"

          idx=$(( n - 1 ))
          if (( idx >= ${#plan[@]} )); then idx=$(( ${#plan[@]} - 1 )); fi
          entry="${plan[idx]}"

          if [[ "${entry}" == "stall" ]]; then
            if (( have_max_time )); then
              status="000"; rc=28
            else
              # Connection alive, no bytes, no timeout: curl reports success
              # and the caller is handed an EMPTY file.
              status="200"; rc=0
            fi
          else
            status="${entry%%:*}"
            rc="${entry##*:}"
            if [[ "${rc}" == "0" ]]; then
              [[ -n "${out}" ]] && printf '%s' "${R2_STUB_SUCCESS_BODY:-OK}" >> "${out}"
            elif [[ "${status}" != "000" ]]; then
              [[ -n "${out}" ]] && printf '<?xml version="1.0"?><Error><Code>SlowDown</Code><Message>stub error body for attempt %s, padded to be conspicuous if it survives into a later attempt</Message></Error>' "${n}" >> "${out}"
            fi
          fi

          if [[ "${rc}" == "0" ]]; then break; fi
        done

        printf '%s' "${status}"
        exit "${rc}"
        """#

    /// A stub `date` that advances 7 minutes per SigV4 clock read.
    ///
    /// Two attempts therefore straddle R2's ~15-minute clock-skew window, which
    /// is the failure a reused signature produces (403 RequestTimeTooSkewed).
    /// Anything that is not the SigV4 format string is delegated to /bin/date.
    private static let dateStub = #"""
        #!/bin/bash
        set -u
        if [[ "${1:-}" == "-u" && "${2:-}" == "+%Y%m%dT%H%M%SZ" ]]; then
          counter="${R2_STUB_DATE_COUNTER}"
          n=$(( $(cat "${counter}" 2>/dev/null || echo 0) + 1 ))
          printf '%s' "${n}" > "${counter}"
          printf '20260730T00%02d00Z' $(( (n - 1) * 7 ))
          exit 0
        fi
        exec /bin/date "$@"
        """#

    /// A stub `aws` that succeeds, so taking the fallback branch is observable.
    private static let awsStub = #"""
        #!/bin/bash
        set -u
        { printf 'aws'; for a in "$@"; do printf ' %s' "${a}"; done; printf '\n'; } \
          >> "${R2_STUB_AWS_LOG}"
        next_is_dest=0
        for a in "$@"; do
          if (( next_is_dest )); then
            next_is_dest=0
            case "${a}" in -*) ;; *) printf 'aws-stub-object' > "${a}" ;; esac
          fi
          case "${a}" in s3://*) next_is_dest=1 ;; esac
        done
        exit "${R2_STUB_AWS_EXIT:-0}"
        """#

    private static let repositoryRoot = FileManager.default.currentDirectoryPath

    private static func write(_ contents: String, to url: URL) throws {
        try contents.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: url.path)
    }

    /// Run one of the real scripts with the stubs in front of a minimal PATH.
    ///
    /// - Parameters:
    ///   - script: repo-relative script path, executed as itself.
    ///   - arguments: the script's two positional arguments.
    ///   - plan: per-attempt stub behaviour, e.g. `"503:22 200:0"` or `"stall"`.
    ///   - stubAWS: install a succeeding `aws` stub on PATH.
    ///   - forceSigned: set `R2_FORCE_SIGNED=1`.
    ///   - readOutputAt: repo/temp path whose contents to return afterwards.
    @discardableResult
    private static func runScript(
        _ script: String,
        arguments: [String],
        workingDirectory: URL,
        plan: String = "200:0",
        successBody: String = "the-object-bytes",
        stubAWS: Bool = false,
        forceSigned: Bool = true,
        extraEnvironment: [String: String] = [:],
        readOutputAt outputPath: String? = nil
    ) throws -> Run {
        let stubs = workingDirectory.appendingPathComponent("__stubs")
        try FileManager.default.createDirectory(at: stubs, withIntermediateDirectories: true)
        try write(curlStub, to: stubs.appendingPathComponent("curl"))
        try write(dateStub, to: stubs.appendingPathComponent("date"))
        if stubAWS {
            try write(awsStub, to: stubs.appendingPathComponent("aws"))
        }

        let curlLog = workingDirectory.appendingPathComponent("curl.log")
        let awsLog = workingDirectory.appendingPathComponent("aws.log")

        var environment: [String: String] = [
            "PATH": "\(stubs.path):/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": workingDirectory.path,
            "R2_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "R2_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "R2_BUCKET_ENDPOINT": "https://acct.r2.cloudflarestorage.com/mybucket",
            // Keep the loop's shape but not its wall clock.
            "R2_RETRY_DELAY_SECONDS": "0",
            "R2_STUB_PLAN": plan,
            "R2_STUB_SUCCESS_BODY": successBody,
            "R2_STUB_CURL_LOG": curlLog.path,
            "R2_STUB_CURL_COUNTER": workingDirectory.appendingPathComponent("curl.n").path,
            "R2_STUB_DATE_COUNTER": workingDirectory.appendingPathComponent("date.n").path,
            "R2_STUB_AWS_LOG": awsLog.path,
        ]
        if forceSigned { environment["R2_FORCE_SIGNED"] = "1" }
        for (key, value) in extraEnvironment { environment[key] = value }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["\(repositoryRoot)/\(script)"] + arguments
        process.currentDirectoryURL = workingDirectory
        process.environment = environment
        let out = Pipe()
        let err = Pipe()
        process.standardOutput = out
        process.standardError = err
        try process.run()
        let stdoutData = out.fileHandleForReading.readDataToEndOfFile()
        let stderrData = err.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        return Run(
            status: process.terminationStatus,
            stdout: String(decoding: stdoutData, as: UTF8.self),
            stderr: String(decoding: stderrData, as: UTF8.self),
            attempts: parseAttempts(at: curlLog),
            awsInvocations: (try? String(contentsOf: awsLog, encoding: .utf8))?
                .split(separator: "\n").map(String.init) ?? [],
            outputContents: outputPath.flatMap {
                let url =
                    $0.hasPrefix("/")
                    ? URL(fileURLWithPath: $0)
                    : workingDirectory.appendingPathComponent($0)
                return try? String(contentsOf: url, encoding: .utf8)
            }
        )
    }

    private static func parseAttempts(at url: URL) -> [Attempt] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        var attempts: [Attempt] = []
        var current: Attempt?
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("ATTEMPT ") {
                current = Attempt(
                    index: Int(line.dropFirst("ATTEMPT ".count)) ?? -1,
                    preexistingOutputBytes: 0, argv: [])
            } else if line.hasPrefix("PREEXISTING ") {
                current?.preexistingOutputBytes = Int(line.dropFirst("PREEXISTING ".count)) ?? -1
            } else if line.hasPrefix("ARG ") {
                current?.argv.append(String(line.dropFirst("ARG ".count)))
            } else if line == "END", let attempt = current {
                attempts.append(attempt)
                current = nil
            }
        }
        return attempts
    }

    private static func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("r2-exec-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static let downloadScript = ".github/scripts/download-r2-object.sh"
    private static let uploadScript = ".github/scripts/upload-r2-object.sh"

    // MARK: - F2: every attempt is signed afresh

    /// Six attempts, six DIFFERENT x-amz-date values and six DIFFERENT
    /// signatures.
    ///
    /// The retry used to be curl's own (`--retry 5 --retry-all-errors`), and
    /// curl replays the argv it was handed — measured against real R2, all six
    /// attempts carried the same x-amz-date, because `amz_date` was computed
    /// once before curl started. A signature is only valid inside R2's
    /// ~15-minute clock-skew window, so a run that spends long enough in
    /// retries turns a transient 503 into a permanent-looking 403
    /// RequestTimeTooSkewed. The stub curl honours `--retry` in-process for
    /// exactly this reason, so the pre-fix script still produces six logged
    /// attempts here — with identical headers.
    @Test
    func everyRetryIsSignedAfresh() throws {
        let workspace = try Self.temporaryDirectory()
        let run = try Self.runScript(
            Self.downloadScript,
            arguments: ["correctness_prompts/golden.json", "out/golden.json"],
            workingDirectory: workspace,
            plan: "503:22 503:22 503:22 503:22 503:22 200:0",
            readOutputAt: "out/golden.json"
        )

        #expect(
            run.status == 0,
            "download should have recovered on the sixth attempt: \(run.stderr)")
        #expect(
            run.attempts.count == 6,
            """
            expected 6 network attempts, saw \(run.attempts.count). Each must be a \
            SEPARATE curl invocation so the script can re-sign between them; curl's \
            own --retry cannot, because the headers are already fixed in its argv.
            """
        )

        let dates = run.attempts.compactMap { $0.headerValue("x-amz-date") }
        #expect(dates.count == run.attempts.count, "every attempt must send x-amz-date")
        #expect(
            Set(dates).count == run.attempts.count,
            """
            attempts reused an x-amz-date: \(dates). A retried request must carry a \
            FRESH date or it can fall outside R2's ~15 min skew window and be \
            rejected 403 RequestTimeTooSkewed no matter how healthy the bucket is.
            """
        )

        let signatures = run.attempts.compactMap { $0.authorizationSignature }
        #expect(signatures.count == run.attempts.count, "every attempt must be signed")
        #expect(
            Set(signatures).count == run.attempts.count,
            """
            attempts reused a SigV4 signature: \(signatures). The signature covers \
            x-amz-date, so a fresh date with a stale signature is just as dead as a \
            stale date.
            """
        )
        for signature in signatures {
            #expect(
                signature.count == 64 && signature.allSatisfy { $0.isHexDigit },
                "signature \(signature) is not a 64-hex SigV4 signature")
        }

        #expect(
            run.outputContents == "the-object-bytes",
            "final file should hold only the successful attempt's body, got \(run.outputContents ?? "<missing>")"
        )
    }

    /// The same property for the upload script.
    @Test
    func everyUploadRetryIsSignedAfresh() throws {
        let workspace = try Self.temporaryDirectory()
        let payload = workspace.appendingPathComponent("payload.json")
        try "{\"score\":1}".write(to: payload, atomically: true, encoding: .utf8)

        let run = try Self.runScript(
            Self.uploadScript,
            arguments: ["scores/run.json", payload.path],
            workingDirectory: workspace,
            plan: "500:22 500:22 200:0"
        )

        #expect(run.status == 0, "upload should have recovered: \(run.stderr)")
        #expect(run.attempts.count == 3, "expected 3 separate curl invocations")
        let dates = run.attempts.compactMap { $0.headerValue("x-amz-date") }
        let signatures = run.attempts.compactMap { $0.authorizationSignature }
        #expect(Set(dates).count == 3, "upload attempts reused an x-amz-date: \(dates)")
        #expect(
            Set(signatures).count == 3,
            "upload attempts reused a signature: \(signatures)")
    }

    // MARK: - F2: timeouts

    /// A stalled attempt must fail and be retried, not hang.
    ///
    /// Without `--max-time`, a connection that stays open and delivers nothing
    /// triggers neither curl's retry nor any timeout: the transfer "succeeds"
    /// with an empty body and hangs the job to its 30-minute limit in the worst
    /// case. The stub reproduces that: on a `stall` attempt it returns HTTP 200
    /// / exit 0 having written nothing *unless* `--max-time` is present, in
    /// which case it reports exit 28 like real curl.
    @Test
    func aStalledAttemptTimesOutAndIsRetriedRatherThanSucceedingEmpty() throws {
        let workspace = try Self.temporaryDirectory()
        let run = try Self.runScript(
            Self.downloadScript,
            arguments: ["correctness_prompts/golden.json", "out/golden.json"],
            workingDirectory: workspace,
            plan: "stall stall 200:0",
            readOutputAt: "out/golden.json"
        )

        #expect(run.status == 0, "download should have recovered: \(run.stderr)")
        #expect(
            run.outputContents == "the-object-bytes",
            """
            the script accepted a stalled transfer: wrote \
            \(run.outputContents.map { "\($0.count) byte(s)" } ?? "no file") instead of \
            the object. A stalled 200 delivers no bytes and, without --max-time, \
            reports success — the caller then fails a pinned-sha256 check for a \
            reason that has nothing to do with the object.
            """
        )
        #expect(
            run.attempts.count == 3,
            "expected the two stalled attempts to be abandoned and retried, saw \(run.attempts.count)"
        )
    }

    /// Both scripts must bound connect and total transfer time on every attempt.
    @Test
    func everyAttemptBoundsConnectAndTotalTime() throws {
        for (script, arguments, needsPayload) in [
            (Self.downloadScript, ["correctness_prompts/g.json", "out/g.json"], false),
            (Self.uploadScript, ["scores/run.json", "payload.json"], true),
        ] as [(String, [String], Bool)] {
            let workspace = try Self.temporaryDirectory()
            if needsPayload {
                try "x".write(
                    to: workspace.appendingPathComponent("payload.json"),
                    atomically: true, encoding: .utf8)
            }
            let run = try Self.runScript(
                script, arguments: arguments, workingDirectory: workspace,
                plan: "503:22 200:0")

            #expect(run.status == 0, "\(script) failed: \(run.stderr)")
            #expect(!run.attempts.isEmpty, "\(script) never invoked curl")
            for attempt in run.attempts {
                #expect(
                    attempt.flagValue("--connect-timeout") == "30",
                    """
                    \(script) attempt \(attempt.index) has no --connect-timeout 30. \
                    A dead peer then blocks on the OS default, and later retries can \
                    fall outside R2's clock-skew window.
                    """
                )
                #expect(
                    attempt.flagValue("--max-time") == "600",
                    """
                    \(script) attempt \(attempt.index) has no --max-time 600. A \
                    connection that stalls mid-body then never returns, and the CI \
                    job dies on its own limit with no diagnosis.
                    """
                )
            }
        }
    }

    /// The retry must not be delegated back to curl.
    @Test
    func noAttemptDelegatesRetryToCurl() throws {
        let workspace = try Self.temporaryDirectory()
        let run = try Self.runScript(
            Self.downloadScript,
            arguments: ["correctness_prompts/g.json", "out/g.json"],
            workingDirectory: workspace,
            plan: "503:22 200:0")

        #expect(!run.attempts.isEmpty)
        for attempt in run.attempts {
            #expect(
                !attempt.argv.contains("--retry") && !attempt.argv.contains("--retry-all-errors"),
                """
                attempt \(attempt.index) hands the retry to curl. curl replays its \
                argv, so the Authorization and x-amz-date it retries with are the \
                ones computed before it started: a retry that cannot re-sign.
                """
            )
        }
    }

    // MARK: - F2: the output file is truncated between attempts

    /// A failed attempt's error document must not survive into the next one.
    @Test
    func theOutputFileIsTruncatedBetweenAttempts() throws {
        let workspace = try Self.temporaryDirectory()
        let run = try Self.runScript(
            Self.downloadScript,
            arguments: ["correctness_prompts/golden.json", "out/golden.json"],
            workingDirectory: workspace,
            plan: "503:22 503:22 200:0",
            readOutputAt: "out/golden.json"
        )

        #expect(run.status == 0, "download should have recovered: \(run.stderr)")
        for attempt in run.attempts {
            #expect(
                attempt.preexistingOutputBytes == 0,
                """
                attempt \(attempt.index) began with \(attempt.preexistingOutputBytes) \
                byte(s) already in the --output file. curl truncates only once it \
                opens that file, which it never does when the attempt dies at connect \
                time, so a previous attempt's <Error> body can sit under — or ahead \
                of — the real object. The caller verifies a pinned sha256 and would \
                report a golden mismatch for a transport artefact.
                """
            )
        }
        #expect(
            run.outputContents == "the-object-bytes",
            """
            the downloaded file is not exactly the object: \
            \(run.outputContents ?? "<missing>")
            """
        )
    }

    // MARK: - F3: the object key charset is enforced

    /// Keys this script cannot sign correctly must be refused before any
    /// request is made, with a message that says why.
    @Test
    func unsignableObjectKeysAreRefusedBeforeAnyRequest() throws {
        let rejected = [
            "correctness_prompts/has space.json",
            "correctness_prompts/frag#ment.json",
            "correctness_prompts/query?x=1.json",
            "correctness_prompts/caf\u{00e9}.json",
            "correctness_prompts/star*.json",
            "correctness_prompts/semi;colon.json",
        ]

        for (script, secondArgument, needsPayload) in [
            (Self.downloadScript, "out/o.json", false),
            (Self.uploadScript, "payload.json", true),
        ] as [(String, String, Bool)] {
            for key in rejected {
                let workspace = try Self.temporaryDirectory()
                if needsPayload {
                    try "x".write(
                        to: workspace.appendingPathComponent("payload.json"),
                        atomically: true, encoding: .utf8)
                }
                let run = try Self.runScript(
                    script, arguments: [key, secondArgument],
                    workingDirectory: workspace)

                #expect(
                    run.status == 2,
                    """
                    \(script) accepted the unsignable key \(key) (exit \(run.status)). \
                    Nothing here percent-encodes, so the key bytes are signed verbatim \
                    and handed to curl verbatim: a space is curl exit 3, '#' truncates \
                    at the fragment, '?' starts a query string, and non-ASCII is \
                    percent-encoded on the wire after being signed raw — the last three \
                    arrive as 403 SignatureDoesNotMatch against a healthy bucket.
                    """
                )
                #expect(
                    run.stderr.contains("[A-Za-z0-9._/-]"),
                    """
                    \(script) rejected \(key) without naming the enforced charset. \
                    stderr was: \(run.stderr)
                    """
                )
                #expect(
                    run.attempts.isEmpty,
                    "\(script) issued \(run.attempts.count) request(s) for \(key) anyway")
            }
        }
    }


    // MARK: - F4: the payload hash survives exotic input paths

    /// `x-amz-content-sha256` must be the digest of the file's CONTENT, whatever
    /// the file is called.
    ///
    /// `shasum -a 256 "${path}"` prefixes the digest with a backslash and
    /// escapes the name when the path holds a backslash or a newline, so the
    /// signed hash becomes a 65-character non-hex string; a path starting with
    /// '-' is parsed as an option and yields nothing at all.
    @Test
    func thePayloadHashIsTheContentDigestForExoticInputPaths() throws {
        let body = "payload-content"
        // sha256("payload-content")
        let expected = "b7367c22dfc669fdf6f9fcdb91112e6aee109312a2fe68a1508b00dba48cc9cb"

        for name in ["back\\slash.json", "new\nline.json", "-dash.json", "plain.json"] {
            let workspace = try Self.temporaryDirectory()
            try body.write(
                to: workspace.appendingPathComponent(name), atomically: true, encoding: .utf8)

            // Relative, so a leading '-' really is exposed to argument parsing.
            let run = try Self.runScript(
                Self.uploadScript, arguments: ["scores/run.json", name],
                workingDirectory: workspace)

            #expect(
                run.status == 0,
                """
                upload of a file named \(name.debugDescription) failed (exit \
                \(run.status)): \(run.stderr)
                """
            )
            guard let attempt = run.attempts.first else {
                Issue.record("no request was made for \(name.debugDescription)")
                continue
            }
            #expect(
                attempt.headerValue("x-amz-content-sha256") == expected,
                """
                x-amz-content-sha256 for \(name.debugDescription) was \
                \(attempt.headerValue("x-amz-content-sha256") ?? "<absent>"), expected \
                \(expected) — the digest of the file's bytes. The same value is signed \
                into the canonical request, so a mangled one is a 403 \
                SignatureDoesNotMatch that reads as a credentials fault.
                """
            )
            #expect(
                attempt.headerValue("Authorization")?.contains("Signature=") == true,
                "the request for \(name.debugDescription) was not signed")
        }
    }

    // MARK: - F5: no redirect following

    /// `--location` must be gone from both scripts.
    ///
    /// Measured against real curl 8.7.1 (macOS system curl, the one the runners
    /// use) with two local servers and `-H Authorization: … -H x-amz-date: …
    /// -H x-amz-content-sha256: …`:
    ///
    ///     cross-host hop 1 (127.0.0.1)        authorization PRESENT
    ///     cross-host hop 2 (localhost, 302)   authorization ABSENT
    ///                                         x-amz-date    PRESENT
    ///                                         x-amz-content-sha256 PRESENT
    ///     same-host  hop 2 (/other-path)      authorization PRESENT (signature
    ///                                         still bound to /start)
    ///
    /// So a cross-host redirect delivers an UNSIGNED request that still carries
    /// the signing headers, and a same-host redirect delivers a signature for
    /// the wrong path. R2 path-style issues no legitimate redirects, so
    /// following one can only turn a loud failure into a confusing one — and,
    /// on a cross-host hop, ship the request somewhere it was never signed for.
    ///
    /// This asserts on the argv curl was actually handed by the real script,
    /// not on the script's text.
    @Test
    func noAttemptFollowsRedirects() throws {
        for (script, arguments, needsPayload) in [
            (Self.downloadScript, ["correctness_prompts/g.json", "out/g.json"], false),
            (Self.uploadScript, ["scores/run.json", "payload.json"], true),
        ] as [(String, [String], Bool)] {
            let workspace = try Self.temporaryDirectory()
            if needsPayload {
                try "x".write(
                    to: workspace.appendingPathComponent("payload.json"),
                    atomically: true, encoding: .utf8)
            }
            let run = try Self.runScript(
                script, arguments: arguments, workingDirectory: workspace)

            #expect(!run.attempts.isEmpty, "\(script) never invoked curl")
            for attempt in run.attempts {
                #expect(
                    !attempt.argv.contains("--location") && !attempt.argv.contains("-L"),
                    """
                    \(script) attempt \(attempt.index) follows redirects. curl drops \
                    the Authorization header across hosts but keeps x-amz-date and \
                    x-amz-content-sha256, so the follow-up request arrives unsigned \
                    with signing headers attached — a 400/403 pointing at the wrong \
                    cause.
                    """
                )
            }
        }
    }

    // MARK: - F6: the signed path has an execution environment

    /// `R2_FORCE_SIGNED=1` must bypass the aws CLI even when `aws` works.
    ///
    /// Without this the signer is reachable only by *not* having `aws`
    /// installed, which is why it went unexecuted until a production run on
    /// M5-C. Everything above depends on this knob.
    @Test
    func forceSignedBypassesAWorkingAWSCLI() throws {
        for (script, arguments, needsPayload, banner) in [
            (
                Self.downloadScript, ["correctness_prompts/g.json", "out/g.json"], false,
                "download-r2-object: using signed HTTPS download"
            ),
            (
                Self.uploadScript, ["scores/run.json", "payload.json"], true,
                "upload-r2-object: using signed HTTPS upload"
            ),
        ] as [(String, [String], Bool, String)] {
            // Baseline: `aws` present and working -> the fallback is taken and
            // the signer never runs. This is the production reality on M5-A.
            let fallbackWorkspace = try Self.temporaryDirectory()
            if needsPayload {
                try "x".write(
                    to: fallbackWorkspace.appendingPathComponent("payload.json"),
                    atomically: true, encoding: .utf8)
            }
            let fallback = try Self.runScript(
                script, arguments: arguments, workingDirectory: fallbackWorkspace,
                stubAWS: true, forceSigned: false)
            #expect(fallback.status == 0, "\(script) aws fallback failed: \(fallback.stderr)")
            #expect(
                !fallback.awsInvocations.isEmpty,
                "\(script) did not use the aws CLI even though it was on PATH")
            #expect(
                fallback.attempts.isEmpty,
                """
                \(script) reached the signer with a working aws CLI on PATH. If that \
                is now the default, the premise of this test has changed — but note \
                the whole point: the signer historically never ran, so it was never \
                tested.
                """
            )

            // With the knob: same PATH, same working `aws`, signer runs anyway.
            let signedWorkspace = try Self.temporaryDirectory()
            if needsPayload {
                try "x".write(
                    to: signedWorkspace.appendingPathComponent("payload.json"),
                    atomically: true, encoding: .utf8)
            }
            let signed = try Self.runScript(
                script, arguments: arguments, workingDirectory: signedWorkspace,
                stubAWS: true, forceSigned: true)
            #expect(signed.status == 0, "\(script) signed path failed: \(signed.stderr)")
            #expect(
                signed.awsInvocations.isEmpty,
                """
                R2_FORCE_SIGNED=1 did not skip the aws CLI for \(script); it ran \
                \(signed.awsInvocations.count) time(s). Without a way to force the \
                signed path, the SigV4 signer can only be exercised by uninstalling \
                aws — i.e. never, until a production run does it by accident.
                """
            )
            #expect(
                !signed.attempts.isEmpty,
                "R2_FORCE_SIGNED=1 did not reach the signed HTTPS path for \(script)")
            #expect(
                signed.stdout.contains(banner),
                "\(script) did not announce the signed path; stdout: \(signed.stdout)")
        }
    }

    /// The request the signer actually emits must be the one it signed.
    ///
    /// A whole-script check that the URL, the three signed headers and the
    /// SignedHeaders list agree — the pieces the extraction-based tests in
    /// R2SignatureTests verify in isolation, here observed on the wire.
    @Test
    func theEmittedRequestMatchesWhatWasSigned() throws {
        let workspace = try Self.temporaryDirectory()
        let run = try Self.runScript(
            Self.downloadScript,
            arguments: ["correctness_prompts/laguna/golden.json", "out/g.json"],
            workingDirectory: workspace, readOutputAt: "out/g.json")

        #expect(run.status == 0, "download failed: \(run.stderr)")
        let attempt = try #require(run.attempts.first)
        #expect(
            attempt.url
                == "https://acct.r2.cloudflarestorage.com/mybucket/correctness_prompts/laguna/golden.json"
        )
        // Empty-payload digest: the GET body is empty and must be signed as such.
        #expect(
            attempt.headerValue("x-amz-content-sha256")
                == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        let authorization = try #require(attempt.headerValue("Authorization"))
        #expect(authorization.hasPrefix("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/"))
        #expect(
            authorization.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"),
            """
            SignedHeaders must list exactly the headers sent, sorted. Authorization \
            was: \(authorization)
            """
        )
        let amzDate = try #require(attempt.headerValue("x-amz-date"))
        #expect(
            authorization.contains("/\(amzDate.prefix(8))/auto/s3/aws4_request"),
            """
            the credential scope's date must be the x-amz-date's day. A mismatch is \
            403 SignatureDoesNotMatch. Authorization was: \(authorization)
            """
        )
        #expect(run.outputContents == "the-object-bytes")
    }
}
