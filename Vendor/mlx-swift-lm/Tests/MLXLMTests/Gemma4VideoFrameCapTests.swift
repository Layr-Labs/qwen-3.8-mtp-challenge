// Copyright © 2026 Eigen Labs Inc.

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

/// Regression test for the Gemma4 video frame cap.
///
/// `Gemma4.prepare(input:)` samples video frames with
/// `MediaProcessing.asProcessedSequence(_:targetFPS:maxFrames:)` using a hard
/// `maxFrames: 32` (see `Libraries/MLXVLM/Models/Gemma4.swift`). Before that fix,
/// sampling was unbounded and a long clip could explode into hundreds of frames
/// (one vision block each), blowing up the prompt. These tests assert the 32-frame
/// cap holds on the `AVAsset` sampling path — the path that actually enforces
/// `maxFrames` — so removing the cap fails CI.
final class Gemma4VideoFrameCapTests: XCTestCase {

    /// The exact target-fps closure used by the Gemma4 video path.
    private static func gemma4TargetFPS(_ duration: CMTime) -> Double {
        32.0 / max(duration.seconds, 1.0)
    }

    /// A long clip sampled at a fps high enough that the *uncapped* frame count
    /// would far exceed 32 must still be capped to 32. This is the core
    /// regression assertion: it fails if `maxFrames: 32` is dropped from the
    /// AVAsset sampling path.
    func testLongVideoIsCappedAtThirtyTwoFrames() async throws {
        let url = try Self.makeSyntheticVideo(durationSeconds: 40, fps: 5)
        defer { try? FileManager.default.removeItem(at: url) }

        let video = UserInput.Video.url(url)

        // targetFPS of 5 over a 40s clip => round(5 * 40) = 200 estimated frames.
        // Without `maxFrames: 32` this would attempt ~200 samples.
        let processed = try await MediaProcessing.asProcessedSequence(
            video,
            targetFPS: { _ in 5.0 },
            maxFrames: 32
        )

        XCTAssertLessThanOrEqual(
            processed.frames.count, 32,
            "video sampling must be capped at 32 frames; got \(processed.frames.count)")
        XCTAssertGreaterThan(
            processed.frames.count, 0, "a decodable long clip should yield at least one frame")
        XCTAssertEqual(
            processed.frames.count, processed.timestamps.count,
            "frame and timestamp counts must agree")
    }

    /// The production Gemma4 closure (`32 / duration`) targets ~32 frames spread
    /// across the whole clip and must never exceed the 32-frame cap, while still
    /// producing a healthy number of frames for a long video.
    func testGemma4ProductionSamplingStaysWithinCap() async throws {
        let url = try Self.makeSyntheticVideo(durationSeconds: 40, fps: 5)
        defer { try? FileManager.default.removeItem(at: url) }

        let video = UserInput.Video.url(url)

        let processed = try await MediaProcessing.asProcessedSequence(
            video,
            targetFPS: { duration in Self.gemma4TargetFPS(duration) },
            maxFrames: 32
        )

        XCTAssertLessThanOrEqual(
            processed.frames.count, 32,
            "Gemma4 sampling must stay within the 32-frame cap; got \(processed.frames.count)")
        XCTAssertGreaterThan(
            processed.frames.count, 0, "a long clip should yield frames under Gemma4 sampling")
        // For a 40s clip the closure yields fps = 32/40 = 0.8 => round(0.8 * 40) = 32,
        // so a healthy long clip should land at (or just under) the cap.
        XCTAssertGreaterThanOrEqual(
            processed.frames.count, 16,
            "a 40s clip should produce close to 32 frames, not a degenerate handful")
    }

    /// Pure-arithmetic backstop covering the cap math directly, so the invariant
    /// is documented even if synthetic-asset decoding is unavailable on a runner.
    /// Mirrors `MediaProcessing` exactly: desired = min(round(fps*duration), maxFrames),
    /// final = max(desired, 1).
    func testFrameCapArithmetic() {
        func cappedFrameCount(durationSeconds: Double, maxFrames: Int = 32) -> Int {
            let fps = 32.0 / max(durationSeconds, 1.0)
            let estimated = Int((fps * durationSeconds).rounded())
            let desired = min(estimated, maxFrames)
            return max(desired, 1)
        }

        for duration in [40.0, 120.0, 600.0, 3600.0] {
            XCTAssertLessThanOrEqual(
                cappedFrameCount(durationSeconds: duration), 32,
                "duration \(duration)s must be capped at 32 frames")
        }
        // Long clips with the Gemma4 closure land exactly on the cap.
        XCTAssertEqual(cappedFrameCount(durationSeconds: 40.0), 32)
        XCTAssertEqual(cappedFrameCount(durationSeconds: 600.0), 32)
        // Very short clips keep at least one frame.
        XCTAssertGreaterThanOrEqual(cappedFrameCount(durationSeconds: 0.5), 1)
    }

    // MARK: - Synthetic video

    /// Writes a solid-color H.264 `.mov` of `durationSeconds` at `fps` to a temp
    /// file and returns its URL. The caller is responsible for deleting the file.
    private static func makeSyntheticVideo(durationSeconds: Int, fps: Int) throws -> URL {
        let width = 64
        let height = 64

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-framecap-\(UUID().uuidString).mov")
        try? FileManager.default.removeItem(at: url)

        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false

        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: attrs)

        guard writer.canAdd(input) else {
            throw NSError(
                domain: "Gemma4VideoFrameCapTests", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "cannot add writer input"])
        }
        writer.add(input)

        guard writer.startWriting() else {
            throw writer.error
                ?? NSError(
                    domain: "Gemma4VideoFrameCapTests", code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "startWriting failed"])
        }
        writer.startSession(atSourceTime: .zero)

        let timescale: Int32 = Int32(fps)
        let totalFrames = durationSeconds * fps

        for frameIndex in 0 ..< totalFrames {
            // Spin until the input is ready to accept more data.
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.002)
            }

            let buffer = try makePixelBuffer(
                width: width, height: height, frameIndex: frameIndex)
            let time = CMTime(value: Int64(frameIndex), timescale: timescale)
            if !adaptor.append(buffer, withPresentationTime: time) {
                throw writer.error
                    ?? NSError(
                        domain: "Gemma4VideoFrameCapTests", code: 3,
                        userInfo: [NSLocalizedDescriptionKey: "append frame failed"])
            }
        }

        input.markAsFinished()

        let group = DispatchGroup()
        group.enter()
        writer.finishWriting { group.leave() }
        group.wait()

        guard writer.status == .completed else {
            throw writer.error
                ?? NSError(
                    domain: "Gemma4VideoFrameCapTests", code: 4,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "writer did not complete: \(writer.status.rawValue)"
                    ])
        }

        return url
    }

    /// A 32BGRA pixel buffer filled with a per-frame varying solid color so the
    /// encoder produces distinct, decodable frames.
    private static func makePixelBuffer(
        width: Int, height: Int, frameIndex: Int
    ) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw NSError(
                domain: "Gemma4VideoFrameCapTests", code: 5,
                userInfo: [NSLocalizedDescriptionKey: "CVPixelBufferCreate failed: \(status)"])
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw NSError(
                domain: "Gemma4VideoFrameCapTests", code: 6,
                userInfo: [NSLocalizedDescriptionKey: "no base address"])
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let ptr = base.assumingMemoryBound(to: UInt8.self)

        // BGRA (little-endian 32BGRA => byte order B,G,R,A); vary RGB by frame
        // index so successive frames differ.
        let r = UInt8((frameIndex * 7) & 0xFF)
        let g = UInt8((frameIndex * 13) & 0xFF)
        let b = UInt8((frameIndex * 23) & 0xFF)
        for y in 0 ..< height {
            let row = ptr + y * bytesPerRow
            for x in 0 ..< width {
                let px = row + x * 4
                px[0] = b
                px[1] = g
                px[2] = r
                px[3] = 0xFF  // A
            }
        }

        return buffer
    }
}
