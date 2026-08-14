// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Abstract form of a model that processes language.
public protocol BaseLanguageModel: Module {
    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Optionally preprocess the weights with access to safetensor metadata.
    ///
    /// The default implementation forwards to ``sanitize(weights:)``.
    /// Models can override this to inspect metadata (e.g. check `metadata["format"] == "mlx"`)
    /// and skip or customize sanitization accordingly.
    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray]
}

extension BaseLanguageModel {
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights)
    }
}

/// Time/Height/Width struct to represent information about input images.
public struct THW: Sendable {

    public let t: Int
    public let h: Int
    public let w: Int

    public init(_ t: Int, _ h: Int, _ w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var values: (Int, Int, Int) {
        (t, h, w)
    }

    public var product: Int { t * h * w }
}

/// Representation of ``LanguageModel`` input.
///
/// This can contain text (tokens), prepared images (`MLXArray`), or other media as
/// needed. ``LMInput`` is produced by ``UserInputProcessor`` in response
/// to ``UserInput``.
///
/// The ``ModelContext`` holds the ``UserInputProcessor`` associated with a
/// ``LanguageModel``.
public struct LMInput {
    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?

    /// Representation of tokenized input text.
    public struct Text {

        /// input token array
        public let tokens: MLXArray

        /// optional mask array
        public let mask: MLXArray?

        public init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
        }

        public subscript(
            indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask?[indices, stream: stream])
        }

        public subscript(
            text indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask)
        }
    }

    /// Representation of prepared input image(s).
    public struct ProcessedImage {

        /// Concatenated pixels from one or more images
        public let pixels: MLXArray
        /// Time, height, and width of the images
        public let frames: [THW]?

        public init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    /// Representation of prepared input video(s).
    /// For now, this is virtually identical to ProcessedImage.
    public struct ProcessedVideo {

        public let pixels: MLXArray
        public let frames: [THW]?

        public init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    public init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.init(text: .init(tokens: tokens, mask: mask))
    }

    public init(
        text: LMInput.Text, image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
    }
}

/// ``LanguageModel`` step output. This is consumed internally
/// by the ``TokenIterator``.
public struct LMOutput {

    /// logits (one hot vector of probabilities for tokens)
    public let logits: MLXArray

    /// optional ``State`` to carry forward into the next step
    public let state: State?

    public struct State {
        public let crossAttentionStates: MLXArray?

        public init(crossAttentionStates: MLXArray? = nil) {
            self.crossAttentionStates = crossAttentionStates
        }
    }

    public init(logits: MLXArray, state: LMOutput.State? = nil) {
        self.logits = logits
        self.state = state
    }
}

/// The result of the call to ``LanguageModel/prepare(_:cache:windowSize:)``
public enum PrepareResult {
    /// tokens to process by the ``TokenIterator``
    case tokens(LMInput.Text)

    /// logits representing the next token
    case logits(LMOutput)
}

/// Interface for all Language Models (e.g. LLM, VLM).
///
/// The language model is typically called by the ``TokenIterator`` and it:
///
/// - consumes the ``LMInput``
/// - calls ``prepare(_:cache:windowSize:)`` to initialize the KVCache and consume the prompt
/// - calls ``callAsFunction(_:cache:state:)-9kuvf`` for each token, producing an ``LMOutput``
/// - the ``TokenIterator`` accumulates this information into a ``GenerateResult``
public protocol LanguageModel: BaseLanguageModel {

    /// Prepare the cache state and consume the ``LMInput``.
    ///
    /// This can return:
    /// - ``PrepareResult/tokens(_:)`` if the caller should evaluate the (remaining) tokens normally
    /// - ``PrepareResult/logits(_:)`` to produce the next token from the prompt
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult

    /// Primary entry point to produce a step (single token) from the model
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput

    /// Models may implement this simplified interface if they do not produce any ``LMOutput/State``
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    /// create a new array of ``KVCache``: automatic implementation if self
    /// implements ``KVCacheDimensionProvider``
    func newCache(parameters: GenerateParameters?) -> [KVCache]
}

extension LanguageModel {
    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }
}

/// Protocol for models that support Multi-Token Prediction (MTP) speculative decoding.
///
/// MTP embeds a lightweight draft head inside the model. After each backbone forward,
/// the head proposes token t+2 from the pre-norm hidden state at t+1. A subsequent
/// 2-token verify forward confirms or rejects the draft in one pass.
///
/// Eligibility: single-sequence batches only (`batchSize == 1`).
/// Throughput (greedy, accept rate p): ~1.74× at p≈1.0, ~1.30× at p≈0.5.
///
/// Port of omlx commit 696d90a:
///   patches/mlx_lm_mtp/qwen35_model.py (_patch_text_model / _patch_outer_model)
///   patches/mlx_lm_mtp/batch_generator.py (_is_mtp_eligible)
public protocol MTPCapable: LanguageModel {
    /// True when the MTP head is attached and operational.
    /// Returns false when the model was loaded without MTP (e.g. no --mtp flag),
    /// in which case `mtpForward` must not be called.
    var hasMTPHead: Bool { get }

    /// Run the MTP head. `hidden` is post-norm hidden from the backbone at position t
    /// (i.e., after `model.norm` — the "post_norm" variant). The MTP head's
    /// `pre_fc_norm_hidden` weights were trained on post-norm inputs.
    /// `nextTokenIds` shape: [1, S] (usually S=1). Returns logits [1, S, vocab].
    /// omlx: TextModel.mtp_forward
    /// MTPLX: hidden_variant="post_norm" (default for all generation commands)
    func mtpForward(hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]) -> MLXArray

    /// Allocate fresh KV caches for the MTP head layers.
    /// Called once per draft proposal — never reused across cycles.
    /// omlx: TextModel.make_mtp_cache
    func makeMTPCache() -> [any KVCache]

    /// Forward pass that also returns pre-norm hidden states.
    /// - Parameter nConfirmed: Confirmed prefix length for the 2-token verify input (0 = standard).
    /// - Returns: `(logits [B, S, vocab], preNormHidden [B, S, hiddenSize])`
    ///
    /// The returned hidden is the raw backbone output BEFORE `model.norm`. The MTP head applies
    /// `pre_fc_norm_hidden` itself, so passing post-norm would cause double-normalization.
    /// PR #990: `return out, hidden  # pre-norm hidden for MTP head`
    /// omlx: TextModel.__call__ with return_hidden=True + n_confirmed
    func callWithHidden(input: LMInput.Text, cache: [any KVCache], nConfirmed: Int) -> (MLXArray, MLXArray)
}

/// Optional protocol that can be implemented by ``LanguageModel`` and will
/// provide an automatic implementation of ``LanguageModel/newCache(parameters:)``
public protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Create one cache per layer (kvHeads.count = number of layers)
        // The number of heads per layer (kvHeads[i]) is not used for cache creation
        let numLayers = kvHeads.count

        // Follow Python logic: use RotatingKVCache if maxKVSize is provided
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< numLayers).map { _ in
                RotatingKVCache(maxSize: maxKVSize, keep: 4)
            }
        } else {
            return (0 ..< numLayers).map { _ in KVCacheSimple() }
        }
    }
}
