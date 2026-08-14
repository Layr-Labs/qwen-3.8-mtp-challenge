import Foundation
import MLX
import Testing

private let naxSplitKFactoryPath =
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/jit_kernels.cpp"
private let naxSplitKDeclarationPath =
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels.h"
private let naxSplitKDispatchPath =
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/matmul.cpp"

@Test
func naxSplitKJITFactoryUsesInputDType() throws {
    let factory = try String(contentsOfFile: naxSplitKFactoryPath, encoding: .utf8)
    let declaration = try String(
        contentsOfFile: naxSplitKDeclarationPath, encoding: .utf8)
    let dispatch = try String(contentsOfFile: naxSplitKDispatchPath, encoding: .utf8)

    let functionName = "MTL::ComputePipelineState* get_steel_gemm_splitk_nax_kernel("
    let nextFunctionName = "MTL::ComputePipelineState* get_steel_gemm_segmented_nax_kernel("
    let factoryStart = try #require(factory.range(of: functionName))
    let factoryEnd = try #require(
        factory.range(of: nextFunctionName, range: factoryStart.upperBound..<factory.endIndex)
    )
    let splitKFactory = factory[factoryStart.lowerBound..<factoryEnd.lowerBound]
    #expect(splitKFactory.contains("const array& in,"))
    #expect(splitKFactory.contains("get_type_string(in.dtype()),"))
    #expect(!splitKFactory.contains("get_type_string(out.dtype()),"))

    let declarationStart = try #require(declaration.range(of: functionName))
    let declarationEnd = try #require(
        declaration.range(
            of: nextFunctionName,
            range: declarationStart.upperBound..<declaration.endIndex
        )
    )
    let splitKDeclaration =
        declaration[declarationStart.lowerBound..<declarationEnd.lowerBound]
    #expect(splitKDeclaration.contains("const array& in,"))
    #expect(!splitKDeclaration.contains("const array& out,"))

    let callStart = try #require(
        dispatch.range(of: "auto kernel = get_steel_gemm_splitk_nax_kernel(")
    )
    let callEnd = try #require(
        dispatch.range(of: "compute_encoder.set_compute_pipeline_state(kernel);",
                       range: callStart.upperBound..<dispatch.endIndex)
    )
    let splitKCall = dispatch[callStart.lowerBound..<callEnd.lowerBound]
    #expect(splitKCall.contains(
        """
              /* const array& in = */ a,
        """
    ))
    #expect(!splitKCall.contains("/* const array& out = */ C_split,"))
}

@Test
func naxSplitKBF16MatchesRegularNAXWhenRuntimeTestsAreEnabled() {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    else {
        return
    }

    // Laguna layer-0 o_proj is [1, sequence, 6144] @ [6144, 2048].
    // The neighboring cases cover another sequence length, another output
    // width, and an unaligned K tail while retaining the same BF16 path.
    let shapes = [
        (m: 2, n: 2048, k: 6144),
        (m: 3, n: 2048, k: 6144),
        (m: 2, n: 1984, k: 6144),
        (m: 2, n: 2048, k: 6145),
    ]

    for shape in shapes {
        let kAxis = MLXArray(0..<shape.k)
            .asType(.float32)
            .reshaped([1, shape.k])
        let rowAxis = MLXArray(0..<shape.m)
            .asType(.float32)
            .reshaped([shape.m, 1])
        let columnAxis = MLXArray(0..<shape.n)
            .asType(.float32)
            .reshaped([shape.n, 1])

        let input = (sin(kAxis * 0.013 + rowAxis * 0.17) * 0.125)
            .asType(.bfloat16)
            .reshaped([1, shape.m, shape.k])
        let weight = (cos(kAxis * 0.007 + columnAxis * 0.011) * 0.0625)
            .asType(.bfloat16)
        let inputStrides = input.asData(access: .noCopy).strides
        let weightStrides = weight.asData(access: .noCopy).strides

        #expect(input.shape == [1, shape.m, shape.k])
        #expect(inputStrides == [shape.m * shape.k, shape.k, 1])
        #expect(weight.shape == [shape.n, shape.k])
        #expect(weightStrides == [shape.k, 1])

        // batch_size_out == 1 selects NAX split-K for these large-K shapes.
        let splitK = matmul(input, weight.T)[0]

        // A broadcast A batch plus materialized B batches keep batch_size_out
        // at 2, routing the identical operation to regular NAX. In particular,
        // the nonzero B batch stride prevents folding the batch dimension into M.
        let regularInput = broadcast(input, to: [2, shape.m, shape.k])
        let regularWeight = stacked([weight, weight], axis: 0)
        #expect(regularInput.shape == [2, shape.m, shape.k])
        #expect(regularWeight.shape == [2, shape.n, shape.k])
        let regularNAX = matmul(
            regularInput, regularWeight.transposed(0, 2, 1)
        )[0]
        eval(splitK, regularNAX)
        let maxAbsoluteDifference = abs(
            splitK.asType(.float32) - regularNAX.asType(.float32)
        ).max().item(Float.self)
        print(
            "nax_splitk_case M=\(shape.m) N=\(shape.n) K=\(shape.k) "
                + "a_strides=\(inputStrides) weight_strides=\(weightStrides) "
                + "max_abs_diff=\(maxAbsoluteDifference)"
        )

        #expect(
            isFinite(splitK).all().item(Bool.self),
            "NAX split-K produced a non-finite result for \(shape)"
        )
        #expect(
            isFinite(regularNAX).all().item(Bool.self),
            "regular NAX produced a non-finite reference for \(shape)"
        )
        #expect(
            maxAbsoluteDifference <= 0.005,
            "NAX split-K absolute error exceeded BF16 tolerance for \(shape)"
        )
        #expect(
            allClose(splitK, regularNAX, rtol: 0.01, atol: 0.005).item(Bool.self),
            "NAX split-K diverged from regular NAX for \(shape)"
        )
    }
}
