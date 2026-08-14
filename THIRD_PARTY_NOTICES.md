# Third-Party Notices

This repository's own code is licensed per the top-level `LICENSE` file. The
challenge additionally depends on third-party model artifacts and Swift
packages credited below. Nothing in this file grants rights to the mlxfast
harness itself, and no model weights are distributed inside this repository —
the setup script (`setup.sh`) downloads the pinned artifacts
from their upstream repositories and verifies them against checked-in SHA256
manifests.

## Poolside Laguna XS 2.1 models

Laguna XS 2.1 and the Laguna XS 2.1 DFlash speculator are © Poolside,
published under the OpenMDW-1.1 license with terms on the model cards at
<https://huggingface.co/poolside/Laguna-XS-2.1>. The current NVFP4 MLX
checkpoint is published directly by Poolside under the same license tag.

Artifacts referenced (and hash-pinned) by this repository:

| Artifact | Upstream | License |
|---|---|---|
| `poolside/Laguna-XS-2.1` @ `c405648833500615a2efde76886b8aed4fb9324e` (upstream model, BF16) | Poolside | OpenMDW-1.1 |
| `poolside/Laguna-XS-2.1-NVFP4-mlx` @ `841778bda563a36104dd521e37d99218e46f4f25` (current v2 serial-track reference checkpoint; Poolside NVFP4, group size 16) | Poolside | OpenMDW-1.1 |
| `mlx-community/Laguna-XS-2.1-4bit` @ `c42e0a8f8d504ceacde015a535dcb286d65c8799` (historical `laguna-xs-2.1-serial-v1` reference; MLX affine 4-bit, group size 64) | mlx-community conversion of the above | OpenMDW-1.1 |

Attribution statement: "Laguna XS 2.1 and Laguna XS 2.1 NVFP4 MLX
© Poolside, licensed OpenMDW-1.1
(<https://huggingface.co/poolside/Laguna-XS-2.1>). Historical affine MLX
target conversion by mlx-community."

Compliance notes for anyone redistributing these models or derivative
weights obtained through this harness (e.g. a transformed `weights/`
tree):

- Review and satisfy the OpenMDW-1.1 terms published on the model cards
  before redistribution. TODO(operator): confirm the SPDX identifier and a
  canonical license-text URL before public go-live.
- Mark modified files as changed (the harness's `transform` output is a
  repacked/derivative artifact of the pinned checkpoint; its provenance is
  recorded in the emitted `config.json` and hash manifests).
- Retain the attribution statement above.
- The Apache License 2.0 appendix below is retained for the Swift package
  dependencies listed in the next section.

## Swift package dependencies

The harness links (does not vendor) the following packages via SwiftPM; each
license text is available in the corresponding package repository and, after
a build, under `.build/checkouts/<package>/LICENSE*`:

| Package | License |
|---|---|
| `Layr-Labs/mlx-swift` (fork of `ml-explore/mlx-swift`, incl. the MLX C++ core and Metal kernels) | MIT (© 2023 ml-explore / Apple) |
| `Layr-Labs/mlx-swift-lm` (fork of `ml-explore/mlx-swift-examples` LM libraries) | MIT (© 2024 ml-explore / Apple) |
| `huggingface/swift-transformers`, `huggingface/swift-huggingface`, `huggingface/swift-jinja` | Apache-2.0 |
| Apple `swift-*` packages (`argument-parser`, `algorithms`, `asn1`, `async-algorithms`, `atomics`, `certificates`, `collections`, `configuration`, `crypto`, `distributed-tracing`, `http-structured-headers`, `http-types`, `log`, `metrics`, `nio` family, `numerics`, `service-context`, `syntax`, `system`) | Apache-2.0 |
| `swift-server/async-http-client`, `swift-server/swift-service-lifecycle`, `hummingbird-project/hummingbird` | Apache-2.0 |
| `ibireme/yyjson` | MIT |
| `mattt/EventSource` | MIT |

## Tooling

- `macmon` (GPU telemetry used by the local/ranked thermal gate) is
  <https://github.com/vladkens/macmon>, MIT; it is installed by `setup.sh`
  via Homebrew and is not part of this repository.

---

## Appendix: Apache License, Version 2.0

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
